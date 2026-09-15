"""워커 PC측 에이전트 (2번안: 에이전트+중앙서버). GUI 없이 백그라운드로 돌며 중앙 서버로
먼저 접속(outbound)해서 명령을 받으면 inference.run()을 그대로 실행하고 진행 로그/결과를
서버로 올림. 워커 PC에 인바운드 포트를 열 필요가 없음(방화벽 문제 회피).

실행: python main.py --agent --server http://central-pc:8765 --token SECRET [--name PC1]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import socket
import string
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import gpu_setup
import inference
import trainerscript
import training

# ponytail: 2.0였다가 1.0으로 줄임 - 명령 수신/로그 반영 지연을 줄여서 체감 반응성을
# 높임. LAN 안에서 JSON 몇 바이트 주고받는 정도라 1초로 줄여도 부담 적음.
POLL_INTERVAL_SECONDS = 1.0

# 이 워커의 기본 출력 폴더/모델 pt 경로(둘 다 로컬 경로). 작업 명령에 output/model이 안
# 왔을 때 씀 - 중앙 PC가 원격 출력/모델 경로를 매번 NAS로 잡아버리면(출력: 사용자 보고
# 이미지당 12~13초가 50초로 느려짐 / 모델: 중앙 PC가 그 NAS를 직접 호스팅하면 중앙 PC의
# 디스크·네트워크 자원까지 쓰게 됨 - "중앙 PC는 컨트롤만 하고 싶다"는 사용자 요청) 때문에,
# 워커 로컬 경로를 한 번 등록해두고 재사용하려는 것. 중앙 PC가 "set_output_root"/
# "set_model_path" 명령으로 언제든 다시 지정할 수 있음.
_output_root_override: Optional[str] = None
_model_path_override: Optional[str] = None


def _agent_config_path(name: str) -> str:
    return os.path.join(os.path.dirname(sys.executable), f"_agent_{name}.json")


def _load_agent_config(name: str, key: str) -> Optional[str]:
    path = _agent_config_path(name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get(key) or None
    except Exception:  # noqa: BLE001 - 설정 파일 손상 시 미설정 취급
        return None


def _save_agent_config(name: str, key: str, value: str) -> None:
    try:
        with open(_agent_config_path(name), "w", encoding="utf-8") as fh:
            json.dump({key: value}, fh)
    except OSError:
        pass


def _load_output_root_override() -> Optional[str]:
    return _load_agent_config("output_root", "outputRoot")


def _save_output_root_override(path: str) -> None:
    _save_agent_config("output_root", "outputRoot", path)


def _load_model_path_override() -> Optional[str]:
    return _load_agent_config("model_path", "modelPath")


def _save_model_path_override(path: str) -> None:
    _save_agent_config("model_path", "modelPath", path)


def _resolve_output_root(command: dict) -> str:
    """명령에 output이 명시돼 있으면 그걸 그대로 씀(기존 동작 그대로 - 중앙이 명시적으로
    NAS 등을 지정한 경우 그대로 존중). 비어 있으면 이 워커에 등록된 기본 경로를 씀."""
    output = (command.get("output") or "").strip()
    if output:
        return output
    if _output_root_override:
        return _output_root_override
    raise ValueError("출력 폴더가 지정되지 않았고, 이 워커에도 기본 저장 경로가 설정돼 있지 않습니다.")


def _resolve_model_path(command: dict) -> str:
    """output과 같은 패턴 - 명령에 model이 명시돼 있으면 우선, 없으면 워커 기본 모델 경로."""
    model = (command.get("model") or "").strip()
    if model:
        return model
    if _model_path_override:
        return _model_path_override
    raise ValueError("모델 pt 경로가 지정되지 않았고, 이 워커에도 기본 모델 경로가 설정돼 있지 않습니다.")


def _list_dir_entries(path: str, mode: str) -> list[str]:
    """중앙 PC(RemoteBrowseDialog)가 이 워커의 실제 폴더 구조를 보고 "워커 기본 저장/모델
    경로"를 고를 수 있게 함. path가 비어 있으면 드라이브 목록(루트)을 돌려줌. 폴더는
    "이름\\"처럼 뒤에 구분자를 붙여서 파일과 구분함 - mode="pt_files"일 때만 .pt 파일도
    같이 돌려줌(모델 선택용, 출력 폴더 선택은 폴더만 필요해서 mode="folders").

    예전에 이런 실시간 원격 탐색기(list_dir 폴링)를 매 작업 시작 전 소스/모델/출력을 매번
    골라야 하는 용도로 썼다가 "연결 불안정" 문제로 뺐던 적 있음(5d1ba12). 지금은 가끔
    설정하는 기본 경로 전용이라 실패해도 다이얼로그에서 재시도하거나 수동 입력으로 바로
    대체 가능해서 그때와 부담이 다름."""
    if not path:
        return sorted(f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\"))
    names = os.listdir(path)
    entries = sorted(f"{n}\\" for n in names if os.path.isdir(os.path.join(path, n)))
    if mode == "pt_files":
        entries += sorted(n for n in names
                           if n.lower().endswith(".pt") and os.path.isfile(os.path.join(path, n)))
    return entries


def _handle_list_dir(server: str, token: str, agent_id: str, command: dict) -> None:
    request_id = command.get("requestId")
    path = command.get("path") or ""
    try:
        entries = _list_dir_entries(path, command.get("mode", "folders"))
        payload = {"agentId": agent_id, "requestId": request_id, "path": path, "entries": entries}
    except OSError as exc:
        payload = {"agentId": agent_id, "requestId": request_id, "path": path, "error": str(exc)}
    with contextlib.suppress(Exception):
        _post(server, token, "/dirlist", payload)


def _post(server: str, token: str, path: str, payload: dict, timeout: float = 10.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server.rstrip("/") + path, data=body, method="POST",
        headers={"Content-Type": "application/json", "X-Control-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class _TeeToServer(io.TextIOBase):
    """print() 출력을 실제 콘솔에도 남기고, 줄바꿈/1초 간격으로 모아 서버 /log 로도 올림.
    inference.run()은 내부에서 print(flush=True)만 쓰므로 stdout 교체만으로 충분함.

    real_stdout이 None일 수 있음 - pythonw.exe(콘솔 없음)로 뜬 GUI 프로세스는 sys.stdout이
    원래 None이고, "이 PC를 워커로 접속"으로 같은 프로세스 안에서 에이전트를 돌릴 때(아직
    BackgroundCallWorker가 한 번도 sys.stdout을 바꿔치기 안 한 상태) 그 None을 그대로
    캡처하게 됨 - None.write()를 부르면 죽으므로(사용자 보고: "'NoneType' object has no
    attribute 'write'") None이면 그냥 건너뜀(콘솔에 echo만 못 할 뿐, 서버 로그 전송엔 지장 없음)."""

    def __init__(self, real_stdout, flush_callback) -> None:
        self._real = real_stdout
        self._buffer: list[str] = []
        self._flush_callback = flush_callback
        self._last_flush = 0.0

    def write(self, text: str) -> int:
        if self._real is not None:
            self._real.write(text)
        self._buffer.append(text)
        now = time.time()
        if "\n" in text or (now - self._last_flush) > 1.0:
            self.flush()
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._flush_callback("".join(self._buffer))
            self._buffer = []
        self._last_flush = time.time()


def _upload_result(server: str, token: str, agent_id: str, run_root: str) -> None:
    """작업이 성공하고 command에 mirror=1이 있을 때 run_root 전체를 zip으로 묶어 중앙 서버로
    올림(controlserver.py의 /upload, 중앙 저장 경로가 설정돼 있으면 그 밑에 풀림). 업로드
    실패는 작업 자체 실패로 취급하지 않고 로그만 남김 - 결과는 이미 이 워커 PC 로컬에 있음."""
    run_root_path = Path(run_root)
    with tempfile.TemporaryDirectory() as tmp:
        zip_base = os.path.join(tmp, "run")
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=run_root_path)
        print(f"[Agent] 결과를 중앙으로 업로드 중... ({zip_path})", flush=True)
        with open(zip_path, "rb") as fh:
            data = fh.read()
    req = urllib.request.Request(
        server.rstrip("/") + "/upload", data=data, method="POST",
        headers={"Content-Type": "application/octet-stream", "X-Control-Token": token,
                 "X-Agent-Id": agent_id, "X-Run-Name": run_root_path.name})
    with urllib.request.urlopen(req, timeout=300) as resp:
        json.loads(resp.read().decode("utf-8"))
    print("[Agent] 업로드 완료.", flush=True)


def _run_job(server: str, token: str, agent_id: str, command: dict,
             on_output=None, on_done=None, activity_marker=None) -> None:
    """job type별로 실제 작업을 실행함. inference/training 둘 다 print()로 진행 상황을
    내보내므로 stdout을 여기서 한 번만 가로채서(_TeeToServer) 서버 /log 로 올림.

    on_output/on_done: 이 워커 PC 자체가 GUI로 떠 있는 경우(main.py의 "이 PC를 워커로
    접속"), 서버로 보내는 것과 별개로 이 프로세스 안의 InferenceTab에도 같은 로그/완료를
    바로 보여주기 위한 콜백(사용자 요청: "중앙 제어는 컨트롤만, 워커 PC에서도 추론
    페이지처럼 작업 도는 게 보이게"). CLI(--agent)로 콘솔만 띄운 경우엔 None.

    activity_marker: [timestamp] 형태의 1칸짜리 리스트 - 실제 진행 로그가 찍힐 때마다
    현재 시각으로 갱신함. run_agent의 liveness_pinger가 이걸로 "최근에 진짜 로그가 있었는지"
    판단해서, 타일 처리 로그가 이미 자주 찍히고 있는데도 10초마다 "여전히 실행 중입니다"를
    또 얹어 보내는 걸 막음(사용자 보고: "10초마다 뜨는 이 메시지는 안 띄워도 될 거 같다 -
    추론이 안 돌아가고 있을 때만 띄워달라")."""
    def send_lines(text: str) -> None:
        lines = [ln for ln in text.split("\n") if ln]
        if not lines:
            return
        if activity_marker is not None:
            activity_marker[0] = time.time()
        with contextlib.suppress(Exception):
            _post(server, token, "/log", {"agentId": agent_id, "lines": lines})
        if on_output is not None:
            for line in lines:
                on_output(line)

    real_stdout = sys.stdout
    sys.stdout = _TeeToServer(real_stdout, send_lines)
    ok, message = True, "완료"
    try:
        if command.get("type") == "start_training":
            training.run(command["args"])
            message = "학습 완료"
        else:
            result = inference.run(
                command["source"], _resolve_output_root(command), _resolve_model_path(command),
                command.get("runName"), command["options"])
            message = result.to_display_text()
            if command.get("mirror"):
                try:
                    _upload_result(server, token, agent_id, result.runRootPath)
                except Exception as exc:  # noqa: BLE001 - 업로드 실패해도 추론 자체는 성공이라 계속 보고
                    print(f"[Agent] 중앙 업로드 실패: {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001 - 실패도 서버에 보고해야 다른 PC 결과와 취합 가능
        ok = False
        message = str(exc)
    sys.stdout.flush()
    sys.stdout = real_stdout
    with contextlib.suppress(Exception):
        _post(server, token, "/done", {"agentId": agent_id, "ok": ok, "message": message})
    if on_done is not None:
        on_done(ok, message)


def run_agent(server: str, token: str, agent_id: str,
              stop_event: Optional[threading.Event] = None, log=print,
              output_root: Optional[str] = None, model_path: Optional[str] = None,
              on_job_started=None, on_job_output=None, on_job_done=None) -> None:
    """워커 루프 본체. CLI(--agent)에서도, main.py의 ControlPanel(GUI에서 "이 PC를 워커로
    접속" 버튼 -> QThread)에서도 이 함수 하나를 그대로 씀 - stop_event로 그만둘 수 있고,
    log 콜백으로 print 대신 GUI 쪽에 상태를 보낼 수 있음(기본값은 CLI 그대로 동작).

    output_root/model_path: --output-root/--model-path로 준 값이 있으면 그걸 기본값으로
    쓰고 저장함(다음 실행에도 유지). 없으면 지난번에 저장된 값을 그대로 불러옴(둘 다 없으면
    매 작업마다 중앙에서 output/model을 명시해야 함).

    on_job_started/on_job_output/on_job_done: 이 워커가 GUI로도 떠 있을 때(main.py) 실제
    작업 진행 상황을 이 프로세스 안 InferenceTab에도 그대로 보여주기 위한 콜백 - CLI로만
    쓰면 None(기본 동작 그대로)."""
    global _output_root_override, _model_path_override
    _output_root_override = output_root or _load_output_root_override()
    if output_root:
        _save_output_root_override(output_root)
    _model_path_override = model_path or _load_model_path_override()
    if model_path:
        _save_model_path_override(model_path)

    stop_event = stop_event or threading.Event()
    log(f"[Agent] {agent_id} -> {server} 접속 시도...")
    gpu_state = gpu_setup.status()
    while not stop_event.is_set():
        try:
            _post(server, token, "/register",
                  {"agentId": agent_id, "hostname": socket.gethostname(), "gpu": gpu_state,
                   "outputRoot": _output_root_override or "", "modelPath": _model_path_override or ""})
            break
        except Exception as exc:  # noqa: BLE001 - URL 오타/서버가 아직 안 뜬 상태 등 뭐든 재시도
            log(f"[Agent] 서버 연결 실패({exc}), {POLL_INTERVAL_SECONDS}초 후 재시도...")
            stop_event.wait(POLL_INTERVAL_SECONDS)
    if stop_event.is_set():
        return

    log("[Agent] 등록 완료. 명령 대기 중...")

    # ponytail: 예전엔 _run_job()을 이 폴링 루프 안에서 그대로(동기로) 불렀음 - 작업이
    # 15초(ONLINE_TIMEOUT_SECONDS)보다 오래 걸리면(대부분의 추론/학습, 특히 TensorRT engine
    # 빌드처럼 몇 분씩 걸리는 경우) 그 동안 /poll(하트비트)을 못 보내서 중앙 PC 목록에
    # 빨간불(오프라인)로 잘못 뜸(사용자 보고: 작업 중인데 끊긴 것처럼 보임). 작업을 별도
    # 스레드로 돌려서 폴링 루프는 항상 1초마다 하트비트를 계속 보내게 분리함.
    busy_event = threading.Event()
    job_started_at = [0.0]
    last_output_at = [0.0]
    liveness_stop = threading.Event()

    def run_job_in_background(command: dict) -> None:
        try:
            _run_job(server, token, agent_id, command, on_output=on_job_output, on_done=on_job_done,
                     activity_marker=last_output_at)
        finally:
            busy_event.clear()

    def liveness_pinger() -> None:
        # TensorRT engine 빌드처럼 네이티브 라이브러리가 오래 블로킹하면서 print()를 전혀
        # 안 하는 구간이 있음 - 중앙 Summary 로그에 몇 분씩 새 줄이 안 뜨면 멈춘 것처럼
        # 보임(사용자 보고). 그런 무응답 구간에서만 10초마다 "아직 실행 중" 한 줄을 올림 -
        # 원래도 그 의도였는데, 실제로는 타일 처리 로그가 이미 자주 찍히고 있어도 무조건
        # 10초마다 또 얹어 보내고 있었음(사용자 보고: "이 메시지는 실행 중이 아닐 때만
        # 띄워달라"). last_output_at(실제 진행 로그가 찍힌 마지막 시각)이 10초 넘게 안
        # 갱신됐을 때만(=진짜 조용한 구간일 때만) 보냄.
        last_ping = 0.0
        while not liveness_stop.is_set():
            now = time.time()
            if (busy_event.is_set() and now - last_output_at[0] > 10 and now - last_ping > 10):
                elapsed = int(now - job_started_at[0])
                with contextlib.suppress(Exception):
                    _post(server, token, "/log", {
                        "agentId": agent_id,
                        "lines": [f"[Agent] 작업 진행 중... (경과 {elapsed}초, 여전히 실행 중입니다)"]})
                last_ping = time.time()
            liveness_stop.wait(1.0)

    threading.Thread(target=liveness_pinger, daemon=True).start()

    poll_failed = False
    while not stop_event.is_set():
        try:
            progress = "작업 실행 중" if busy_event.is_set() else ""
            result = _post(server, token, "/poll",
                            {"agentId": agent_id, "progress": progress, "busy": busy_event.is_set(),
                             "outputRoot": _output_root_override or "", "modelPath": _model_path_override or ""})
        except Exception as exc:  # noqa: BLE001 - 위와 같은 이유로 폭넓게 잡아서 재시도
            poll_failed = True
            log(f"[Agent] 폴링 실패({exc})")
            stop_event.wait(POLL_INTERVAL_SECONDS)
            continue

        if poll_failed:
            # 중앙 PC를 껐다 켜면(재시작) controlserver.py의 ControlServer가 메모리째 새로
            # 생겨서 이 워커의 hostname/gpu/기본 경로 정보가 사라짐(heartbeat=/poll은
            # progress만 갱신, 등록 정보는 최초 /register 때만 감) - 그런데 이 워커는 계속
            # 폴링만 하고 있어서 다시 등록을 안 함. 게다가 이 워커 화면(worker_status_label)에도
            # 마지막 "폴링 실패" 문구가 그대로 남아서 실제로는 복구됐는데 끊긴 것처럼 보임
            # (사용자 보고: "중앙을 껐다 켜면 워커에서는 연결이 끊어져 보임"). 재연결
            # 성공하면 다시 /register해서 중앙 쪽 정보를 되살리고, 워커 화면 + 중앙 Summary
            # 로그(둘 다) 에 재연결/현재 상태를 남김.
            poll_failed = False
            with contextlib.suppress(Exception):
                _post(server, token, "/register",
                      {"agentId": agent_id, "hostname": socket.gethostname(), "gpu": gpu_state,
                       "outputRoot": _output_root_override or "", "modelPath": _model_path_override or ""})
            status = (f"작업 실행 중(경과 {int(time.time() - job_started_at[0])}초)"
                      if busy_event.is_set() else "대기 중")
            reconnect_msg = f"[Agent] 중앙 서버에 재연결됨 - 현재 상태: {status}"
            log(reconnect_msg)
            with contextlib.suppress(Exception):
                _post(server, token, "/log", {"agentId": agent_id, "lines": [reconnect_msg]})

        command = result.get("command")
        if command and command.get("type") == "list_dir":
            _handle_list_dir(server, token, agent_id, command)
        elif command and command.get("type") in ("start_job", "start_training"):
            label = command.get("runName") or command.get("name") or "(자동 이름)"
            log(f"[Agent] 작업 수신: {label}")
            trainerscript.resume()  # 이전 작업이 일시정지 상태로 남아있었을 수 있는 경우 방어적으로 초기화
            if on_job_started is not None:
                on_job_started(label)
            busy_event.set()
            job_started_at[0] = time.time()
            last_output_at[0] = time.time()
            threading.Thread(target=run_job_in_background, args=(command,), daemon=True).start()
        elif command and command.get("type") in ("pause_job", "resume_job"):
            # 배치 경계에서만 멈춤(trainerscript.wait_if_paused) - 현재 처리 중인 배치는
            # 끝까지 돌고 다음 배치 전에 멈춤(사용자 요청: "추론 일시중지 버튼"). 워커 화면 +
            # 중앙 Summary 로그 둘 다에 남김(재연결 알림과 같은 패턴).
            if command["type"] == "pause_job":
                trainerscript.pause()
                status_msg = "[Agent] 일시정지 요청됨 - 현재 배치 처리가 끝나면 멈춥니다."
            else:
                trainerscript.resume()
                status_msg = "[Agent] 재개됨."
            log(status_msg)
            if on_job_output is not None:
                on_job_output(status_msg)
            with contextlib.suppress(Exception):
                _post(server, token, "/log", {"agentId": agent_id, "lines": [status_msg]})
        elif command and command.get("type") == "set_output_root":
            _output_root_override = (command.get("path") or "").strip() or None
            if _output_root_override:
                _save_output_root_override(_output_root_override)
            log(f"[Agent] 기본 저장 경로 변경: {_output_root_override or '(미설정)'}")
        elif command and command.get("type") == "set_model_path":
            _model_path_override = (command.get("path") or "").strip() or None
            if _model_path_override:
                _save_model_path_override(_model_path_override)
            log(f"[Agent] 기본 모델 경로 변경: {_model_path_override or '(미설정)'}")
        stop_event.wait(POLL_INTERVAL_SECONDS)

    liveness_stop.set()
    # 사용자가 명시적으로 접속 해제한 경우 - 중앙 PC가 15초 타임아웃까지 기다리지 않고
    # 바로 오프라인으로 표시할 수 있게 알려줌(사용자 보고: 해제해도 목록에 계속 초록불).
    with contextlib.suppress(Exception):
        _post(server, token, "/unregister", {"agentId": agent_id})
    log("[Agent] 접속 해제됨.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="중앙 서버 주소, 예: http://192.168.0.10:8765")
    parser.add_argument("--token", required=True)
    parser.add_argument("--name", default=None, help="비우면 호스트명을 에이전트 ID로 씀")
    parser.add_argument("--output-root", default=None,
                         help="이 워커의 기본 저장 경로(로컬). 비우면 저장된 값을 쓰거나, "
                              "매 작업마다 중앙에서 output을 지정해야 함")
    parser.add_argument("--model-path", default=None,
                         help="이 워커의 기본 모델 pt 경로(로컬). 비우면 저장된 값을 쓰거나, "
                              "매 작업마다 중앙에서 model을 지정해야 함")
    args = parser.parse_args()
    run_agent(args.server, args.token, args.name or socket.gethostname(),
              output_root=args.output_root, model_path=args.model_path)


if __name__ == "__main__":
    main()
