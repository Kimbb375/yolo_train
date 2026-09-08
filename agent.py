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
import training

# ponytail: 2.0였다가 1.0으로 줄임 - 명령 수신/로그 반영 지연을 줄여서 체감 반응성을
# 높임. LAN 안에서 JSON 몇 바이트 주고받는 정도라 1초로 줄여도 부담 적음.
POLL_INTERVAL_SECONDS = 1.0


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


def _run_job(server: str, token: str, agent_id: str, command: dict) -> None:
    """job type별로 실제 작업을 실행함. inference/training 둘 다 print()로 진행 상황을
    내보내므로 stdout을 여기서 한 번만 가로채서(_TeeToServer) 서버 /log 로 올림."""
    def send_lines(text: str) -> None:
        lines = [ln for ln in text.split("\n") if ln]
        if lines:
            with contextlib.suppress(Exception):
                _post(server, token, "/log", {"agentId": agent_id, "lines": lines})

    real_stdout = sys.stdout
    sys.stdout = _TeeToServer(real_stdout, send_lines)
    ok, message = True, "완료"
    try:
        if command.get("type") == "start_training":
            training.run(command["args"])
            message = "학습 완료"
        else:
            result = inference.run(
                command["source"], command["output"], command["model"],
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


def run_agent(server: str, token: str, agent_id: str,
              stop_event: Optional[threading.Event] = None, log=print) -> None:
    """워커 루프 본체. CLI(--agent)에서도, main.py의 ControlPanel(GUI에서 "이 PC를 워커로
    접속" 버튼 -> QThread)에서도 이 함수 하나를 그대로 씀 - stop_event로 그만둘 수 있고,
    log 콜백으로 print 대신 GUI 쪽에 상태를 보낼 수 있음(기본값은 CLI 그대로 동작)."""
    stop_event = stop_event or threading.Event()
    log(f"[Agent] {agent_id} -> {server} 접속 시도...")
    gpu_state = gpu_setup.status()
    while not stop_event.is_set():
        try:
            _post(server, token, "/register",
                  {"agentId": agent_id, "hostname": socket.gethostname(), "gpu": gpu_state})
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
    liveness_stop = threading.Event()

    def run_job_in_background(command: dict) -> None:
        try:
            _run_job(server, token, agent_id, command)
        finally:
            busy_event.clear()

    def liveness_pinger() -> None:
        # TensorRT engine 빌드처럼 네이티브 라이브러리가 오래 블로킹하면서 print()를 전혀
        # 안 하는 구간이 있음 - 중앙 Summary 로그에 몇 분씩 새 줄이 안 뜨면 멈춘 것처럼
        # 보임(사용자 보고). 10초마다 "아직 실행 중" 한 줄을 올려서 살아있음을 보여줌.
        last_ping = 0.0
        while not liveness_stop.is_set():
            if busy_event.is_set() and time.time() - last_ping > 10:
                elapsed = int(time.time() - job_started_at[0])
                with contextlib.suppress(Exception):
                    _post(server, token, "/log", {
                        "agentId": agent_id,
                        "lines": [f"[Agent] 작업 진행 중... (경과 {elapsed}초, 여전히 실행 중입니다)"]})
                last_ping = time.time()
            liveness_stop.wait(1.0)

    threading.Thread(target=liveness_pinger, daemon=True).start()

    while not stop_event.is_set():
        try:
            progress = "작업 실행 중" if busy_event.is_set() else ""
            result = _post(server, token, "/poll",
                            {"agentId": agent_id, "progress": progress, "busy": busy_event.is_set()})
        except Exception as exc:  # noqa: BLE001 - 위와 같은 이유로 폭넓게 잡아서 재시도
            log(f"[Agent] 폴링 실패({exc})")
            stop_event.wait(POLL_INTERVAL_SECONDS)
            continue

        command = result.get("command")
        if command and command.get("type") in ("start_job", "start_training"):
            label = command.get("runName") or command.get("name") or "(자동 이름)"
            log(f"[Agent] 작업 수신: {label}")
            busy_event.set()
            job_started_at[0] = time.time()
            threading.Thread(target=run_job_in_background, args=(command,), daemon=True).start()
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
    args = parser.parse_args()
    run_agent(args.server, args.token, args.name or socket.gethostname())


if __name__ == "__main__":
    main()
