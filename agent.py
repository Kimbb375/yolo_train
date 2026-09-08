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
import training

POLL_INTERVAL_SECONDS = 2.0


def _post(server: str, token: str, path: str, payload: dict, timeout: float = 10.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server.rstrip("/") + path, data=body, method="POST",
        headers={"Content-Type": "application/json", "X-Control-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class _TeeToServer(io.TextIOBase):
    """print() 출력을 실제 콘솔에도 남기고, 줄바꿈/1초 간격으로 모아 서버 /log 로도 올림.
    inference.run()은 내부에서 print(flush=True)만 쓰므로 stdout 교체만으로 충분함."""

    def __init__(self, real_stdout, flush_callback) -> None:
        self._real = real_stdout
        self._buffer: list[str] = []
        self._flush_callback = flush_callback
        self._last_flush = 0.0

    def write(self, text: str) -> int:
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


def _list_dir(path: str):
    """중앙 PC가 이 워커 PC의 경로를 골라야 할 때(6번/5번 탭에서 원격 대상 선택 중
    "찾기..." -> RemoteBrowseDialog) 씀. path가 비어있으면 윈도우 드라이브 목록을 줌."""
    if not path:
        drives = [f"{letter}:\\" for letter in string.ascii_uppercase if os.path.exists(f"{letter}:\\")]
        return [{"name": d, "path": d, "isDir": True} for d in drives], None
    try:
        entries = []
        with os.scandir(path) as it:
            for item in it:
                try:
                    entries.append({"name": item.name, "path": item.path, "isDir": item.is_dir()})
                except OSError:
                    continue
        entries.sort(key=lambda e: (not e["isDir"], e["name"].lower()))
        return entries, None
    except OSError as exc:
        return [], str(exc)


def _handle_list_dir(server: str, token: str, agent_id: str, command: dict) -> None:
    path = command.get("path") or ""
    entries, error = _list_dir(path)
    with contextlib.suppress(Exception):
        _post(server, token, "/dir_result", {
            "agentId": agent_id, "requestId": command.get("requestId"),
            "path": path, "entries": entries, "error": error})


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
    while not stop_event.is_set():
        try:
            result = _post(server, token, "/poll", {"agentId": agent_id, "progress": ""})
        except Exception as exc:  # noqa: BLE001 - 위와 같은 이유로 폭넓게 잡아서 재시도
            log(f"[Agent] 폴링 실패({exc})")
            stop_event.wait(POLL_INTERVAL_SECONDS)
            continue

        command = result.get("command")
        if command and command.get("type") == "list_dir":
            _handle_list_dir(server, token, agent_id, command)
        elif command and command.get("type") in ("start_job", "start_training"):
            label = command.get("runName") or command.get("name") or "(자동 이름)"
            log(f"[Agent] 작업 수신: {label}")
            _run_job(server, token, agent_id, command)
        stop_event.wait(POLL_INTERVAL_SECONDS)

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
