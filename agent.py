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
import socket
import sys
import time
import urllib.error
import urllib.request

import gpu_setup
import inference

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


def _run_job(server: str, token: str, agent_id: str, command: dict) -> None:
    def send_lines(text: str) -> None:
        lines = [ln for ln in text.split("\n") if ln]
        if lines:
            with contextlib.suppress(Exception):
                _post(server, token, "/log", {"agentId": agent_id, "lines": lines})

    real_stdout = sys.stdout
    sys.stdout = _TeeToServer(real_stdout, send_lines)
    ok, message = True, "완료"
    try:
        result = inference.run(
            command["source"], command["output"], command["model"],
            command.get("runName"), command["options"])
        message = result.to_display_text()
    except Exception as exc:  # noqa: BLE001 - 실패도 서버에 보고해야 다른 PC 결과와 취합 가능
        ok = False
        message = str(exc)
    finally:
        sys.stdout.flush()
        sys.stdout = real_stdout
    with contextlib.suppress(Exception):
        _post(server, token, "/done", {"agentId": agent_id, "ok": ok, "message": message})


def run_agent(server: str, token: str, agent_id: str) -> None:
    print(f"[Agent] {agent_id} -> {server} 접속 시도...")
    gpu_state = gpu_setup.status()
    while True:
        try:
            _post(server, token, "/register",
                  {"agentId": agent_id, "hostname": socket.gethostname(), "gpu": gpu_state})
            break
        except (urllib.error.URLError, OSError) as exc:
            print(f"[Agent] 서버 연결 실패({exc}), {POLL_INTERVAL_SECONDS}초 후 재시도...")
            time.sleep(POLL_INTERVAL_SECONDS)

    print("[Agent] 등록 완료. 명령 대기 중...")
    while True:
        try:
            result = _post(server, token, "/poll", {"agentId": agent_id, "progress": ""})
        except (urllib.error.URLError, OSError) as exc:
            print(f"[Agent] 폴링 실패({exc})")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        command = result.get("command")
        if command and command.get("type") == "start_job":
            print(f"[Agent] 작업 수신: {command.get('runName') or '(자동 이름)'}")
            _run_job(server, token, agent_id, command)
        time.sleep(POLL_INTERVAL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="중앙 서버 주소, 예: http://192.168.0.10:8765")
    parser.add_argument("--token", required=True)
    parser.add_argument("--name", default=None, help="비우면 호스트명을 에이전트 ID로 씀")
    args = parser.parse_args()
    run_agent(args.server, args.token, args.name or socket.gethostname())


if __name__ == "__main__":
    main()
