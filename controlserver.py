"""중앙 PC <-> 워커 PC 원격 제어 서버 (에이전트+중앙서버 방식, 2번안).

표준 라이브러리 http.server만 씀 - fastapi/uvicorn/websockets 같은 새 의존성을 추가하면
배포 zip 용량이 다시 커짐(pyside6-essentials로 줄인 맥락과 상충돼서 stdlib로 감).

프로토콜: 워커 PC의 agent.py가 먼저 이 서버로 접속(outbound, 짧은 주기 폴링)함 -> 워커 PC는
인바운드 포트/방화벽 설정이 전혀 필요 없음. 실시간성은 폴링 주기(기본 2초)만큼 지연됨 - 진짜
WebSocket 대비 트레이드오프지만 새 패키지 없이 되는 게 이득이라 판단.

엔드포인트(전부 JSON, X-Control-Token 헤더로 인증):
  POST /register  {agentId, hostname, gpu}          최초 접속 등록
  POST /poll      {agentId, progress}                대기 중인 명령 하나 반환(없으면 null)
  POST /log       {agentId, lines: [...]}            진행 로그 append
  POST /done      {agentId, ok, message}             작업 종료 보고
  POST /command   {targetAgentId, command}           컨트롤러가 명령 큐에 적재 ("all" 가능)
  GET  /status                                       전체 에이전트 상태 스냅샷(컨트롤러 UI용)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

ONLINE_TIMEOUT_SECONDS = 15.0
LOG_TAIL_LIMIT = 500


@dataclass
class AgentState:
    agentId: str
    hostname: str = ""
    gpu: str = ""
    lastSeen: float = 0.0
    currentJob: Optional[dict] = None
    progress: str = ""
    logTail: list = field(default_factory=list)


class ControlServer:
    """상태는 전부 메모리에만 있음 - 서버를 재시작하면 에이전트 목록이 사라짐(에이전트가 다시
    폴링하면서 자동 재등록함). ponytail: 재시작이 잦아서 문제되면 그때 파일로 저장하는 걸 추가."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._lock = threading.Lock()
        self._agents: dict[str, AgentState] = {}
        self._commands: dict[str, list[dict]] = {}
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def check_token(self, token: Optional[str]) -> bool:
        return token == self._token

    def register(self, agent_id: str, hostname: str, gpu: str) -> None:
        with self._lock:
            state = self._agents.setdefault(agent_id, AgentState(agentId=agent_id))
            state.hostname = hostname
            state.gpu = gpu
            state.lastSeen = time.time()

    def heartbeat(self, agent_id: str, progress: str) -> None:
        with self._lock:
            state = self._agents.setdefault(agent_id, AgentState(agentId=agent_id))
            state.lastSeen = time.time()
            if progress:
                state.progress = progress

    def append_log(self, agent_id: str, lines: list[str]) -> None:
        with self._lock:
            state = self._agents.get(agent_id)
            if state is None:
                return
            state.logTail.extend(lines)
            if len(state.logTail) > LOG_TAIL_LIMIT:
                state.logTail = state.logTail[-LOG_TAIL_LIMIT:]

    def mark_done(self, agent_id: str, ok: bool, message: str) -> None:
        with self._lock:
            state = self._agents.get(agent_id)
            if state is not None:
                state.currentJob = None
                state.progress = ("완료: " if ok else "실패: ") + message

    def queue_command(self, target_agent_id: str, command: dict) -> None:
        with self._lock:
            targets = list(self._agents.keys()) if target_agent_id == "all" else [target_agent_id]
            for agent_id in targets:
                self._commands.setdefault(agent_id, []).append(command)
                state = self._agents.get(agent_id)
                if state is not None and command.get("type") == "start_job":
                    state.currentJob = command
                    state.progress = "대기 중(명령 전달됨)"

    def take_command(self, agent_id: str) -> Optional[dict]:
        with self._lock:
            queue = self._commands.get(agent_id)
            return queue.pop(0) if queue else None

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                "agents": [
                    {
                        "agentId": s.agentId, "hostname": s.hostname, "gpu": s.gpu,
                        "online": (now - s.lastSeen) < ONLINE_TIMEOUT_SECONDS,
                        "currentJob": s.currentJob, "progress": s.progress,
                        "logTail": list(s.logTail[-50:]),
                    }
                    for s in self._agents.values()
                ]
            }

    def start(self, port: int) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:  # 요청마다 콘솔에 access log 안 찍음
                pass

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                return json.loads(raw or b"{}")

            def _send_json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                return server.check_token(self.headers.get("X-Control-Token"))

            def do_GET(self) -> None:  # noqa: N802 - http.server 규약
                if self.path != "/status":
                    self._send_json(404, {"error": "not found"})
                    return
                if not self._authorized():
                    self._send_json(403, {"error": "unauthorized"})
                    return
                self._send_json(200, server.snapshot())

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send_json(403, {"error": "unauthorized"})
                    return
                try:
                    data = self._read_json()
                    if self.path == "/register":
                        server.register(data["agentId"], data.get("hostname", ""), data.get("gpu", ""))
                        self._send_json(200, {"ok": True})
                    elif self.path == "/poll":
                        server.heartbeat(data["agentId"], data.get("progress", ""))
                        self._send_json(200, {"command": server.take_command(data["agentId"])})
                    elif self.path == "/log":
                        server.append_log(data["agentId"], data.get("lines", []))
                        self._send_json(200, {"ok": True})
                    elif self.path == "/done":
                        server.mark_done(data["agentId"], bool(data.get("ok")), data.get("message", ""))
                        self._send_json(200, {"ok": True})
                    elif self.path == "/command":
                        server.queue_command(data["targetAgentId"], data["command"])
                        self._send_json(200, {"ok": True})
                    else:
                        self._send_json(404, {"error": "not found"})
                except (KeyError, ValueError) as exc:
                    self._send_json(400, {"error": str(exc)})

        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
