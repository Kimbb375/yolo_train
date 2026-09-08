"""중앙 PC <-> 워커 PC 원격 제어 서버 (에이전트+중앙서버 방식, 2번안).

표준 라이브러리 http.server만 씀 - fastapi/uvicorn/websockets 같은 새 의존성을 추가하면
배포 zip 용량이 다시 커짐(pyside6-essentials로 줄인 맥락과 상충돼서 stdlib로 감).

프로토콜: 워커 PC의 agent.py가 먼저 이 서버로 접속(outbound, 짧은 주기 폴링)함 -> 워커 PC는
인바운드 포트/방화벽 설정이 전혀 필요 없음. 실시간성은 폴링 주기(기본 2초)만큼 지연됨 - 진짜
WebSocket 대비 트레이드오프지만 새 패키지 없이 되는 게 이득이라 판단.

엔드포인트(전부 JSON, X-Control-Token 헤더로 인증):
  POST /register  {agentId, hostname, gpu}          최초 접속 등록
  POST /unregister {agentId}                        명시적 접속 해제(즉시 오프라인 표시)
  POST /poll      {agentId, progress}                대기 중인 명령 하나 반환(없으면 null)
  POST /log       {agentId, lines: [...]}            진행 로그 append
  POST /done      {agentId, ok, message}             작업 종료 보고
  POST /command   {targetAgentId, command}           컨트롤러가 명령 큐에 적재 ("all" 가능)
  POST /upload    (raw zip bytes, X-Agent-Id/X-Run-Name 헤더)  결과 중앙 저장(mirror_root 설정 시)
  GET  /status                                       전체 에이전트 상태 스냅샷(컨트롤러 UI용)
"""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
        self._mirror_root: Optional[str] = None
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def check_token(self, token: Optional[str]) -> bool:
        return token == self._token

    def set_mirror_root(self, path: Optional[str]) -> None:
        with self._lock:
            self._mirror_root = path or None

    def receive_upload(self, agent_id: str, run_name: str, data: bytes) -> bool:
        """agent.py가 작업 완료 후(옵션 mirror=1일 때) 올린 run_root 전체 zip을 이 PC의
        중앙 저장 경로 밑에 풀어줌. ponytail: 전체를 메모리로 받음(단순함 우선) - 워커 결과
        폴더가 아주 커지면(수 GB) 청크 스트리밍으로 바꿔야 함.

        압축 해제 전에 각 항목 경로가 dest_dir 밖으로 못 나가게 검사함("zip slip" 방지 -
        agent.py가 만드는 정상 zip은 문제없지만, 토큰 유출 시 조작된 zip으로 임의 경로에
        파일을 쓰는 걸 막기 위한 방어)."""
        with self._lock:
            mirror_root = self._mirror_root
        if not mirror_root or not data:
            return False
        dest_dir = (Path(mirror_root) / agent_id / run_name).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for member in zf.infolist():
                    target = (dest_dir / member.filename).resolve()
                    if target != dest_dir and dest_dir not in target.parents:
                        raise ValueError(f"unsafe zip entry path: {member.filename}")
                zf.extractall(dest_dir)
            return True
        except (zipfile.BadZipFile, OSError, ValueError):
            return False

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

    def unregister(self, agent_id: str) -> None:
        """워커가 "접속 해제"로 스스로 끊을 때 호출됨 - lastSeen을 0으로 눌러서 snapshot()의
        online 판정(15초 타임아웃)을 기다릴 필요 없이 바로 오프라인으로 보이게 함
        (사용자 보고: 접속 해제해도 PC 목록에 계속 초록불로 남아있음)."""
        with self._lock:
            state = self._agents.get(agent_id)
            if state is not None:
                state.lastSeen = 0.0
                state.progress = "연결 해제됨"

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
                        # 컨트롤러 쪽이 이 리스트를 인덱스 기반으로 증분 표시하므로(main.py
                        # InferenceTab._poll_remote) 여기서 또 자르면 안 됨 - append_log에서
                        # 이미 LOG_TAIL_LIMIT(500)로 상한을 걸어둠.
                        "logTail": list(s.logTail),
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
                if self.path == "/upload":
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length else b""
                    ok = server.receive_upload(
                        self.headers.get("X-Agent-Id", "unknown"),
                        self.headers.get("X-Run-Name", "run"), raw)
                    self._send_json(200 if ok else 400, {"ok": ok})
                    return
                try:
                    data = self._read_json()
                    if self.path == "/register":
                        server.register(data["agentId"], data.get("hostname", ""), data.get("gpu", ""))
                        self._send_json(200, {"ok": True})
                    elif self.path == "/unregister":
                        server.unregister(data["agentId"])
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
