"""controlserver.ControlServer 핵심 로직(인증/명령 큐/샤딩 대상/스냅샷) 검증.
실제 소켓은 안 열고(포트 충돌 없이 반복 실행 가능) 순수 상태 로직만 확인함.

python test_controlserver.py 로 직접 실행.
"""

import io
import os
import tempfile
import zipfile

import controlserver


def check_auth_and_register() -> None:
    server = controlserver.ControlServer(token="secret")
    assert server.check_token("secret")
    assert not server.check_token("wrong")
    assert not server.check_token(None)

    server.register("pc1", "PC-ONE", "RTX A4500")
    snapshot = server.snapshot()
    assert len(snapshot["agents"]) == 1
    assert snapshot["agents"][0]["agentId"] == "pc1"
    assert snapshot["agents"][0]["online"]

    print("OK: 토큰 인증 + 에이전트 등록/스냅샷 검증.")


def check_unregister_marks_offline_immediately() -> None:
    server = controlserver.ControlServer(token="secret")
    server.register("pc1", "PC-ONE", "")
    assert server.snapshot()["agents"][0]["online"]

    server.unregister("pc1")
    agent = server.snapshot()["agents"][0]
    assert not agent["online"]
    assert agent["progress"] == "연결 해제됨"

    print("OK: unregister()가 15초 타임아웃을 안 기다리고 바로 오프라인으로 표시함 검증.")


def check_command_queue_and_targeting() -> None:
    server = controlserver.ControlServer(token="secret")
    server.register("pc1", "PC-ONE", "")
    server.register("pc2", "PC-TWO", "")

    assert server.take_command("pc1") is None  # 큐 비었을 때 None

    server.queue_command("pc1", {"type": "start_job", "source": "x"})
    command = server.take_command("pc1")
    assert command is not None and command["type"] == "start_job"
    assert server.take_command("pc1") is None  # 한 번 꺼내면 소비됨

    server.queue_command("all", {"type": "start_job", "source": "y"})
    assert server.take_command("pc1")["source"] == "y"
    assert server.take_command("pc2")["source"] == "y"

    print("OK: 명령 큐(단건 소비) + all 대상 브로드캐스트 검증.")


def check_log_and_done() -> None:
    server = controlserver.ControlServer(token="secret")
    server.register("pc1", "PC-ONE", "")
    server.queue_command("pc1", {"type": "start_job"})

    server.append_log("pc1", ["line1", "line2"])
    snapshot = server.snapshot()
    agent = snapshot["agents"][0]
    assert agent["logTail"] == ["line1", "line2"]
    assert agent["currentJob"] is not None

    server.mark_done("pc1", True, "완료됨")
    snapshot = server.snapshot()
    agent = snapshot["agents"][0]
    assert agent["currentJob"] is None
    assert agent["progress"] == "완료: 완료됨"

    print("OK: 로그 append + 작업 완료 처리(currentJob 초기화) 검증.")


def _make_zip_bytes(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def check_upload_extracts_and_blocks_zip_slip() -> None:
    server = controlserver.ControlServer(token="secret")
    with tempfile.TemporaryDirectory() as tmp:
        # mirror_root 미설정이면 업로드는 그냥 무시(False)됨 - 설정 안 한 사용자에게 실수로
        # 아무 데나 안 풀리게.
        assert not server.receive_upload("pc1", "run1", _make_zip_bytes({"a.txt": "hi"}))

        server.set_mirror_root(tmp)
        good_zip = _make_zip_bytes({"candidates.json": "{}", "candidates/a.png": "x"})
        assert server.receive_upload("pc1", "run1", good_zip)
        assert os.path.isfile(os.path.join(tmp, "pc1", "run1", "candidates.json"))
        assert os.path.isfile(os.path.join(tmp, "pc1", "run1", "candidates", "a.png"))

        # 조작된 zip(경로 순회로 dest_dir 밖에 쓰려는 것)은 통째로 거부되어야 함(zip slip 방지).
        evil_zip = _make_zip_bytes({"../../evil.txt": "pwned"})
        assert not server.receive_upload("pc1", "run2", evil_zip)
        assert not os.path.isfile(os.path.join(tmp, "evil.txt"))

    print("OK: /upload 결과 압축 해제 + mirror_root 미설정시 무시 + zip slip 차단 검증.")


if __name__ == "__main__":
    check_auth_and_register()
    check_unregister_marks_offline_immediately()
    check_command_queue_and_targeting()
    check_log_and_done()
    check_upload_extracts_and_blocks_zip_slip()
