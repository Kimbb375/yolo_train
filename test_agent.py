"""agent._TeeToServer 검증 - pythonw.exe(콘솔 없음)라 sys.stdout이 None인 상태에서도
안 죽어야 함(사용자 보고: "'NoneType' object has no attribute 'write'").

python test_agent.py 로 직접 실행.
"""

import agent


def check_tee_survives_none_real_stdout() -> None:
    flushed = []
    tee = agent._TeeToServer(None, flushed.append)
    tee.write("hello\n")
    tee.flush()
    assert flushed == ["hello\n"]

    print("OK: real_stdout=None이어도 write()가 안 죽고 flush_callback으로 전달됨 검증.")


def check_tee_still_echoes_to_real_stdout_when_present() -> None:
    written = []

    class _FakeStream:
        def write(self, text):
            written.append(text)

    flushed = []
    tee = agent._TeeToServer(_FakeStream(), flushed.append)
    tee.write("line1\n")
    tee.flush()
    assert written == ["line1\n"]
    assert flushed == ["line1\n"]

    print("OK: real_stdout이 있으면 그대로 echo도 되면서 flush_callback도 호출됨 검증.")


if __name__ == "__main__":
    check_tee_survives_none_real_stdout()
    check_tee_still_echoes_to_real_stdout_when_present()
