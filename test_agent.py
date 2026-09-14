"""agent._TeeToServer 검증 - pythonw.exe(콘솔 없음)라 sys.stdout이 None인 상태에서도
안 죽어야 함(사용자 보고: "'NoneType' object has no attribute 'write'").

python test_agent.py 로 직접 실행.
"""

import agent
import inference


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


def check_resolve_output_root() -> None:
    assert agent._resolve_output_root({"output": "C:\\explicit"}) == "C:\\explicit"

    previous = agent._output_root_override
    try:
        agent._output_root_override = "C:\\default"
        assert agent._resolve_output_root({"output": ""}) == "C:\\default"
        assert agent._resolve_output_root({}) == "C:\\default"

        agent._output_root_override = None
        try:
            agent._resolve_output_root({"output": ""})
            raise AssertionError("output도 없고 기본 경로도 없으면 실패해야 함")
        except ValueError:
            pass
    finally:
        agent._output_root_override = previous

    print("OK: 출력 폴더 결정 - 명시 output 우선, 없으면 워커 기본값, 둘 다 없으면 에러.")


def check_resolve_model_path() -> None:
    assert agent._resolve_model_path({"model": "C:\\explicit.pt"}) == "C:\\explicit.pt"

    previous = agent._model_path_override
    try:
        agent._model_path_override = "C:\\default.pt"
        assert agent._resolve_model_path({"model": ""}) == "C:\\default.pt"
        assert agent._resolve_model_path({}) == "C:\\default.pt"

        agent._model_path_override = None
        try:
            agent._resolve_model_path({"model": ""})
            raise AssertionError("model도 없고 기본 경로도 없으면 실패해야 함")
        except ValueError:
            pass
    finally:
        agent._model_path_override = previous

    print("OK: 모델 pt 경로 결정 - 명시 model 우선, 없으면 워커 기본값, 둘 다 없으면 에러.")


def check_run_job_forwards_output_and_done_locally() -> None:
    # 워커 PC가 GUI로도 떠 있으면(main.py _AgentThread) 서버 /log,/done 전송과 별개로
    # on_output/on_done 콜백을 통해 이 프로세스 안 InferenceTab에도 바로 보여줘야 함
    # (사용자 요청: "워커 PC에서도 추론 페이지처럼 작업 도는 게 보이게").
    class _FakeResult:
        runRootPath = "run_root"

        def to_display_text(self) -> str:
            return "완료 메시지"

    real_inference_run = inference.run
    real_post = agent._post

    def fake_inference_run(source, output, model, run_name, options):
        print("hello from job", flush=True)
        return _FakeResult()

    inference.run = fake_inference_run
    agent._post = lambda *a, **k: {}

    outputs: list[str] = []
    done: list[tuple] = []
    try:
        agent._run_job(
            "http://central", "tok", "agentA",
            {"type": "start_job", "source": "s", "output": "o", "model": "m", "options": ""},
            on_output=outputs.append, on_done=lambda ok, msg: done.append((ok, msg)))
    finally:
        inference.run = real_inference_run
        agent._post = real_post

    assert any("hello from job" in line for line in outputs)
    assert done == [(True, "완료 메시지")]
    print("OK: _run_job이 on_output/on_done 콜백으로 서버 전송과 별개로 로컬에도 전달함.")


if __name__ == "__main__":
    check_tee_survives_none_real_stdout()
    check_tee_still_echoes_to_real_stdout_when_present()
    check_resolve_output_root()
    check_resolve_model_path()
    check_run_job_forwards_output_and_done_locally()
