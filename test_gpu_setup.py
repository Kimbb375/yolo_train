"""GPU torch 지연 설치(gpu_setup.py) + trainerscript._wants_gpu 검증.
실제 cu126 torch를 받으면 수 GB라 여기서는 subprocess/torch import를 흉내내서
제어 흐름(마커 파일, 재시도 방지, dev 모드 스킵)만 확인함. 실제 다운로드/설치 성공
여부는 GPU 있는 실 환경에서만 확인 가능 - docs에 알려진 미검증 항목으로 남김.

python test_gpu_setup.py 로 직접 실행.
"""

import json
import os
import sys
import tempfile
import types

import appversion
import gpu_setup
import trainerscript


def check_wants_gpu() -> None:
    assert trainerscript._wants_gpu(["--source", "s", "--model", "m"]) is True
    assert trainerscript._wants_gpu(["--device", "cpu"]) is False
    assert trainerscript._wants_gpu(["--device", "CPU"]) is False
    assert trainerscript._wants_gpu(["--device", "0"]) is True
    assert trainerscript._wants_gpu(["--options", "tile=640, device=cpu, conf=0.1"]) is False
    assert trainerscript._wants_gpu(["--options", "tile=640, device=0, conf=0.1"]) is True
    print("OK: _wants_gpu - --device 플래그/--options 내장 device= 둘 다 cpu 감지.")


def check_dev_mode_skips() -> None:
    # appversion.COMMIT_SHA가 None(로컬 uv run)이면 dev venv 건드리지 않고 바로 False.
    previous = appversion.COMMIT_SHA
    appversion.COMMIT_SHA = None
    try:
        assert gpu_setup.ensure_cuda_torch(log=lambda *_: None) is False
    finally:
        appversion.COMMIT_SHA = previous
    print("OK: 로컬 개발(COMMIT_SHA 없음)에서는 GPU 설치 시도 없이 스킵.")


def check_already_cuda_available() -> None:
    previous_sha = appversion.COMMIT_SHA
    appversion.COMMIT_SHA = "deadbeef"
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    sys.modules["torch"] = fake_torch
    try:
        assert gpu_setup.ensure_cuda_torch(log=lambda *_: None) is True
    finally:
        appversion.COMMIT_SHA = previous_sha
        del sys.modules["torch"]
    print("OK: CUDA torch가 이미 있으면 설치 시도 없이 True.")


def check_install_flow_and_retry_guard() -> None:
    previous_sha = appversion.COMMIT_SHA
    previous_executable = sys.executable
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = fake_torch
    appversion.COMMIT_SHA = "deadbeef"

    logs: list[str] = []

    class _FakeProcess:
        stdout = iter(["Collecting torch==2.13.0+cu126\n", "Successfully installed torch\n"])

        def wait(self) -> int:
            return 0

    import subprocess
    real_popen = subprocess.Popen
    captured_args: list = []

    def _fake_popen(args, **k):
        captured_args.append(args)
        return _FakeProcess()

    subprocess.Popen = _fake_popen

    with tempfile.TemporaryDirectory() as tmp:
        # 배포판 레이아웃: <배포 폴더>/python/pythonw.exe (venv Scripts/ 아님 - standalone CPython 통째 복사)
        os.makedirs(os.path.join(tmp, "python"), exist_ok=True)
        sys.executable = os.path.join(tmp, "python", "pythonw.exe")
        try:
            # 1) 처음 시도: pip install(가짜) 성공 -> 마커 기록, 그래도 이번 실행은 False
            result = gpu_setup.ensure_cuda_torch(log=logs.append)
            assert result is False
            marker_path = os.path.join(tmp, "python", "_gpu_torch_attempted.json")
            assert os.path.isfile(marker_path)
            with open(marker_path, encoding="utf-8") as fh:
                marker = json.load(fh)
            assert marker == {"torchVersion": gpu_setup.TORCH_VERSION, "installed": True}
            assert any("설치를 시작합니다" in line for line in logs)
            # externally-managed(PEP 668) standalone python이라 이 플래그 없으면 pip install이
            # "error: externally-managed-environment"로 실패함 - 실제로 한 번 이걸 빼먹어서 배포에서 터짐.
            assert "--break-system-packages" in captured_args[0]
            assert any("재시작해야" in line for line in logs)

            # 2) 재시도: 여전히 CUDA 없음(가짜 torch 그대로) -> 마커 있으니 재설치 없이 바로 False
            logs.clear()
            call_count = {"n": 0}
            def _should_not_be_called(*a, **k):
                call_count["n"] += 1
                return _FakeProcess()
            subprocess.Popen = _should_not_be_called
            result2 = gpu_setup.ensure_cuda_torch(log=logs.append)
            assert result2 is False
            assert call_count["n"] == 0, "마커 있으면 재설치 시도하면 안 됨"
            assert any("CUDA를 못 찾습니다" in line for line in logs)
        finally:
            sys.executable = previous_executable
            subprocess.Popen = real_popen
            appversion.COMMIT_SHA = previous_sha
            del sys.modules["torch"]

    print("OK: 첫 설치 시도 -> 마커 기록 -> 재시도 시 중복 설치 없이 스킵.")


def check_failed_install_allows_retry() -> None:
    # installed=False(pip 실패 등)로 남은 마커는 재시도를 막으면 안 됨 - 실제로
    # --break-system-packages 빠뜨린 버그를 고친 뒤에도 옛 마커 때문에 계속 재시도
    # 안 되는 걸 막기 위한 회귀 테스트.
    previous_sha = appversion.COMMIT_SHA
    previous_executable = sys.executable
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = fake_torch
    appversion.COMMIT_SHA = "deadbeef"

    class _FakeFailingProcess:
        stdout = iter(["error: externally-managed-environment\n"])

        def wait(self) -> int:
            return 1

    import subprocess
    real_popen = subprocess.Popen
    call_count = {"n": 0}

    def _fake_popen(args, **k):
        call_count["n"] += 1
        return _FakeFailingProcess()

    subprocess.Popen = _fake_popen

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "python"), exist_ok=True)
        sys.executable = os.path.join(tmp, "python", "pythonw.exe")
        try:
            result1 = gpu_setup.ensure_cuda_torch(log=lambda *_: None)
            assert result1 is False
            assert call_count["n"] == 1

            result2 = gpu_setup.ensure_cuda_torch(log=lambda *_: None)
            assert result2 is False
            assert call_count["n"] == 2, "실패한 시도는 마커가 있어도 재시도 가능해야 함"
        finally:
            sys.executable = previous_executable
            subprocess.Popen = real_popen
            appversion.COMMIT_SHA = previous_sha
            del sys.modules["torch"]

    print("OK: 설치 실패(installed=False) 마커는 재시도를 막지 않음.")


if __name__ == "__main__":
    check_wants_gpu()
    check_dev_mode_skips()
    check_already_cuda_available()
    check_install_flow_and_retry_guard()
    check_failed_install_allows_retry()
