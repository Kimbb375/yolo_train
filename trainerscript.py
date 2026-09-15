"""trainer/*.py 스크립트를 서브프로세스 없이 같은 프로세스 안에서 import해서 실행하는 공용 헬퍼.

C# 원본은 각 스크립트를 Process.Start(python.exe, trainer/xxx.py, --args...)로 쏘았음.
trainer/*.py는 원래부터 순수 파이썬(argparse + ultralytics)이라 그대로 재사용함 (5번/6번 공용).

배포는 포터블 venv 폴더 형태(Run.bat로 .venv\\Scripts\\pythonw.exe main.py 실행)라
개발 중(uv run)과 배포판 둘 다 이 파일 위치 기준 상대 경로가 그대로 앱 루트임 -
PyInstaller onefile 시절 썼던 sys._MEIPASS 분기는 더 이상 필요 없어서 제거함
(GPU torch 크기 문제로 [[gpu_setup]] 방식으로 전환하면서 패키징 자체를 바꿈).
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import threading
from typing import Sequence

TRAINER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer")

# ultralytics(torch)를 실제로 쓰는 스크립트만 GPU 준비를 거침 (check_env/download_model 등은 불필요).
_GPU_MODULES = {"train", "infer_tiles", "infer_tif_memory", "bench_batch"}

# 추론 일시중지/재개(사용자 요청: "추론 일시중지 버튼") - 이 프로세스 안에서 한 번에 한
# 작업만 도는 전제(로컬은 BackgroundCallWorker 하나, 워커는 busy_event로 동시 실행 막음)라
# 모듈 전역 Event 하나로 충분함. main.py/module.main() 시그니처를 바꿀 필요 없이 trainer/
# infer_tif_memory.py가 배치 경계에서 이 모듈을 직접 import해서 확인하는 구조 - trainerscript는
# 이미 항상 먼저 import된 상태라(run()이 이 프로세스 안에서 모듈을 불러옴) 순환 import 문제 없음.
_RUNNING_EVENT = threading.Event()
_RUNNING_EVENT.set()  # 기본은 실행 중(일시정지 아님)


def pause() -> None:
    _RUNNING_EVENT.clear()


def resume() -> None:
    _RUNNING_EVENT.set()


def is_paused() -> bool:
    return not _RUNNING_EVENT.is_set()


def wait_if_paused() -> None:
    """일시정지 상태가 아니면 즉시 통과, 일시정지 상태면 resume()이 불릴 때까지 블로킹함.
    타일 배치 경계처럼 중단해도 안전한 지점에서 호출함 - 배치 처리 자체는 끝까지 돌고 다음
    배치 시작 전에만 멈추므로 즉시 반응은 아님(GPU 연산 중간에 끊는 건 안 함)."""
    _RUNNING_EVENT.wait()


def _wants_gpu(args: Sequence[str]) -> bool:
    """--device cpu(개별 플래그)와 --options "...device=cpu..."(옵션 문자열 내장) 둘 다 커버.
    device 관련 힌트가 전혀 없으면 각 스크립트 기본값이 GPU(0/auto)라 GPU 원하는 것으로 취급."""
    joined = " ".join(str(a) for a in args).lower()
    return not re.search(r"device[=\s]+cpu\b", joined)


def run(module_name: str, args: Sequence[str]) -> None:
    """trainer/{module_name}.py의 main()을 같은 프로세스 안에서 argv만 바꿔서 호출."""
    if module_name in _GPU_MODULES and _wants_gpu(args):
        import gpu_setup
        if not gpu_setup.ensure_cuda_torch():
            raise RuntimeError(
                "GPU torch가 아직 준비되지 않았습니다. 위 로그를 확인하고, 설치가 끝났다면 "
                "앱을 재시작한 뒤 다시 시도하세요. (device=cpu로 옵션을 바꾸면 지금 바로 CPU로 진행 가능)")

    if TRAINER_DIR not in sys.path:
        sys.path.insert(0, TRAINER_DIR)
    module = importlib.import_module(module_name)

    previous_argv = sys.argv
    sys.argv = [module_name + ".py", *args]
    try:
        module.main()
    finally:
        sys.argv = previous_argv
