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
from typing import Sequence

TRAINER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer")

# ultralytics(torch)를 실제로 쓰는 스크립트만 GPU 준비를 거침 (check_env/download_model 등은 불필요).
_GPU_MODULES = {"train", "infer_tiles", "infer_tif_memory"}


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
