"""trainer/*.py 스크립트를 서브프로세스 없이 같은 프로세스 안에서 import해서 실행하는 공용 헬퍼.

C# 원본은 각 스크립트를 Process.Start(python.exe, trainer/xxx.py, --args...)로 쏘았음.
trainer/*.py는 원래부터 순수 파이썬(argparse + ultralytics)이라 그대로 재사용함 (5번/6번 공용).

trainer/ 폴더는 pyside6_app/trainer/ 에 vendoring(복사)해서 이 앱 저장소 안에 들어있음
— PyInstaller로 패키징할 때 같이 딸려가야 하는 파일이라 바깥(C# 프로젝트 루트)에 계속
의존하면 exe에서 빠짐. app_root()는 개발 중(uv run)과 PyInstaller onefile 빌드
(sys._MEIPASS로 압축 해제됨) 둘 다 처리함.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Sequence


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # type: ignore[attr-defined]  # PyInstaller onefile 압축 해제 경로
    return os.path.dirname(os.path.abspath(__file__))


TRAINER_DIR = os.path.join(_app_root(), "trainer")


def run(module_name: str, args: Sequence[str]) -> None:
    """trainer/{module_name}.py의 main()을 같은 프로세스 안에서 argv만 바꿔서 호출."""
    if TRAINER_DIR not in sys.path:
        sys.path.insert(0, TRAINER_DIR)
    module = importlib.import_module(module_name)

    previous_argv = sys.argv
    sys.argv = [module_name + ".py", *args]
    try:
        module.main()
    finally:
        sys.argv = previous_argv
