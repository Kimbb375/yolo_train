"""GPU(CUDA) torch 준비: 처음 GPU(device!=cpu)로 추론/학습을 돌릴 때, 이 앱이 실행 중인
포터블 standalone 파이썬(python/ 폴더, uv-managed CPython을 그대로 복사한 것 - venv 아님)에
실제 `pip install`로 cu126 torch/torchvision을 한 번만 설치함.

exe 하나에 CUDA torch를 통째로 번들하면 용량이 1.5~2GB+ 로 커져서 GitHub Release 2GB
제한에 걸릴 수 있음 - 그래서 배포판은 CPU torch만 들고 있는 가벼운 포터블 파이썬 폴더로
두고, GPU가 필요해지는 시점에만 그 PC에서 실제로 받게 함(C# 원본의 TrainingDataExtractor/
python/ 폴더도 같은 방식 - cu121 torch를 미리 설치해둔 포터블 파이썬이었음).

같은 프로세스에 이미 로드된 torch(CPU)는 새로 설치해도 즉시 바꿔치기 못함(네이티브 확장
모듈 핫스왑은 파이썬 자체가 지원 안 함) - 그래서 설치 후에는 이번 실행을 멈추고 앱을
재시작하라고 안내함. 재시작하면 새 프로세스가 방금 설치된 GPU torch를 그대로 불러옴.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import appversion

TORCH_VERSION = "2.13.0+cu126"
TORCHVISION_VERSION = "0.28.0+cu126"
INDEX_URL = "https://download.pytorch.org/whl/cu126"


def _marker_path() -> str:
    # sys.executable = <배포 폴더>/python/python(w).exe (포터블 standalone CPython, venv 아님)
    return os.path.join(os.path.dirname(sys.executable), "_gpu_torch_attempted.json")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch import 실패도 "사용 불가"로 취급
        return False


def _already_attempted_this_version() -> bool:
    marker = _marker_path()
    if not os.path.isfile(marker):
        return False
    try:
        with open(marker, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 - 마커 파일 손상 시 재시도
        return False
    return data.get("torchVersion") == TORCH_VERSION


def ensure_cuda_torch(log=print) -> bool:
    """CUDA torch를 바로 쓸 수 있으면 True. 없으면 설치를 시도(성공하든 실패하든) 후 항상
    False - 이번 실행에서는 못 쓰고(같은 프로세스 제약) 재시작해야 적용됨."""
    if _cuda_available():
        return True

    if not appversion.COMMIT_SHA:
        return False  # 로컬 개발(uv run) - dev venv에 자동 설치는 안 함(이미 있으면 위에서 True로 빠짐)

    if _already_attempted_this_version():
        log("[GPU] cu126 torch는 이미 설치를 시도했지만 이 PC에서 CUDA를 못 찾습니다 "
            "(GPU가 없거나 드라이버 미설치일 수 있음) - CPU로 진행합니다.")
        return False

    log(f"[GPU] CUDA torch가 없어서 설치를 시작합니다 ({INDEX_URL}, 수 분 소요될 수 있음)...")
    ok = False
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install",
             f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
             "--index-url", INDEX_URL],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log("[GPU] " + line.rstrip())
        ok = process.wait() == 0
    except Exception as exc:  # noqa: BLE001 - 설치 실패는 CPU 폴백으로 처리
        log(f"[GPU] 설치 실패: {exc}")

    try:
        with open(_marker_path(), "w", encoding="utf-8") as fh:
            json.dump({"torchVersion": TORCH_VERSION, "installed": ok}, fh)
    except OSError:
        pass

    log("[GPU] 설치 완료. 앱을 재시작해야 GPU가 적용됩니다." if ok else
        "[GPU] 설치가 실패했습니다. 위 로그를 확인하세요.")
    return False
