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

import importlib.util
import json
import os
import subprocess
import sys

import appversion

TORCH_VERSION = "2.13.0+cu126"
TORCHVISION_VERSION = "0.28.0+cu126"
INDEX_URL = "https://download.pytorch.org/whl/cu126"

# pip 패키지명 -> import 모듈명 (onnxruntime-gpu는 import 시 onnxruntime).
_TENSORRT_PACKAGES = {"onnx": "onnx", "onnxslim": "onnxslim",
                      "onnxruntime-gpu": "onnxruntime", "tensorrt": "tensorrt"}


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
    # installed=False(pip 실패 등)면 재시도 허용 - 성공적으로 설치까지 끝났는데 이 PC에
    # CUDA가 없는 경우만 재시도 안 함(매번 몇 분씩 재다운로드하는 걸 막기 위함).
    return data.get("torchVersion") == TORCH_VERSION and data.get("installed") is True


def status() -> str:
    """부수효과 없이 현재 상태만 확인(설치 시도 안 함) - 6번 탭 진입 시 표시용.
    "available": CUDA 바로 사용 가능. "unavailable": 설치까지 끝났는데 이 PC에 CUDA 없음.
    "not_installed": 아직 설치 안 함(버튼으로 설치 가능)."""
    if _cuda_available():
        return "available"
    if _already_attempted_this_version():
        return "unavailable"
    return "not_installed"


def ensure_cuda_torch(log=print, force: bool = False) -> bool:
    """CUDA torch를 바로 쓸 수 있으면 True. 없으면 설치를 시도(성공하든 실패하든) 후 항상
    False - 이번 실행에서는 못 쓰고(같은 프로세스 제약) 재시작해야 적용됨.

    force=True: 이전에 설치 성공 기록이 있어도(예: 드라이버 업데이트 후 재시도) 다시 pip
    install을 돎 - GPU 설치 버튼을 사용자가 직접 누른 경우에만 씀."""
    if _cuda_available():
        return True

    if not appversion.COMMIT_SHA:
        return False  # 로컬 개발(uv run) - dev venv에 자동 설치는 안 함(이미 있으면 위에서 True로 빠짐)

    if not force and _already_attempted_this_version():
        log("[GPU] cu126 torch는 이미 설치를 시도했지만 이 PC에서 CUDA를 못 찾습니다 "
            "(GPU가 없거나 드라이버 미설치일 수 있음) - CPU로 진행합니다.")
        return False

    log(f"[GPU] CUDA torch가 없어서 설치를 시작합니다 ({INDEX_URL}, 수 분 소요될 수 있음)...")
    ok = False
    try:
        # --force-reinstall 없으면 pip이 "torch==2.13.0+cu126이 이미 설치돼 있음"으로 보고
        # 아무것도 안 하고 성공 처리해버림 - 재설치 버튼을 눌러도(설치는 이미 한 번
        # installed=True로 기록된 상태) 실제로는 재다운로드가 전혀 안 일어나는 버그가 있었음
        # (사용자 보고: "재설치 성공했다는데 재시작해도 여전히 CUDA 못 찾음"). --no-deps를
        # 같이 줘서 torch/torchvision 본체만 다시 받고, numpy 등 의존 패키지는 안 건드림
        # (--index-url이 일반 PyPI가 아니라 저 의존 패키지들을 못 찾아서 실패할 수 있음).
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "--break-system-packages",
             "--force-reinstall", "--no-deps", "--no-cache-dir",
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


def _tensorrt_marker_path() -> str:
    return os.path.join(os.path.dirname(sys.executable), "_tensorrt_attempted.json")


def _tensorrt_missing_packages() -> list[str]:
    return [pkg for pkg, module in _TENSORRT_PACKAGES.items() if importlib.util.find_spec(module) is None]


def _tensorrt_already_attempted() -> bool:
    marker = _tensorrt_marker_path()
    if not os.path.isfile(marker):
        return False
    try:
        with open(marker, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 - 마커 파일 손상 시 재시도
        return False
    return data.get("installed") is True


def tensorrt_status() -> str:
    """"available": onnx/onnxslim/onnxruntime-gpu/tensorrt 전부 설치되어 6-1번 TensorRT
    engine 변환이 바로 가능. "unavailable": 설치를 시도했지만 실패로 끝남(재시도 가능).
    "not_installed": 아직 설치 안 함."""
    if not _tensorrt_missing_packages():
        return "available"
    if _tensorrt_already_attempted():
        return "unavailable"
    return "not_installed"


def ensure_tensorrt(log=print, force: bool = False) -> bool:
    """TensorRT engine 변환에 필요한 패키지들을 미리 설치. GPU torch(cu126)와 달리 이번
    프로세스가 아직 import한 적 없는 모듈들이라 설치 성공하면 재시작 없이 바로 쓸 수 있음
    (ultralytics의 자체 자동설치는 --break-system-packages 없이 pip을 불러서 uv가 관리하는
    포터블 python에서 항상 externally-managed-environment로 실패하기 때문에 직접 설치함)."""
    missing = _tensorrt_missing_packages()
    if not missing:
        return True

    if not force and _tensorrt_already_attempted():
        log("[TensorRT] 이전에 설치를 시도했지만 실패했습니다. 네트워크 상태를 확인하고 다시 시도하세요.")
        return False

    log(f"[TensorRT] 설치 시작: {', '.join(missing)} (수 분 소요될 수 있음)...")
    ok = False
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", *missing],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log("[TensorRT] " + line.rstrip())
        ok = process.wait() == 0
    except Exception as exc:  # noqa: BLE001 - 설치 실패는 .pt 폴백으로 처리
        log(f"[TensorRT] 설치 실패: {exc}")

    try:
        with open(_tensorrt_marker_path(), "w", encoding="utf-8") as fh:
            json.dump({"installed": ok}, fh)
    except OSError:
        pass

    log("[TensorRT] 설치 완료. 바로 사용할 수 있습니다." if ok else
        "[TensorRT] 설치가 실패했습니다. 위 로그를 확인하세요.")
    return ok
