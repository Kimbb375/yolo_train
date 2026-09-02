"""5번 학습 — trainer/train.py를 python.exe 서브프로세스로 쏘는 대신
같은 프로세스 안에서 그대로 import해서 실행함.

C# 원본은 Process.Start(python.exe, trainer/train.py --dataset ... --model ...) 였음.
trainer/train.py는 원래부터 순수 파이썬(ultralytics 호출)이라 로직을 새로 짤 필요가 없고,
그냥 같은 인터프리터 안에서 import해서 argparse 인자만 그대로 넘기면 됨
— 이게 PySide6 전환의 핵심 이점("C#<->파이썬 프로세스 경계가 사라짐")을 실증하는 부분.
"""

from __future__ import annotations

from typing import Sequence

import trainerscript

AUGMENTATION_KEYS = (
    "degrees", "translate", "scale", "fliplr", "flipud",
    "hsv_h", "hsv_s", "hsv_v", "mosaic", "mixup", "copy_paste", "patience",
)


def parse_augmentation_options(text: str) -> list[str]:
    """C# ParseAugmentationOptions 포팅. "degrees=5, translate=0.03, ..." -> ["--degrees","5",...]"""
    args: list[str] = []
    if not text or not text.strip():
        return args

    allowed = {key.lower() for key in AUGMENTATION_KEYS}
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid augmentation option: {item}")
        key, value = item.split("=", 1)
        key, value = key.strip().lstrip("-"), value.strip()
        if not key or not value:
            raise ValueError(f"Invalid augmentation option: {item}")
        if key.lower() not in allowed:
            raise ValueError(f"Unknown augmentation option: {key}")
        try:
            float(value)
        except ValueError as exc:
            raise ValueError(f"Augmentation option value must be numeric: {key}={value}") from exc
        args.append("--" + key)
        args.append(value)
    return args


def build_train_args(
    dataset_roots: Sequence[str], model_path: str, imgsz: int, epochs: int, batch: str,
    device: str, project: str, name: str, workers: int, augmentation_text: str,
) -> list[str]:
    args = [
        "--dataset", ";".join(dataset_roots),
        "--model", model_path,
        "--imgsz", str(imgsz),
        "--epochs", str(epochs),
        "--batch", batch,
        "--device", device or "auto",
        "--project", project,
        "--name", name or "yolo_whale",
        "--workers", str(workers),
    ]
    args.extend(parse_augmentation_options(augmentation_text))
    return args


def run(args: Sequence[str]) -> None:
    """trainer/train.py의 main()을 같은 프로세스 안에서 호출함 (서브프로세스 없음)."""
    trainerscript.run("train", args)
