"""5번 학습 — trainer/train.py를 python.exe 서브프로세스로 쏘는 대신
같은 프로세스 안에서 그대로 import해서 실행함.

C# 원본은 Process.Start(python.exe, trainer/train.py --dataset ... --model ...) 였음.
trainer/train.py는 원래부터 순수 파이썬(ultralytics 호출)이라 로직을 새로 짤 필요가 없고,
그냥 같은 인터프리터 안에서 import해서 argparse 인자만 그대로 넘기면 됨
— 이게 PySide6 전환의 핵심 이점("C#<->파이썬 프로세스 경계가 사라짐")을 실증하는 부분.
"""

from __future__ import annotations

import os
import sys
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


# ponytail: 이 파일의 나머지 함수들과 달리 여기만 서브프로세스로 띄움 — DDP는 torchrun이
# 프로세스를 시작하기 *전에* RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT를 심어줘야
# trainer/train.py의 --multinode(enable_multinode_ddp)가 동작함. 이미 이 GUI 프로세스 안에서
# ultralytics가 import돼 있으면 RANK/LOCAL_RANK가 -1로 이미 굳어 있어서(모듈 임포트 시점에
# 한 번만 읽는 전역값) 실행 중에 os.environ만 바꿔서는 못 고침 — 그래서 인프로세스 호출을
# 포기하고 실제 별도 프로세스로 띄움. 5번 학습의 다른 경로(단일 노드)는 그대로 인프로세스 유지.
# torch.distributed.run 대신 trainer/torchrun_launcher.py를 씀: 이 torch 빌드는 Windows에서
# TCPStore(use_libuv=True)가 기본값이라 그냥 torchrun을 쓰면 시작하자마자 DistStoreError로
# 죽음 (_ddp_windows_libuv_fix.py 참고). 실제 2프로세스 로컬 시뮬레이션으로 검증함.
def build_multinode_command(
    args: Sequence[str], node_count: int, node_rank: int, master_addr: str, master_port: str,
) -> list[str]:
    python_exe = os.path.join(os.path.dirname(sys.executable), "python.exe")
    launcher_path = os.path.join(trainerscript.TRAINER_DIR, "torchrun_launcher.py")
    script_path = os.path.join(trainerscript.TRAINER_DIR, "train.py")
    return [
        python_exe, launcher_path,
        "--nnodes", str(node_count),
        "--node-rank", str(node_rank),
        "--nproc_per_node", "1",
        "--master_addr", master_addr,
        "--master_port", str(master_port),
        script_path, *args, "--multinode",
    ]
