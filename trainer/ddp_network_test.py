"""두 컴퓨터가 진짜 DDP로 통신되는지만 확인하는 최소 테스트 (YOLO/데이터셋 전혀 안 씀).

마스터 컴퓨터에서:
  python trainer\\torchrun_launcher.py --nnodes=2 --node-rank=0 --nproc_per_node=1 --master_addr=<마스터IP> --master_port=29500 trainer\\ddp_network_test.py

워커 컴퓨터에서 (같은 시각에):
  python trainer\\torchrun_launcher.py --nnodes=2 --node-rank=1 --nproc_per_node=1 --master_addr=<마스터IP> --master_port=29500 trainer\\ddp_network_test.py

둘 다 몇 초 안에 "OK (synced)" 줄이 뜨면 네트워크/환경은 문제 없는 것 - 문제는 YOLO 학습 쪽에 있는 것.
1~2분 넘게 아무 것도 안 뜨면 네트워크/환경 자체가 문제인 것.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ddp_windows_libuv_fix import patch_tcpstore_no_libuv

patch_tcpstore_no_libuv()

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print(f"[start] rank={rank}/{world_size}, connecting...", flush=True)
dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
print(f"[connected] rank={rank}/{world_size}", flush=True)

device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
tensor = torch.tensor([float(rank)], device=device)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
expected = world_size * (world_size - 1) / 2
status = "OK (synced)" if abs(tensor.item() - expected) < 1e-6 else "MISMATCH"
print(f"[result] rank={rank}/{world_size} sum={tensor.item():.1f} expected={expected:.1f} -> {status}", flush=True)

dist.destroy_process_group()
print("[done]", flush=True)
