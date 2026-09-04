"""Shared fix for a genuine PyTorch-on-Windows bug that blocks every torchrun-based multi-node
launch: TCPStore defaults to use_libuv=True, but this build's compiled extension has no libuv
support, so every rendezvous fails with DistStoreError before anything else runs. Setting the
USE_LIBUV=0 env var does NOT help - none of the call sites that matter here pass it through.

Needed in two separate processes:
- the torchrun launcher process itself (static/c10d rendezvous store), see torchrun_launcher.py
- each training worker process (its own dist.init_process_group -> env:// rendezvous store),
  applied from train.py's enable_multinode_ddp() before ultralytics calls _setup_ddp().

ponytail: torch.distributed.rendezvous is BOTH a submodule and a function re-exported into the
torch.distributed package under the same name - `import torch.distributed.rendezvous as x` silently
binds x to the function, not the submodule, so patching through it does nothing. sys.modules is the
only reliable way to reach the real submodule object.
"""
from __future__ import annotations

import functools
import sys

_RENDEZVOUS_MODULES = (
    "torch.distributed.rendezvous",
    "torch.distributed.elastic.rendezvous.static_tcp_rendezvous",
    "torch.distributed.elastic.rendezvous.c10d_rendezvous_backend",
)


def patch_tcpstore_no_libuv() -> None:
    import torch.distributed as dist

    if getattr(dist.TCPStore, "_no_libuv_patched", False):
        return

    original = dist.TCPStore

    @functools.wraps(original, updated=[])
    def patched(*args, **kwargs):
        kwargs["use_libuv"] = False
        return original(*args, **kwargs)

    patched._no_libuv_patched = True
    dist.TCPStore = patched

    for module_name in _RENDEZVOUS_MODULES:
        __import__(module_name)
        sys.modules[module_name].TCPStore = patched
