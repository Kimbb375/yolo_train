"""torchrun (python -m torch.distributed.run) replacement for Windows.

See _ddp_windows_libuv_fix.py for why this is needed: plain `python -m torch.distributed.run`
fails immediately on this platform/build with DistStoreError before a single training line runs.
training.build_multinode_command() launches this instead of the stdlib module.
"""
from __future__ import annotations

import sys

from _ddp_windows_libuv_fix import patch_tcpstore_no_libuv

patch_tcpstore_no_libuv()

from torch.distributed.run import main  # noqa: E402 - must import after the patch above

if __name__ == "__main__":
    sys.exit(main())
