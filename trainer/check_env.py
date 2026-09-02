import json
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))


def main():
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }

    import torch
    import ultralytics

    result["torch"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    result["cuda_device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
        result["cuda_runtime"] = torch.version.cuda
    result["ultralytics"] = ultralytics.__version__
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
