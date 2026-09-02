import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--output", default=str(ROOT / "models"))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / args.model

    from ultralytics import YOLO

    cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        YOLO(args.model)
    finally:
        os.chdir(cwd)

    if not model_path.exists():
        raise FileNotFoundError(f"Model was not downloaded to {model_path}")

    print(model_path)


if __name__ == "__main__":
    main()
