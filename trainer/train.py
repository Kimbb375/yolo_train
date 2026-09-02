import argparse
import os
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))


def split_dataset_roots(dataset_text: str) -> list[Path]:
    return [
        Path(item.strip().strip('"'))
        for item in re.split(r"[;\n|]+", dataset_text)
        if item.strip()
    ]


def dataset_split_path(dataset_root: Path, split: str) -> str:
    image_split = dataset_root / "images" / split
    if image_split.exists():
        return str(image_split.resolve()).replace("\\", "/")
    return str((dataset_root / "images").resolve()).replace("\\", "/")


def validate_dataset_root(dataset_root: Path):
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {dataset_root}")
    if not (dataset_root / "images").exists() or not (dataset_root / "labels").exists():
        raise FileNotFoundError(f"Dataset must contain images and labels folders: {dataset_root}")


def ensure_dataset_yaml(dataset_root: Path, image_size: int) -> Path:
    dataset_root = dataset_root.resolve()
    yaml_path = dataset_root / "dataset.yaml"
    split_layout = (dataset_root / "images" / "train").exists()
    data = {
        "path": str(dataset_root).replace("\\", "/"),
        "train": "images/train" if split_layout else "images",
        "val": "images/val" if split_layout else "images",
        "names": {0: "whale"},
    }
    if (dataset_root / "images" / "test").exists():
        data["test"] = "images/test"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"dataset_yaml={yaml_path}")
    print(f"image_size={image_size}")
    return yaml_path


def ensure_mixed_dataset_yaml(dataset_roots: list[Path], image_size: int, project: Path, name: str) -> Path:
    resolved_roots = [root.resolve() for root in dataset_roots]
    yaml_root = project.resolve() / "_mixed_datasets"
    yaml_root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "mixed"
    yaml_path = yaml_root / f"{safe_name}_{image_size}.yaml"
    data = {
        "train": [dataset_split_path(root, "train") for root in resolved_roots],
        "val": [dataset_split_path(root, "val") for root in resolved_roots],
        "names": {0: "whale"},
    }
    test_paths = [str((root / "images" / "test").resolve()).replace("\\", "/") for root in resolved_roots if (root / "images" / "test").exists()]
    if test_paths:
        data["test"] = test_paths
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"dataset_yaml={yaml_path}")
    print(f"image_size={image_size}")
    print("mixed_datasets:")
    for root in resolved_roots:
        print(f"- {root}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset folder, or multiple folders separated by semicolon/newline/pipe")
    parser.add_argument("--model", default=str(ROOT / "models" / "yolo11m.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", default="auto")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 0,1")
    parser.add_argument("--project", default=str(ROOT / "data" / "output" / "runs"))
    parser.add_argument("--name", default="yolo11m_whale")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--degrees", type=float, default=10.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.5)
    parser.add_argument("--hsv_h", type=float, default=0.0)
    parser.add_argument("--hsv_s", type=float, default=0.2)
    parser.add_argument("--hsv_v", type=float, default=0.2)
    parser.add_argument("--mosaic", type=float, default=0.2)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy_paste", type=float, default=0.0)
    parser.add_argument("--patience", type=float, default=30.0)
    args = parser.parse_args()

    dataset_roots = split_dataset_roots(args.dataset)
    if not dataset_roots:
        raise FileNotFoundError("No dataset folders were provided.")
    for dataset_root in dataset_roots:
        validate_dataset_root(dataset_root)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")

    data_yaml = ensure_dataset_yaml(dataset_roots[0], args.imgsz) if len(dataset_roots) == 1 else ensure_mixed_dataset_yaml(dataset_roots, args.imgsz, Path(args.project), args.name)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    device = None if args.device.lower() == "auto" else args.device
    batch = -1 if args.batch.lower() == "auto" else int(args.batch)

    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        device=device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        patience=int(args.patience),
        exist_ok=True,
    )
    print(f"results={results.save_dir}")


if __name__ == "__main__":
    main()
