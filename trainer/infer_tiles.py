import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="predict")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", default="auto")
    parser.add_argument("--max_det", type=int, default=300)
    args = parser.parse_args()

    from ultralytics import YOLO

    batch = -1 if str(args.batch).lower() == "auto" else int(args.batch)
    device = None if str(args.device).lower() == "auto" else args.device
    print(f"tile_predict_source={args.source}", flush=True)
    print(f"tile_predict_project={Path(args.project) / args.name}", flush=True)
    print(
        f"tile_predict_options=imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, "
        f"batch={args.batch}, device={args.device}, max_det={args.max_det}",
        flush=True,
    )
    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        batch=batch,
        max_det=args.max_det,
        save=False,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    print(f"predict_results={Path(args.project) / args.name}")
    print(f"tiles={len(results)}")


if __name__ == "__main__":
    main()
