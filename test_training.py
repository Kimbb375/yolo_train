"""5번 학습 검증: 1->3->4 파이프라인으로 실제 합성 YOLO 데이터셋을 만들고,
trainer/train.py를 같은 프로세스 안에서 호출해서 진짜 ultralytics 학습 1 epoch을 돌림.

torch/ultralytics가 무거워서 시간 좀 걸림. python test_training.py 로 직접 실행.
"""

import json
import os
import random
import tempfile

import numpy as np
import tifffile

import trainingdataset
import training
import yolodataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = os.path.join(PROJECT_ROOT, "yolo26n.pt")


def check_parse_augmentation_options() -> None:
    args = training.parse_augmentation_options("degrees=5, translate=0.03, patience=30")
    assert args == ["--degrees", "5", "--translate", "0.03", "--patience", "30"], args
    assert training.parse_augmentation_options("") == []
    try:
        training.parse_augmentation_options("unknown_key=1")
        assert False, "should have raised"
    except ValueError:
        pass
    print("OK: parse_augmentation_options (C# ParseAugmentationOptions 포팅) 검증.")


def _build_yolo_dataset(tmp: str) -> str:
    capture_date = "20260608"
    source_base_name = "00099"
    source_root = os.path.join(tmp, "source")
    date_dir = os.path.join(source_root, capture_date)
    os.makedirs(date_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(date_dir, f"{source_base_name}.tif"),
                      np.zeros((1500, 2000, 3), dtype="uint8"))

    centers = [(300, 300), (800, 300), (300, 800), (800, 800)]
    objects = []
    for object_id, (cx, cy) in enumerate(centers, start=1):
        box = {"left": cx - 4, "top": cy - 4, "right": cx + 4, "bottom": cy + 4, "width": 8, "height": 8}
        objects.append({
            "objectId": object_id, "captureDate": capture_date, "sourceBaseName": source_base_name,
            "sourceTifName": f"{source_base_name}.tif", "tileId": 1, "classId": 0, "className": "whale",
            "labelPath": "dummy.txt", "labelLineNumber": 1, "localBox": box, "globalBox": box,
            "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 0.004, "height": 0.004},
        })

    object_db_path = os.path.join(tmp, "object_db.json")
    with open(object_db_path, "w", encoding="utf-8") as fh:
        json.dump({"objects": objects}, fh)

    output_root = os.path.join(tmp, "output")
    trainingdataset.build(object_db_path, source_root, output_root, output_sizes=[64], rng=random.Random(1))

    yolo_root = os.path.join(tmp, "yolo")
    yolodataset.organize(output_root, yolo_root, 64, train_ratio=50, validation_ratio=50,
                          test_ratio=0, predict_ratio=0, seed=1)
    return os.path.join(yolo_root, "64")


def check_real_training_smoke_test() -> None:
    assert os.path.isfile(BASE_MODEL), f"base model not found: {BASE_MODEL}"

    with tempfile.TemporaryDirectory() as tmp:
        dataset_root = _build_yolo_dataset(tmp)
        project = os.path.join(tmp, "runs")

        args = training.build_train_args(
            dataset_roots=[dataset_root], model_path=BASE_MODEL, imgsz=64, epochs=1,
            batch="2", device="cpu", project=project, name="pilot_smoke", workers=0,
            augmentation_text="degrees=0, translate=0, scale=0, fliplr=0, flipud=0, "
                               "hsv_h=0, hsv_s=0, hsv_v=0, mosaic=0, mixup=0, copy_paste=0, patience=1",
        )
        training.run(args)

        weights_dir = os.path.join(project, "pilot_smoke", "weights")
        best_pt = os.path.join(weights_dir, "best.pt")
        last_pt = os.path.join(weights_dir, "last.pt")
        assert os.path.isfile(best_pt), f"best.pt not produced: {best_pt}"
        assert os.path.isfile(last_pt), f"last.pt not produced: {last_pt}"

    print("OK: 같은 프로세스 안에서 trainer/train.py import -> 실제 ultralytics 1 epoch 학습 -> best.pt/last.pt 생성 확인.")


if __name__ == "__main__":
    check_parse_augmentation_options()
    check_real_training_smoke_test()
