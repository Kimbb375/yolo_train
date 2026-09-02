"""4번 YOLO 정렬 검증: 3번(학습 타일) 산출물을 그대로 이어받아
그룹(개체 단위, 5개 zone 크롭) 분할 시 데이터 누수가 없는지 + 시드 재현성을 확인함.

python test_yolodataset.py 로 직접 실행.
"""

import glob
import json
import os
import random
import tempfile

import numpy as np
import tifffile

import trainingdataset
import yolodataset


def _build_source_dataset(tmp: str) -> str:
    capture_date = "20260608"
    source_base_name = "00099"
    source_root = os.path.join(tmp, "source")
    date_dir = os.path.join(source_root, capture_date)
    os.makedirs(date_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(date_dir, f"{source_base_name}.tif"),
                      np.zeros((1500, 2000, 3), dtype="uint8"))

    centers = [(700, 700), (1200, 700), (700, 1000)]
    objects = []
    for object_id, (cx, cy) in enumerate(centers, start=1):
        box = {"left": cx - 10, "top": cy - 10, "right": cx + 10, "bottom": cy + 10, "width": 20, "height": 20}
        objects.append({
            "objectId": object_id, "captureDate": capture_date, "sourceBaseName": source_base_name,
            "sourceTifName": f"{source_base_name}.tif", "tileId": 1, "classId": 0, "className": "whale",
            "labelPath": "dummy.txt", "labelLineNumber": 1, "localBox": box, "globalBox": box,
            "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 0.01, "height": 0.01},
        })

    object_db_path = os.path.join(tmp, "object_db.json")
    with open(object_db_path, "w", encoding="utf-8") as fh:
        json.dump({"objects": objects}, fh)

    output_root = os.path.join(tmp, "output")
    result = trainingdataset.build(object_db_path, source_root, output_root,
                                    output_sizes=[512], rng=random.Random(1))
    assert result.imageCount == 15, result.imageCount  # 개체 3 * zone 5
    return output_root


def _object_id_of_split(target_size_root: str, split: str, object_id: int) -> int:
    pattern = os.path.join(target_size_root, "images", split, f"*_obj{object_id:06d}_*.png")
    return len(glob.glob(pattern))


def check_no_group_leakage_and_determinism() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_root = _build_source_dataset(tmp)

        target1 = os.path.join(tmp, "yolo1")
        result1 = yolodataset.organize(output_root, target1, 512, 70, 15, 15, 0, seed=7)
        assert result1.groupCount == 3
        assert result1.sampleCount == 15
        assert sum(s.imageCount for s in result1.splits) == 15
        assert sum(s.groupCount for s in result1.splits) == 3

        target_size_root = os.path.join(target1, "512")
        for object_id in (1, 2, 3):
            per_split = {s.name: _object_id_of_split(target_size_root, s.name, object_id)
                         for s in result1.splits}
            nonzero = [count for count in per_split.values() if count > 0]
            assert nonzero == [5], f"object {object_id} leaked across splits: {per_split}"

        # predict 비율 0 -> predict 그룹 0개
        predict_summary = next(s for s in result1.splits if s.name == "predict")
        assert predict_summary.groupCount == 0
        assert predict_summary.imageCount == 0

        assert os.path.isfile(os.path.join(target_size_root, "dataset.yaml"))
        assert os.path.isfile(os.path.join(target_size_root, "classes.txt"))

        # 같은 seed로 다시 돌리면 그룹->split 배정이 그대로 재현돼야 함
        target2 = os.path.join(tmp, "yolo2")
        result2 = yolodataset.organize(output_root, target2, 512, 70, 15, 15, 0, seed=7)
        assignment1 = {s.name: s.groupCount for s in result1.splits}
        assignment2 = {s.name: s.groupCount for s in result2.splits}
        assert assignment1 == assignment2, (assignment1, assignment2)
        for object_id in (1, 2, 3):
            split_of_1 = next(s.name for s in result1.splits
                               if _object_id_of_split(target_size_root, s.name, object_id) == 5)
            split_of_2 = next(s.name for s in result2.splits
                               if _object_id_of_split(os.path.join(target2, "512"), s.name, object_id) == 5)
            assert split_of_1 == split_of_2, f"object {object_id}: seed 재현 안 됨 {split_of_1} != {split_of_2}"

    print("OK: 그룹(개체) 단위 분할 - 데이터 누수 없음 + seed 재현성 확인.")


if __name__ == "__main__":
    check_no_group_leakage_and_determinism()
