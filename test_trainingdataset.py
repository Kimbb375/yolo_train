"""3번 학습 타일 검증: 실제 초대형 원본 TIF가 없어서(§docs/pyside6-env-setup.md 참고)
합성 TIF + object_db.json으로 5구역(tl/tr/bl/br/cc) 크롭/라벨 로직을 검증함.

python test_trainingdataset.py 로 직접 실행.
"""

import json
import os
import random
import tempfile

import numpy as np
import tifffile
from PIL import Image

import trainingdataset
from trainingdataset import _format_six


def check_format_six() -> None:
    assert _format_six(0.0) == "0"
    assert _format_six(1.0) == "1"
    assert _format_six(0.5) == "0.5"
    assert _format_six(0.123456789) == "0.123457"
    assert _format_six(0.0625) == "0.0625"
    print("OK: _format_six matches C# \"0.######\" formatting.")


def check_five_zone_crop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        capture_date = "20260608"
        source_base_name = "00099"
        source_root = os.path.join(tmp, "source")
        date_dir = os.path.join(source_root, capture_date)
        os.makedirs(date_dir, exist_ok=True)

        source_width, source_height = 2000, 1500
        array = np.zeros((source_height, source_width, 3), dtype="uint8")
        tif_path = os.path.join(date_dir, f"{source_base_name}.tif")
        tifffile.imwrite(tif_path, array)

        box = {"left": 900, "top": 700, "right": 940, "bottom": 740, "width": 40, "height": 40}
        document = {
            "objects": [{
                "objectId": 1,
                "captureDate": capture_date,
                "sourceBaseName": source_base_name,
                "sourceTifName": f"{source_base_name}.tif",
                "tileId": 1,
                "classId": 0,
                "className": "whale",
                "labelPath": "dummy.txt",
                "labelLineNumber": 1,
                "localBox": box,
                "globalBox": box,
                "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 0.02, "height": 0.02},
            }]
        }
        object_db_path = os.path.join(tmp, "object_db.json")
        with open(object_db_path, "w", encoding="utf-8") as fh:
            json.dump(document, fh)

        output_root = os.path.join(tmp, "output")
        result = trainingdataset.build(object_db_path, source_root, output_root,
                                        output_sizes=[640], rng=random.Random(42))

        assert result.warnings == [], result.warnings
        assert result.imageCount == 5, result.imageCount
        assert result.labelFileCount == 5

        size_root = os.path.join(output_root, "640")
        images_root = os.path.join(size_root, "images")
        labels_root = os.path.join(size_root, "labels")

        with open(os.path.join(size_root, "crop_metadata.json"), encoding="utf-8") as fh:
            metadata = json.load(fh)
        assert len(metadata) == 5
        assert {entry["zoneName"] for entry in metadata} == {"tl", "tr", "bl", "br", "cc"}

        for entry in metadata:
            crop = entry["cropBoxInSource"]
            assert crop["width"] == 640 and crop["height"] == 640
            assert 0 <= crop["left"] <= source_width - 640
            assert 0 <= crop["top"] <= source_height - 640
            # 640 크롭 안에 40px짜리 개체가 완전히 들어가는지 (경계 걸림 없음 확인)
            assert crop["left"] <= box["left"] and crop["right"] >= box["right"]
            assert crop["top"] <= box["top"] and crop["bottom"] >= box["bottom"]

            image_path = os.path.join(images_root, entry["baseName"] + ".png")
            with Image.open(image_path) as img:
                assert img.size == (640, 640)

            label_path = os.path.join(labels_root, entry["baseName"] + ".txt")
            with open(label_path, encoding="utf-8") as fh:
                line = fh.read().strip()
            class_id, cx, cy, w, h = line.split()
            assert class_id == "0"
            # 개체가 크롭 안에 완전히 들어가므로 라벨 width/height는 원본 40px과 정확히 일치해야 함
            assert abs(float(w) * 640 - 40) < 1e-6
            assert abs(float(h) * 640 - 40) < 1e-6

        for path in (os.path.join(size_root, "classes.txt"), os.path.join(size_root, "predefined_classes.txt"),
                     os.path.join(images_root, "classes.txt"), os.path.join(images_root, "predefined_classes.txt"),
                     os.path.join(labels_root, "classes.txt")):
            assert os.path.isfile(path), path
        assert not os.path.isfile(os.path.join(labels_root, "predefined_classes.txt"))

    print("OK: 5구역(tl/tr/bl/br/cc) 크롭/라벨/메타데이터 생성 검증 통과.")


if __name__ == "__main__":
    check_format_six()
    check_five_zone_crop()
