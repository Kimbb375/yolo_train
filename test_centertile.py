"""2.2번 중앙 크롭(centertile.py) 검증: 개체당 cc 크롭 1장만 나오는지 + 같이 나오는
object_db.json이 9번(labelsync.synchronize)의 '8번 기준 JSON'으로 바로 호환되는지 확인.

python test_centertile.py 로 직접 실행.
"""

import json
import os
import random
import tempfile

import numpy as np
import tifffile
from PIL import Image

import centertile
import labelsync


def _make_object(object_id: int, left: int, top: int, capture_date: str, source_base_name: str) -> dict:
    box = {"left": left, "top": top, "right": left + 40, "bottom": top + 40, "width": 40, "height": 40}
    return {
        "objectId": object_id, "captureDate": capture_date, "sourceBaseName": source_base_name,
        "sourceTifName": f"{source_base_name}.tif", "tileId": object_id, "classId": 0, "className": "whale",
        "labelPath": "dummy.txt", "labelLineNumber": 1, "localBox": box, "globalBox": box,
        "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 0.02, "height": 0.02},
    }


def check_center_only_crop_and_labelsync_compat() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        capture_date = "20260608"
        source_base_name = "00099"
        source_root = os.path.join(tmp, "source")
        date_dir = os.path.join(source_root, capture_date)
        os.makedirs(date_dir, exist_ok=True)

        source_width, source_height = 2000, 1500
        array = np.zeros((source_height, source_width, 3), dtype="uint8")
        tifffile.imwrite(os.path.join(date_dir, f"{source_base_name}.tif"), array)

        # 서로 멀리 떨어진 개체 2개 - 각자의 cc 크롭에 상대방 라벨이 섞여 들어오지 않게.
        document = {"objects": [
            _make_object(1, 900, 700, capture_date, source_base_name),
            _make_object(2, 100, 100, capture_date, source_base_name),
        ]}
        object_db_path = os.path.join(tmp, "object_db.json")
        with open(object_db_path, "w", encoding="utf-8") as fh:
            json.dump(document, fh)

        output_root = os.path.join(tmp, "output")
        result = centertile.build(object_db_path, source_root, output_root,
                                   output_sizes=[640], rng=random.Random(42))

        assert result.warnings == [], result.warnings
        assert result.imageCount == 2, result.imageCount  # 5구역이 아니라 개체당 cc 1장만
        assert result.labelFileCount == 2

        size_root = os.path.join(output_root, "640")
        images_root = os.path.join(size_root, "images")
        labels_root = os.path.join(size_root, "labels")
        assert len(os.listdir(images_root)) == 2 + 2 + 2  # (png+txt사본)x2개체 + classes/predefined
        assert len(os.listdir(labels_root)) == 2 + 1  # txt x2개체 + classes.txt

        gen_db_path = os.path.join(size_root, "object_db.json")
        with open(gen_db_path, encoding="utf-8") as fh:
            gen_document = json.load(fh)
        assert gen_document["tileCount"] == 2
        assert gen_document["objectCount"] == 2
        for tile in gen_document["tiles"]:
            assert tile["tileImageWidth"] == 640 and tile["tileImageHeight"] == 640
            assert os.path.isfile(os.path.join(size_root, tile["tileImagePath"]))
            assert os.path.isfile(os.path.join(size_root, tile["labelPath"]))

        # 9번 호환 확인 1: 아무것도 안 고치고 그대로 동기화하면 객체 수 그대로(no-op).
        noop_output = os.path.join(tmp, "noop_synced.json")
        noop_result = labelsync.synchronize(gen_db_path, labels_root, noop_output)
        assert noop_result.processedLabelFileCount == 2
        assert noop_result.synchronizedObjectCount == 2
        assert noop_result.warnings == []

        # 9번 호환 확인 2: 라벨 파일 하나에 줄 추가(=외부 툴에서 개체 추가) -> 반영되는지.
        first_tile = gen_document["tiles"][0]
        edited_label_path = os.path.join(size_root, first_tile["labelPath"])
        with open(edited_label_path, encoding="utf-8") as fh:
            original_text = fh.read()
        with open(edited_label_path, "w", encoding="utf-8") as fh:
            fh.write(original_text.rstrip("\n") + "\n0 0.2 0.2 0.05 0.05\n")

        edited_output = os.path.join(tmp, "edited_synced.json")
        edited_result = labelsync.synchronize(gen_db_path, labels_root, edited_output)
        assert edited_result.synchronizedObjectCount == 3, edited_result
        assert edited_result.addedObjectCount == 1, edited_result

    print("OK: centertile.build - 개체당 cc 크롭 1장만 생성 + 같이 나온 object_db.json이 "
          "labelsync.synchronize(9번)의 기준 JSON으로 그대로 호환됨(무편집 no-op, 편집 반영 둘 다 확인).")


if __name__ == "__main__":
    check_center_only_crop_and_labelsync_compat()
