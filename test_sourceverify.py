"""2번 원본 검증 검증. python test_sourceverify.py 로 직접 실행."""

import os
import tempfile

import numpy as np
import tifffile
from PIL import Image

import sourceverify


def _box(left, top, right, bottom):
    return {"left": left, "top": top, "right": right, "bottom": bottom,
            "width": right - left, "height": bottom - top}


def check_corrected_json_path() -> None:
    assert sourceverify.get_corrected_json_path(r"C:\data\object_db.json") == os.path.join(r"C:\data", "object_db_new.json")
    assert sourceverify.get_corrected_json_path(r"C:\data\object_db_new.json") == os.path.join(r"C:\data", "object_db_new.json")
    print("OK: get_corrected_json_path - _new 접미사 규칙(중복 안 붙음) 검증.")


def check_overlap_logic() -> None:
    records = [
        {"objectId": 1, "captureDate": "20260608", "sourceBaseName": "00021", "globalBox": _box(100, 100, 140, 140)},
        {"objectId": 2, "captureDate": "20260608", "sourceBaseName": "00021", "globalBox": _box(120, 120, 160, 160)},  # obj1과 겹침
        {"objectId": 3, "captureDate": "20260608", "sourceBaseName": "00021", "globalBox": _box(900, 900, 940, 940)},  # 안 겹침
        {"objectId": 4, "captureDate": "20260608", "sourceBaseName": "00099", "globalBox": _box(100, 100, 140, 140)},  # 다른 소스, 좌표는 같아도 무관
    ]
    overlaps_of_1 = sourceverify.find_overlapping_records(records, records[0])
    assert [r["objectId"] for r in overlaps_of_1] == [2]

    same_source = sourceverify.find_same_source_records(records, records[0], {1})
    assert [r["objectId"] for r in same_source] == [2, 3]

    counts = sourceverify.build_overlap_count_map(records)
    assert counts == {1: 1, 2: 1, 3: 0, 4: 0}
    print("OK: find_overlapping_records/find_same_source_records/build_overlap_count_map 검증.")


def check_render_preview() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tif_path = os.path.join(tmp, "src.tif")
        tifffile.imwrite(tif_path, np.zeros((1000, 1000, 3), dtype="uint8"))

        primary = {"objectId": 1, "className": "whale", "globalBox": _box(500, 500, 540, 540)}
        overlap = {"objectId": 2, "className": "whale", "globalBox": _box(520, 520, 560, 560)}
        other = {"objectId": 3, "className": "whale", "globalBox": _box(300, 300, 320, 320)}

        image, crop_bounds = sourceverify.render_preview(tif_path, primary, [overlap], [other], True, True)
        left, top, right, bottom = crop_bounds
        # 컨텍스트 180px 여유 + 소스 안쪽이라 클램프 없음
        assert (left, top, right, bottom) == (320, 320, 720, 720), crop_bounds
        assert image.size == (right - left, bottom - top)

        # 소스 경계(왼쪽위 모서리) 개체 -> 크롭이 (0,0)에서 클램프돼야 함
        edge_primary = {"objectId": 1, "className": "whale", "globalBox": _box(0, 0, 20, 20)}
        _, edge_bounds = sourceverify.render_preview(tif_path, edge_primary, [], [], True, True)
        assert edge_bounds[0] == 0 and edge_bounds[1] == 0

    print("OK: render_preview - 크롭 범위 계산(컨텍스트 180px) + 경계 클램프 검증.")


def check_box_editing() -> None:
    tile = {"tileImageWidth": 640, "tileImageHeight": 640, "tileBoxInSource": _box(1000, 2000, 1640, 2640)}
    record = {"objectId": 1, "captureDate": "20260608", "sourceBaseName": "00021", "sourceTifName": "00021.tif",
              "classId": 0, "className": "whale", "globalBox": _box(1100, 2100, 1140, 2140),
              "localBox": _box(100, 100, 140, 140), "yolo": {"classId": 0, "centerX": 0.1875, "centerY": 0.1875,
                                                              "width": 0.0625, "height": 0.0625}}

    new_global_box = _box(1200, 2200, 1250, 2250)
    updated = sourceverify.update_record_box(record, tile, new_global_box)
    assert updated["localBox"] == {"left": 200, "top": 200, "right": 250, "bottom": 250, "width": 50, "height": 50}
    expected_center_x = (200 + 25) / 640
    assert abs(updated["yolo"]["centerX"] - expected_center_x) < 1e-9

    added = sourceverify.create_added_record(record, new_global_box, sourceverify.next_object_id([record]), tile)
    assert added["objectId"] == 2
    assert added["globalBox"] == new_global_box
    assert added["classId"] == 0 and added["className"] == "whale"

    records = [record, added]
    sourceverify.insert_after(records, record["objectId"], {"objectId": 3, "note": "inserted"})
    assert [r["objectId"] for r in records] == [1, 3, 2]

    sourceverify.delete_record(records, 3)
    assert [r["objectId"] for r in records] == [1, 2]

    print("OK: update_record_box/create_added_record/insert_after/delete_record - 박스 편집 로직 검증.")


if __name__ == "__main__":
    check_corrected_json_path()
    check_overlap_logic()
    check_render_preview()
    check_box_editing()
