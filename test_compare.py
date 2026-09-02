"""8번 매칭/선별 검증. python test_compare.py 로 직접 실행.

- find_best_compare_match / build_compare_rows: MATCH/MISSED/NEW 상태 계산
- load_compare_document: object_db.json 파일 vs 7번 confirmed 폴더 두 경로 다 확인
- export_selected: 체크한 개체의 실제 타일 이미지/라벨 파일까지 복사해서 새 object_db.json 생성
"""

import json
import os
import tempfile

import compare
import review
from objectdb import CoordinateBox


def _box(left, top, right, bottom):
    return {"left": left, "top": top, "right": right, "bottom": bottom,
            "width": right - left, "height": bottom - top}


def _build_left_document(base_root: str) -> dict:
    os.makedirs(base_root, exist_ok=True)
    for name in ("imgA.png", "imgA.txt", "imgB.png", "imgB.txt"):
        with open(os.path.join(base_root, name), "w", encoding="utf-8") as fh:
            fh.write("dummy")

    tiles = [
        {"tileId": 1, "captureDate": "20260608", "sourceBaseName": "00021", "sourceTifName": "00021.tif",
         "tileImagePath": "imgA.png", "labelPath": "imgA.txt",
         "tileBoxInSource": _box(0, 0, 640, 640), "tileImageWidth": 640, "tileImageHeight": 640},
        {"tileId": 2, "captureDate": "20260608", "sourceBaseName": "00072", "sourceTifName": "00072.tif",
         "tileImagePath": "imgB.png", "labelPath": "imgB.txt",
         "tileBoxInSource": _box(0, 0, 640, 640), "tileImageWidth": 640, "tileImageHeight": 640},
    ]
    objects = [
        {"objectId": 1, "captureDate": "20260608", "sourceBaseName": "00021", "sourceTifName": "00021.tif",
         "tileId": 1, "classId": 0, "className": "whale", "labelPath": "imgA.txt", "labelLineNumber": 1,
         "localBox": _box(100, 100, 140, 140), "globalBox": _box(100, 100, 140, 140),
         "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 0.1, "height": 0.1}},
        {"objectId": 2, "captureDate": "20260608", "sourceBaseName": "00072", "sourceTifName": "00072.tif",
         "tileId": 2, "classId": 0, "className": "whale", "labelPath": "imgB.txt", "labelLineNumber": 1,
         "localBox": _box(500, 500, 540, 540), "globalBox": _box(500, 500, 540, 540),
         "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 0.1, "height": 0.1}},
    ]
    return {"generatedAt": "x", "dataRoot": base_root, "tileCount": 2, "objectCount": 2,
            "excludedLabelFileCount": 0, "tiles": tiles, "objects": objects, "warnings": []}


def _build_right_document() -> dict:
    objects = [
        # 왼쪽 obj1(00021, 100,100-140,140)과 크게 겹침 -> MATCH
        {"objectId": 1, "captureDate": "20260608", "sourceBaseName": "00021", "sourceTifName": "00021.tif",
         "tileId": 1, "classId": 0, "className": "whale", "labelPath": "", "labelLineNumber": 1,
         "localBox": _box(105, 105, 145, 145), "globalBox": _box(105, 105, 145, 145),
         "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 0.1, "height": 0.1}},
        # 같은 소스(00021)지만 겹침 없음 -> NEW
        {"objectId": 2, "captureDate": "20260608", "sourceBaseName": "00021", "sourceTifName": "00021.tif",
         "tileId": 1, "classId": 0, "className": "whale", "labelPath": "", "labelLineNumber": 2,
         "localBox": _box(900, 900, 940, 940), "globalBox": _box(900, 900, 940, 940),
         "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 0.1, "height": 0.1}},
        # 왼쪽에 아예 없는 소스 -> NEW
        {"objectId": 3, "captureDate": "20260608", "sourceBaseName": "00099", "sourceTifName": "00099.tif",
         "tileId": 3, "classId": 0, "className": "whale", "labelPath": "", "labelLineNumber": 1,
         "localBox": _box(0, 0, 30, 30), "globalBox": _box(0, 0, 30, 30),
         "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 0.1, "height": 0.1}},
    ]
    return {"generatedAt": "x", "dataRoot": "", "tileCount": 0, "objectCount": 3,
            "excludedLabelFileCount": 0, "tiles": [], "objects": objects, "warnings": []}


def check_match_missed_new() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        left = _build_left_document(os.path.join(tmp, "base"))
        right = _build_right_document()
        threshold = 0.3

        left_rows = compare.build_compare_rows(left["objects"], right["objects"], True, threshold)
        right_rows = compare.build_compare_rows(right["objects"], left["objects"], False, threshold)

        left_status = {row["objectId"]: row["status"] for row in left_rows}
        assert left_status[1] == "MATCH", left_status  # 00021 obj1은 오른쪽과 겹침
        assert left_status[2] == "MISSED", left_status  # 00072 obj2는 오른쪽에 대응 없음

        right_status = {row["objectId"]: row["status"] for row in right_rows}
        assert right_status[1] == "MATCH", right_status
        assert right_status[2] == "NEW", right_status  # 같은 소스인데 겹침 없음
        assert right_status[3] == "NEW", right_status  # 소스 자체가 없음

        match_row = next(row for row in left_rows if row["objectId"] == 1)
        assert match_row["iou"] is not None and match_row["iou"] > 0.3

    print("OK: find_best_compare_match/build_compare_rows - MATCH/MISSED/NEW 상태 계산 검증.")


def check_load_compare_document_both_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        left = _build_left_document(os.path.join(tmp, "base"))
        left_json_path = os.path.join(tmp, "object_db.json")
        with open(left_json_path, "w", encoding="utf-8") as fh:
            json.dump(left, fh)

        loaded_from_file = compare.load_compare_document(left_json_path)
        assert loaded_from_file["objectCount"] == 2

        # 7번 confirmed 폴더 경로도 확인 (review.save_confirmed로 실제 만들어서)
        import inference
        tif_path = os.path.join(tmp, "src.tif")
        import numpy as np, tifffile
        tifffile.imwrite(tif_path, np.zeros((1000, 1000, 3), dtype="uint8"))
        candidate_dict = [{"candidateId": 1, "sourceBaseName": "src", "sourceTifName": "src.tif",
                            "sourceTifPath": tif_path, "tileName": "src__x0_y0_s640", "tileLeft": 0, "tileTop": 0,
                            "classId": 0, "className": "whale", "confidence": 0.9,
                            "globalBox": CoordinateBox(10, 10, 50, 50, 40, 40)}]
        run_root = os.path.join(tmp, "run")
        inference.save_candidate_assets(candidate_dict, run_root, {"tile": "640"})
        rc = review.ReviewCandidate(1, "src", "src.tif", tif_path, "src__x0_y0_s640", 0.9,
                                     CoordinateBox(10, 10, 50, 50, 40, 40),
                                     CoordinateBox(**candidate_dict[0]["candidateCropBox"]),
                                     candidate_dict[0]["candidateImagePath"], candidate_dict[0]["candidateLabelPath"],
                                     candidate_dict[0]["candidateInfoPath"])
        review_output_root = os.path.join(tmp, "review_out")
        review.save_confirmed(rc, review_output_root)

        loaded_from_folder = compare.load_compare_document(review_output_root)
        assert loaded_from_folder["objectCount"] == 1
        assert loaded_from_folder["objects"][0]["sourceTifName"] == "src.tif"

    print("OK: load_compare_document - object_db.json 파일 경로 / 7번 confirmed 폴더 경로 둘 다 확인.")


def check_export_selected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        left = _build_left_document(os.path.join(tmp, "base"))
        selected = [left["objects"][0]]  # obj1(00021)만 체크

        output_json_path = os.path.join(tmp, "export", "object_db_selected.json")
        result_path = compare.export_selected(left, selected, output_json_path)
        assert result_path == output_json_path
        with open(output_json_path, encoding="utf-8") as fh:
            exported = json.load(fh)

        assert exported["objectCount"] == 1
        assert exported["objects"][0]["objectId"] == 1  # 새로 1부터 재번호
        assert exported["objects"][0]["sourceBaseName"] == "00021"
        exported_image = os.path.join(os.path.dirname(output_json_path), exported["tiles"][0]["tileImagePath"])
        exported_label = os.path.join(os.path.dirname(output_json_path), exported["objects"][0]["labelPath"])
        assert os.path.isfile(exported_image), exported_image  # 실제 타일 이미지가 복사됐는지
        assert os.path.isfile(exported_label), exported_label

    print("OK: export_selected - 체크한 개체의 타일 이미지/라벨 실제 복사 + object_db.json 재번호 검증.")


if __name__ == "__main__":
    check_match_missed_new()
    check_load_compare_document_both_paths()
    check_export_selected()
