"""6번 원본 추론 검증. python test_inference.py 로 직접 실행.

- enumerate_tiles: 그리드 타일링 불변식(마지막 타일 우/하단 정렬)
- read_candidates/merge_candidates: 가짜 YOLO 라벨로 좌표 변환 + NMS 그리디 중복 제거
- save_candidate_assets: 합성 TIF로 후보 크롭/라벨/메타데이터 생성
- run(): 1->3->4처럼 실제 파이프라인 전체(진짜 infer_tiles.py + yolo26n.pt)를 합성 TIF로 돌림
"""

import json
import os
import tempfile

import numpy as np
import tifffile
from PIL import Image

import inference
from objectdb import CoordinateBox

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = os.path.join(PROJECT_ROOT, "yolo26n.pt")


def check_enumerate_tiles() -> None:
    # 640 타일, stride 512(overlap 0.2) 기준, 소스 1500x1100 -> 마지막 타일은 항상 우/하단 정렬
    tiles = list(inference.enumerate_tiles(1500, 1100, 640, 512))
    lefts = sorted({left for left, _ in tiles})
    tops = sorted({top for _, top in tiles})
    assert lefts[0] == 0 and lefts[-1] == 1500 - 640, lefts
    assert tops[0] == 0 and tops[-1] == 1100 - 640, tops
    # 소스가 타일보다 작으면 빈 결과
    assert list(inference.enumerate_tiles(300, 300, 640, 512)) == []
    print("OK: enumerate_tiles 그리드 불변식(마지막 타일 우/하단 정렬).")


def check_read_and_merge_candidates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        labels_root = os.path.join(tmp, "labels")
        os.makedirs(labels_root)

        # 타일 두 장, 겹치는 영역에 같은 객체가 서로 다른 confidence로 검출된 상황을 재현
        tile_a = inference.TileInfo("src__x0_y0_s640", "a.png", "src.tif", "src", 0, 0, 640, 640)
        tile_b = inference.TileInfo("src__x400_y0_s640", "b.png", "src.tif", "src", 400, 0, 640, 640)

        # 두 타일이 겹치는 구간(전역 x=400~640)에 같은 개체가 두 번 검출된 상황 재현:
        # 전역 중심 (525,125), 50x50 개체 -> 타일A(오프셋0,0) 로컬중심(525,125), 타일B(오프셋400,0) 로컬중심(125,125)
        with open(os.path.join(labels_root, tile_a.name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("0 0.8203125 0.1953125 0.078125 0.078125 0.9\n")  # global box (500,100,550,150), conf 0.9
        with open(os.path.join(labels_root, tile_b.name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("0 0.1953125 0.1953125 0.078125 0.078125 0.4\n")  # 같은 global box, conf 0.4 (낮음 -> merge에서 제거 대상)
        # 타일B: 전혀 안 겹치는 별개 객체, conf 0.5 -> merge 후에도 살아남아야 함
        with open(os.path.join(labels_root, tile_b.name + ".txt"), "a", encoding="utf-8") as fh:
            fh.write("0 0.8 0.8 0.05 0.05 0.5\n")

        candidates = inference.read_candidates([tile_a, tile_b], labels_root)
        assert len(candidates) == 3, len(candidates)
        for c in candidates:
            assert isinstance(c["globalBox"], CoordinateBox)

        merged = inference.merge_candidates(candidates, merge_iou=0.3)
        # 겹치는 두 개(타일A conf0.9, 타일B conf0.4) 중 높은 confidence만 남고,
        # 안 겹치는 세 번째(conf0.5)는 그대로 살아남아야 함 -> 총 2개
        assert len(merged) == 2, [ (c["confidence"], c["globalBox"]) for c in merged ]
        confidences = sorted(c["confidence"] for c in merged)
        assert confidences == [0.5, 0.9], confidences
        assert [c["candidateId"] for c in merged] == [1, 2]

    print("OK: read_candidates 좌표 변환 + merge_candidates 그리디 NMS(겹치면 최고 confidence만 유지).")


def check_save_candidate_assets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tif_path = os.path.join(tmp, "src.tif")
        tifffile.imwrite(tif_path, np.zeros((2000, 2000, 3), dtype="uint8"))

        box = CoordinateBox(900, 900, 940, 940, 40, 40)
        candidate = {
            "candidateId": 1, "sourceBaseName": "src", "sourceTifName": "src.tif", "sourceTifPath": tif_path,
            "tileName": "src__x640_y640_s640", "tileLeft": 640, "tileTop": 640,
            "classId": 0, "className": "whale", "confidence": 0.812, "globalBox": box,
        }
        run_root = os.path.join(tmp, "run")
        inference.save_candidate_assets([candidate], run_root, {"tile": "640", "candidate_view": "tile"})

        assert candidate["candidateImagePath"] and os.path.isfile(candidate["candidateImagePath"])
        with Image.open(candidate["candidateImagePath"]) as img:
            assert img.size == (640, 640)
        with open(candidate["candidateLabelPath"], encoding="utf-8") as fh:
            parts = fh.read().strip().split()
        assert parts[0] == "0"
        # tile 뷰: crop 원점은 (tileLeft, tileTop)=(640,640), 객체 global (900,900)-(940,940)
        # -> crop 로컬 (260,260)-(300,300), 40/640 크기
        assert abs(float(parts[3]) * 640 - 40) < 1e-6
        assert abs(float(parts[4]) * 640 - 40) < 1e-6
        assert os.path.isfile(candidate["candidateInfoPath"])

    print("OK: save_candidate_assets - 후보 크롭/라벨/info.json 생성 검증.")


def check_real_inference_pipeline() -> None:
    assert os.path.isfile(BASE_MODEL), f"base model not found: {BASE_MODEL}"

    with tempfile.TemporaryDirectory() as tmp:
        source_root = os.path.join(tmp, "source")
        os.makedirs(source_root, exist_ok=True)
        tifffile.imwrite(os.path.join(source_root, "00099.tif"),
                          np.zeros((900, 900, 3), dtype="uint8"))

        output_root = os.path.join(tmp, "output")
        result = inference.run(
            source_root, output_root, BASE_MODEL, "pilot_smoke",
            "tile=640, overlap=0.2, conf=0.5, iou=0.6, imgsz=640, device=cpu, batch=1, max_det=50")

        assert result.tileCount >= 1
        assert os.path.isfile(result.candidateJsonPath)
        with open(result.candidateJsonPath, encoding="utf-8") as fh:
            document = json.load(fh)
        assert document["tileCount"] == result.tileCount
        assert document["candidateCount"] == len(document["candidates"])
        assert document["sourceTifFolderPath"] == source_root
        # 빈 이미지라 탐지가 안 나올 수 있음 - candidateCount는 0 이상이면 됨(파이프라인이 안 죽는 게 핵심)
        assert document["candidateCount"] >= 0

    print("OK: 같은 프로세스 안에서 trainer/infer_tiles.py import -> 실제 ultralytics 추론 -> "
          "candidates.json 생성까지 파이프라인 전체 확인.")


if __name__ == "__main__":
    check_enumerate_tiles()
    check_read_and_merge_candidates()
    check_save_candidate_assets()
    check_real_inference_pipeline()
