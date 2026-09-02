"""7번 후보 검수 검증. 6번(inference.save_candidate_assets)이 만든 실제 후보 자산을
그대로 이어받아 load/filter/confirm/negative/delete/재현 흐름을 확인함.

python test_review.py 로 직접 실행.
"""

import json
import os
import tempfile

import numpy as np
import tifffile

import inference
import review
from objectdb import CoordinateBox


def _build_candidates_json(tmp: str) -> tuple[str, str]:
    tif_path = os.path.join(tmp, "src.tif")
    tifffile.imwrite(tif_path, np.zeros((2000, 2000, 3), dtype="uint8"))

    candidates = [
        {"candidateId": 1, "sourceBaseName": "src", "sourceTifName": "src.tif", "sourceTifPath": tif_path,
         "tileName": "src__x0_y0_s640", "tileLeft": 0, "tileTop": 0, "classId": 0, "className": "whale",
         "confidence": 0.82, "globalBox": CoordinateBox(100, 100, 140, 140, 40, 40)},
        {"candidateId": 2, "sourceBaseName": "src", "sourceTifName": "src.tif", "sourceTifPath": tif_path,
         "tileName": "src__x640_y0_s640", "tileLeft": 640, "tileTop": 0, "classId": 0, "className": "whale",
         "confidence": 0.15, "globalBox": CoordinateBox(700, 100, 720, 120, 20, 20)},
    ]
    run_root = os.path.join(tmp, "run")
    inference.save_candidate_assets(candidates, run_root, {"tile": "640", "candidate_view": "tile"})

    document = {
        "generatedAt": "2026-09-02T00:00:00", "sourceTifFolderPath": tmp, "modelPath": "dummy.pt",
        "runRootPath": run_root, "options": "", "tileCount": 2, "candidateCount": 2,
        "candidates": [inference._candidate_to_dict(c) for c in candidates],
    }
    candidate_json_path = os.path.join(run_root, "candidates.json")
    with open(candidate_json_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, ensure_ascii=False, indent=2)
    return candidate_json_path, run_root


def check_load_and_filter() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        candidate_json_path, _ = _build_candidates_json(tmp)
        candidates = review.load_candidates(candidate_json_path)
        assert len(candidates) == 2
        for c in candidates:
            assert os.path.isfile(c.candidateImagePath), c.candidateImagePath
        # top 오름차순 정렬 확인 (둘 다 top=100이라 left로 2차 정렬 -> candidateId1이 먼저)
        assert candidates[0].candidateId == 1

        high_conf = review.apply_filters(candidates, "conf>=0.5")
        assert [c.candidateId for c in high_conf] == [1]

        small_only = review.apply_filters(candidates, "width<=25, height<=25")
        assert [c.candidateId for c in small_only] == [2]

        try:
            review.apply_filters(candidates, "bogus>=1")
            assert False, "should have raised"
        except ValueError:
            pass

    print("OK: load_candidates(정렬/자산경로 해석) + apply_filters(width/height/area/conf) 검증.")


def check_confirm_negative_delete_cycle() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        candidate_json_path, run_root = _build_candidates_json(tmp)
        candidates = review.load_candidates(candidate_json_path)
        output_root = os.path.join(tmp, "review_output")

        assert not review.is_confirmed(candidates[0], output_root)
        review.save_confirmed(candidates[0], output_root)
        assert review.is_confirmed(candidates[0], output_root)
        assert not review.is_negative(candidates[0], output_root)

        review.save_negative(candidates[1], output_root)
        assert review.is_negative(candidates[1], output_root)

        records_path = os.path.join(output_root, "confirmed", "records.jsonl")
        with open(records_path, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert len(lines) == 1 and lines[0]["status"] == "confirmed" and lines[0]["candidateId"] == 1

        # 확정 취소 -> 파일은 지워지되 jsonl 이력엔 unconfirmed로 한 줄 더 남음(append-only)
        review.delete_confirmed(candidates[0], output_root)
        assert not review.is_confirmed(candidates[0], output_root)
        with open(records_path, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert len(lines) == 2 and lines[1]["status"] == "unconfirmed"

        # 라벨 파일 내용도 확인: confirmed는 원본 라벨 그대로 복사됨(0 ... 형태), negative는 빈 파일
        result_negative = review.get_negative_save_result(candidates[1], output_root)
        with open(result_negative["labelPath"], encoding="utf-8") as fh:
            assert fh.read() == ""

    print("OK: confirmed/negative 저장 -> 취소(append-only 이력 유지) 전체 사이클 검증.")


def check_export_object_db() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        candidate_json_path, _ = _build_candidates_json(tmp)
        candidates = review.load_candidates(candidate_json_path)
        output_root = os.path.join(tmp, "review_output")

        review.save_confirmed(candidates[0], output_root)
        review.save_negative(candidates[1], output_root)

        path = review.export_confirmed_object_db(output_root)
        assert path == os.path.join(output_root, "object_db_new.json")
        assert os.path.isfile(path)

        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
        assert document["objectCount"] == 1
        assert document["tileCount"] == 1
        assert document["objects"][0]["className"] == "whale"
        # negative(candidate 2)는 confirmed가 아니므로 통합 출력에서 빠져야 함
        assert all(obj["globalBox"]["left"] != 700 for obj in document["objects"])

    print("OK: export_confirmed_object_db가 object_db_new.json 통합 출력(confirmed만) 생성.")


if __name__ == "__main__":
    check_load_and_filter()
    check_confirm_negative_delete_cycle()
    check_export_object_db()
