"""9. TXT 보정 반영(labelsync.synchronize) 검증.

data/d1/260608의 실제 4타일/7객체 object_db.json을 베이스로 삼고, 그중 2개 타일만
"보정 TXT 폴더"에 넣어(하나는 줄 삭제=객체 감소, 하나는 줄 추가=객체 증가) 나머지
2개 타일(TXT 없음)은 기존 객체가 그대로 유지되는지까지 확인.

python test_labelsync.py 로 직접 실행.
"""

import json
import os
import shutil
import tempfile

import labelsync
import objectdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_D1 = os.path.join(ROOT, "data", "d1")


def _tile_objects(document: dict, source_tif_name: str) -> list[dict]:
    tile_ids = [t["tileId"] for t in document["tiles"] if t["sourceTifName"] == source_tif_name]
    return [o for o in document["objects"] if o["tileId"] in tile_ids]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_labels_dir = os.path.join(tmp, "260608")
        shutil.copytree(os.path.join(DATA_D1, "260608"), raw_labels_dir)
        base_json = os.path.join(tmp, "object_db.json")
        base_result = objectdb.build(tmp, base_json)
        assert base_result.tileCount == 4 and base_result.objectCount == 7

        with open(base_json, encoding="utf-8") as fh:
            base_document = json.load(fh)
        old_00072 = _tile_objects(base_document, "00072.tif")
        old_00077 = _tile_objects(base_document, "00077.tif")
        assert len(old_00072) == 2 and len(old_00077) == 3

        # 보정 TXT 폴더: 00072는 한 줄 삭제(객체 2->1), 00077은 두 줄 추가(객체 3->5).
        # 00021/00082는 아예 안 넣어서 "TXT 없음 -> 기존 객체 유지" 분기를 검증.
        edited_dir = os.path.join(tmp, "edited")
        os.makedirs(edited_dir)
        with open(os.path.join(edited_dir, "00072_8064_1728_9343_2687.txt"), "w", encoding="utf-8") as fh:
            fh.write("15 0.069531 0.381250 0.068750 0.045833\n")
        with open(os.path.join(edited_dir, "00077_6912_5184_8191_6143.txt"), "w", encoding="utf-8") as fh:
            fh.write("15 0.668750 0.634896 0.021875 0.073958\n"
                      "15 0.721094 0.772917 0.023438 0.079167\n"
                      "15 0.392188 0.497917 0.031250 0.091667\n"
                      "0 0.5 0.5 0.05 0.05\n"
                      "0 0.2 0.2 0.05 0.05\n")

        output_json = os.path.join(tmp, "object_db_txt_synced.json")
        result = labelsync.synchronize(base_json, edited_dir, output_json)

        assert result.processedLabelFileCount == 2, result
        assert result.originalObjectCount == 7, result
        assert result.synchronizedObjectCount == 8, result  # 1(00021) + 1(00072) + 5(00077) + 1(00082)
        assert result.addedObjectCount == 1, result  # 처리된 타일 안에서 net 5(신규)-5(구) = +1
        assert result.removedObjectCount == 0, result
        assert len(result.warnings) == 2  # 00021, 00082 TXT not found
        assert any("00021" in w or "6912_21600" in w for w in result.warnings)

        with open(output_json, encoding="utf-8") as fh:
            synced = json.load(fh)
        assert synced["tileCount"] == 4
        assert synced["objectCount"] == 8

        new_00072 = _tile_objects(synced, "00072.tif")
        assert len(new_00072) == 1
        assert new_00072[0]["objectId"] == old_00072[0]["objectId"], "안 지워진 줄은 기존 objectId 유지해야 함"

        new_00077 = _tile_objects(synced, "00077.tif")
        assert len(new_00077) == 5
        assert [o["objectId"] for o in new_00077[:3]] == [o["objectId"] for o in old_00077], \
            "기존 3줄은 objectId 그대로, 새로 추가된 2줄만 새 id"
        new_ids = {o["objectId"] for o in new_00077[3:]}
        old_ids = {o["objectId"] for o in base_document["objects"]}
        assert new_ids.isdisjoint(old_ids), "새로 추가된 객체는 기존과 안 겹치는 새 objectId를 받아야 함"

        # TXT 없는 타일(00021/00082)은 기존 객체가 통째로 그대로 남아야 함.
        new_00021 = _tile_objects(synced, "00021.tif")
        old_00021 = _tile_objects(base_document, "00021.tif")
        assert new_00021 == old_00021

    print("OK: labelsync.synchronize - 보정 TXT 있는 타일은 줄 수만큼 객체 재계산(기존 objectId는 위치 기준 재사용), "
          "TXT 없는 타일은 기존 객체 그대로 유지.")


if __name__ == "__main__":
    main()
