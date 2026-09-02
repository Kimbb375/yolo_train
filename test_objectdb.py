"""회귀 검증: data/d1 -> C# 버전이 만든 data/object_db.json과 필드 단위로 일치하는지 대조.

python test_objectdb.py 로 직접 실행.
"""

import json
import os
import shutil
import tempfile

import objectdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_D1 = os.path.join(ROOT, "data", "d1")
EXPECTED_JSON = os.path.join(ROOT, "data", "object_db.json")


def strip_volatile(document: dict) -> dict:
    document = dict(document)
    document.pop("generatedAt", None)
    document.pop("dataRoot", None)  # absolute path, environment-dependent
    return document


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # object_db.json 원본은 data/d1 밑에 260608 원본 라벨만 있을 때 생성됨.
        # 지금 data/d1에는 3번 학습 타일 산출물(output/)이 나중에 추가돼 있어서
        # 원본 라벨 폴더(260608)만 복사해 그때 상태를 재현함.
        raw_labels_dir = os.path.join(tmp, "260608")
        shutil.copytree(os.path.join(DATA_D1, "260608"), raw_labels_dir)

        output_path = os.path.join(tmp, "object_db.json")
        result = objectdb.build(tmp, output_path)

        with open(output_path, encoding="utf-8") as fh:
            actual = strip_volatile(json.load(fh))
        with open(EXPECTED_JSON, encoding="utf-8") as fh:
            expected = strip_volatile(json.load(fh))

        assert actual == expected, (
            "object_db.json 불일치.\n"
            f"actual tiles/objects: {result.tileCount}/{result.objectCount}\n"
            f"actual:   {json.dumps(actual, ensure_ascii=False)[:2000]}\n"
            f"expected: {json.dumps(expected, ensure_ascii=False)[:2000]}"
        )
        assert result.tileCount == 4
        assert result.objectCount == 7
        assert result.excludedLabelFileCount == 0
        assert result.warnings == []

    print("OK: PySide6 objectdb.build() output matches C# ObjectDbBuilder output exactly.")


if __name__ == "__main__":
    main()
