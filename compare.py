"""8번 매칭/선별: 왼쪽 기준 데이터 vs 오른쪽 검수 결과를 좌우로 비교하고,
체크한 항목만 새 object_db.json으로 export.

C# Form1.cs의 LoadDualCompareButton_Click / PopulateCompareGrid / FindBestCompareMatch /
ExportSelectedCompareRecords 포팅 (DataGridView 대신 그리드는 GUI 레이어(QTableWidget)에서 구성).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from typing import Optional

import review
from objectdb import CoordinateBox

_DIGIT_RUN = re.compile(r"(\d+)")


def load_compare_document(path: str) -> dict:
    """C# LoadCompareObjectDocument 포팅: 파일이면 object_db.json, 폴더면 7번 confirmed 결과."""
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    if os.path.isdir(path):
        return review.load_confirmed_object_db_document(path)
    raise FileNotFoundError(f"Compare input was not found: {path}")


def _iou(a: dict, b: dict) -> float:
    left, top = max(a["left"], b["left"]), max(a["top"], b["top"])
    right, bottom = min(a["right"], b["right"]), min(a["bottom"], b["bottom"])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return 0.0 if union <= 0 else intersection / union


def _is_same_source(first: dict, second: dict) -> bool:
    return (first["captureDate"].lower() == second["captureDate"].lower() and
            (first["sourceTifName"].lower() == second["sourceTifName"].lower() or
             first["sourceBaseName"].lower() == second["sourceBaseName"].lower()))


def find_best_compare_match(record: dict, candidates: list[dict]) -> tuple[Optional[dict], float]:
    """C# FindBestCompareMatch 포팅: 같은 소스(날짜+영상명/베이스명) 중 IoU 최고."""
    best, best_iou = None, 0.0
    for candidate in candidates:
        if not _is_same_source(record, candidate):
            continue
        iou = _iou(record["globalBox"], candidate["globalBox"])
        if best is None or iou > best_iou:
            best, best_iou = candidate, iou
    return best, best_iou


def _natural_key(value: str):
    """C# CompareNatural 근사 포팅: 숫자 구간은 자릿수 무시하고 크기로, 문자열은 대소문자 무시하고 비교."""
    parts = _DIGIT_RUN.split(value)
    key = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            stripped = part.lstrip("0") or "0"
            key.append((1, len(stripped), stripped))
        else:
            key.append((0, part.upper()))
    return key


def _compare_sort_key(record: dict):
    image_stem = os.path.splitext(record["sourceTifName"])[0]
    box = record["globalBox"]
    return ((record["captureDate"] or "").strip().lower(), _natural_key(image_stem), box["top"], box["left"])


def build_compare_rows(records: list[dict], opposite_records: list[dict], is_base_side: bool,
                        threshold: float) -> list[dict]:
    """C# PopulateCompareGrid 포팅(그리드 자체는 GUI에서 구성, 여기서는 정렬/상태 계산까지)."""
    rows = []
    for record in sorted(records, key=_compare_sort_key):
        match_record, iou = find_best_compare_match(record, opposite_records)
        matched = match_record is not None and iou >= threshold
        status = "MATCH" if matched else ("MISSED" if is_base_side else "NEW")
        box = record["globalBox"]
        rows.append({
            "selected": False, "status": status, "objectId": record["objectId"],
            "date": record["captureDate"], "image": record["sourceTifName"],
            "iou": iou if match_record is not None else None,
            "box": f"{box['left']},{box['top']},{box['right']},{box['bottom']}",
            "record": record,
        })
    return rows


def _copy_dataset_asset(data_root: str, path: str, output_root: str) -> str:
    if not path:
        return ""
    source_path = path if os.path.isabs(path) else os.path.join(data_root, path)
    target_path = os.path.join(output_root, os.path.basename(source_path))
    if os.path.isfile(source_path):
        shutil.copy2(source_path, target_path)
        return target_path
    return source_path


def export_selected(source_document: dict, selected_records: list[dict], output_json_path: str) -> str:
    """C# ExportSelectedCompareRecords 포팅: 체크한 개체의 타일 이미지+라벨을 실제로 복사해서
    새 object_db.json(및 images/labels 폴더)로 export."""
    output_root = os.path.dirname(output_json_path) or os.getcwd()
    os.makedirs(output_root, exist_ok=True)
    images_root = os.path.join(output_root, "images")
    labels_root = os.path.join(output_root, "labels")
    os.makedirs(images_root, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)

    source_tiles = {tile["tileId"]: tile for tile in source_document["tiles"]}
    exported_tiles, exported_objects = [], []

    for record in selected_records:
        tile = source_tiles.get(record["tileId"])
        if tile is None:
            continue

        new_tile_id = len(exported_tiles) + 1
        new_object_id = len(exported_objects) + 1
        image_path = _copy_dataset_asset(source_document.get("dataRoot", ""), tile["tileImagePath"], images_root)
        label_path = _copy_dataset_asset(source_document.get("dataRoot", ""), record.get("labelPath", ""), labels_root)
        if not label_path:
            label_path = os.path.join(labels_root, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
            with open(label_path, "w", encoding="utf-8") as fh:
                fh.write("")

        exported_tiles.append({**tile, "tileId": new_tile_id,
                                "tileImagePath": os.path.relpath(image_path, output_root),
                                "labelPath": os.path.relpath(label_path, output_root)})
        exported_objects.append({**record, "objectId": new_object_id, "tileId": new_tile_id,
                                  "labelPath": os.path.relpath(label_path, output_root)})

    document = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "dataRoot": output_root, "tileCount": len(exported_tiles), "objectCount": len(exported_objects),
        "excludedLabelFileCount": 0, "tiles": exported_tiles, "objects": exported_objects, "warnings": [],
    }
    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, ensure_ascii=False, indent=2)
    return output_json_path
