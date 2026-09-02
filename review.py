"""7번 후보 검수: 6번 candidates.json을 빠르게 넘기며 고래/고래 아님 분류.

C# InferenceReviewTools.cs 포팅 (7번에 필요한 부분만 — 8번 매칭/선별에 쓰이는
ExportConfirmedObjectDb/MergeConfirmedObjectDb 등은 8번 탭 만들 때 포팅함).
"""

from __future__ import annotations

import json
import operator
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from objectdb import CoordinateBox
from trainingdataset import _format_six

_FILTER_PATTERN = re.compile(r"^(width|height|area|conf)\s*(>=|<=|==|>|<)\s*([\d.]+)$")
_FILTER_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq,
}


@dataclass
class ReviewCandidate:
    candidateId: int
    sourceBaseName: str
    sourceTifName: str
    sourceTifPath: str
    tileName: str
    confidence: float
    globalBox: CoordinateBox
    candidateCropBox: CoordinateBox
    candidateImagePath: str
    candidateLabelPath: str
    candidateInfoPath: str


def _resolve_asset_path(candidate_json_directory: str, asset_path: str) -> str:
    """C# ResolveCandidateAssetPath 포팅: 경로가 안 맞으면 candidates/ 폴더, JSON 옆 순으로 재탐색."""
    if not asset_path:
        return asset_path
    if os.path.isfile(asset_path):
        return asset_path

    file_name = os.path.basename(asset_path)
    if not file_name:
        return asset_path

    beside_json = os.path.join(candidate_json_directory, "candidates", file_name)
    if os.path.isfile(beside_json):
        return beside_json

    direct = os.path.join(candidate_json_directory, file_name)
    return direct if os.path.isfile(direct) else asset_path


def load_candidates(candidate_json_path: str) -> list[ReviewCandidate]:
    with open(candidate_json_path, encoding="utf-8") as fh:
        document = json.load(fh)

    directory = os.path.dirname(os.path.abspath(candidate_json_path))
    candidates = []
    for item in document.get("candidates", []):
        global_box = CoordinateBox(**item["globalBox"])
        crop_box_data = item.get("candidateCropBox")
        crop_box = CoordinateBox(**crop_box_data) if crop_box_data else CoordinateBox(
            item.get("tileLeft", 0), item.get("tileTop", 0),
            item.get("tileLeft", 0) + 640, item.get("tileTop", 0) + 640, 640, 640)
        candidates.append(ReviewCandidate(
            item["candidateId"], item["sourceBaseName"], item["sourceTifName"], item["sourceTifPath"],
            item["tileName"], item["confidence"], global_box, crop_box,
            _resolve_asset_path(directory, item.get("candidateImagePath", "")),
            _resolve_asset_path(directory, item.get("candidateLabelPath", "")),
            _resolve_asset_path(directory, item.get("candidateInfoPath", ""))))

    candidates.sort(key=lambda c: (c.sourceTifName.lower(), c.globalBox.top, c.globalBox.left))
    return candidates


def _filter_value(candidate: ReviewCandidate, field: str) -> float:
    if field == "width":
        return candidate.globalBox.width
    if field == "height":
        return candidate.globalBox.height
    if field == "area":
        return candidate.globalBox.width * candidate.globalBox.height
    return candidate.confidence


def parse_filters(text: str) -> list[Callable[[ReviewCandidate], bool]]:
    """"conf>=0.3, width>=20" 같은 필터 식 파싱. width/height/area/conf 지원."""
    conditions = []
    for item in (text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        match = _FILTER_PATTERN.match(item)
        if not match:
            raise ValueError(f"Invalid filter: {item}")
        field, op, value = match.group(1), match.group(2), float(match.group(3))
        op_func = _FILTER_OPS[op]
        conditions.append(lambda c, field=field, op_func=op_func, value=value: op_func(_filter_value(c, field), value))
    return conditions


def apply_filters(candidates: list[ReviewCandidate], filter_text: str) -> list[ReviewCandidate]:
    conditions = parse_filters(filter_text)
    if not conditions:
        return list(candidates)
    return [c for c in candidates if all(condition(c) for condition in conditions)]


def _stem(status: str, candidate: ReviewCandidate) -> str:
    box = candidate.globalBox
    return (f"{status}_{candidate.sourceBaseName}_{box.left}_{box.top}_{box.right}_{box.bottom}"
            f"_cand{candidate.candidateId:06d}")


def get_confirmed_save_result(candidate: ReviewCandidate, output_root: str) -> dict:
    stem = _stem("confirmed", candidate)
    return {
        "imagePath": os.path.join(output_root, "confirmed", "images", stem + ".png"),
        "labelPath": os.path.join(output_root, "confirmed", "labels", stem + ".txt"),
        "infoPath": os.path.join(output_root, "confirmed", stem + ".json"),
    }


def get_negative_save_result(candidate: ReviewCandidate, output_root: str) -> dict:
    stem = _stem("negative", candidate)
    return {
        "imagePath": os.path.join(output_root, "negative", "images", stem + ".png"),
        "labelPath": os.path.join(output_root, "negative", "labels", stem + ".txt"),
        "infoPath": os.path.join(output_root, "negative", stem + ".json"),
    }


def _build_yolo_label(box: CoordinateBox, crop: CoordinateBox) -> str:
    visible_left, visible_top = max(box.left, crop.left), max(box.top, crop.top)
    visible_right, visible_bottom = min(box.right, crop.right), min(box.bottom, crop.bottom)
    width = max(1, visible_right - visible_left)
    height = max(1, visible_bottom - visible_top)
    center_x = (visible_left - crop.left + width / 2.0) / crop.width
    center_y = (visible_top - crop.top + height / 2.0) / crop.height
    return (f"0 {_format_six(center_x)} {_format_six(center_y)} "
            f"{_format_six(width / crop.width)} {_format_six(height / crop.height)}")


def _copy_if_exists(source_path: str, target_path: str) -> None:
    if source_path and os.path.isfile(source_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)


def _delete_if_exists(path: str) -> None:
    if path and os.path.isfile(path):
        os.remove(path)


def _append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_class_files(records_root: str) -> None:
    with open(os.path.join(records_root, "classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("whale\n")
    with open(os.path.join(records_root, "predefined_classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("whale\n")


def _build_record(status: str, candidate: ReviewCandidate, result: dict) -> dict:
    box = candidate.globalBox
    crop = candidate.candidateCropBox
    return {
        "savedAt": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "status": status,
        "candidateId": candidate.candidateId,
        "sourceBaseName": candidate.sourceBaseName,
        "sourceTifName": candidate.sourceTifName,
        "sourceTifPath": candidate.sourceTifPath,
        "confidence": candidate.confidence,
        "globalBox": {"left": box.left, "top": box.top, "right": box.right, "bottom": box.bottom,
                      "width": box.width, "height": box.height},
        "candidateCropBox": {"left": crop.left, "top": crop.top, "right": crop.right, "bottom": crop.bottom,
                              "width": crop.width, "height": crop.height},
        "imagePath": result["imagePath"],
        "labelPath": result["labelPath"],
    }


def save_confirmed(candidate: ReviewCandidate, output_root: str) -> dict:
    records_root = os.path.join(output_root, "confirmed")
    os.makedirs(os.path.join(records_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(records_root, "labels"), exist_ok=True)
    _write_class_files(records_root)

    result = get_confirmed_save_result(candidate, output_root)
    _copy_if_exists(candidate.candidateImagePath, result["imagePath"])
    if os.path.isfile(candidate.candidateLabelPath):
        shutil.copy2(candidate.candidateLabelPath, result["labelPath"])
    else:
        with open(result["labelPath"], "w", encoding="utf-8") as fh:
            fh.write(_build_yolo_label(candidate.globalBox, candidate.candidateCropBox) + "\n")

    record = _build_record("confirmed", candidate, result)
    with open(result["infoPath"], "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    _append_jsonl(os.path.join(records_root, "records.jsonl"), record)
    return result


def save_negative(candidate: ReviewCandidate, output_root: str) -> dict:
    records_root = os.path.join(output_root, "negative")
    os.makedirs(os.path.join(records_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(records_root, "labels"), exist_ok=True)
    _write_class_files(records_root)

    result = get_negative_save_result(candidate, output_root)
    _copy_if_exists(candidate.candidateImagePath, result["imagePath"])
    with open(result["labelPath"], "w", encoding="utf-8") as fh:
        fh.write("")

    record = _build_record("negative", candidate, result)
    with open(result["infoPath"], "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    _append_jsonl(os.path.join(records_root, "records.jsonl"), record)
    return result


def is_confirmed(candidate: ReviewCandidate, output_root: str) -> bool:
    result = get_confirmed_save_result(candidate, output_root)
    return os.path.isfile(result["imagePath"]) and os.path.isfile(result["labelPath"])


def is_negative(candidate: ReviewCandidate, output_root: str) -> bool:
    result = get_negative_save_result(candidate, output_root)
    return os.path.isfile(result["imagePath"]) and os.path.isfile(result["labelPath"])


def delete_confirmed(candidate: ReviewCandidate, output_root: str) -> dict:
    result = get_confirmed_save_result(candidate, output_root)
    _delete_if_exists(result["imagePath"])
    _delete_if_exists(result["labelPath"])
    _delete_if_exists(result["infoPath"])
    records_root = os.path.join(output_root, "confirmed")
    os.makedirs(records_root, exist_ok=True)
    _append_jsonl(os.path.join(records_root, "records.jsonl"), _build_record("unconfirmed", candidate, result))
    return result


def _load_review_record_file(path: str) -> list[dict]:
    if path.lower().endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        value = json.loads(text)
        return [value] if isinstance(value, dict) else list(value)
    except json.JSONDecodeError:
        return [json.loads(chunk) for chunk in _split_top_level_json_objects(text)]


def _split_top_level_json_objects(text: str):
    depth, start, in_string, escaped = 0, -1, False, False
    for index, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start:index + 1]
                start = -1


def _record_key(record: dict) -> tuple:
    box = record["globalBox"]
    return (record["candidateId"], record["sourceTifName"].lower(), box["left"], box["top"], box["right"], box["bottom"])


def load_confirmed_records(review_output_root: str) -> list[dict]:
    """C# LoadConfirmedReviewRecords 포팅: 같은 후보에 대한 마지막 기록만 채택, status==confirmed만."""
    if os.path.isfile(review_output_root):
        records = _load_review_record_file(review_output_root)
    else:
        confirmed_root = review_output_root if os.path.basename(
            review_output_root.rstrip("/\\")).lower() == "confirmed" else os.path.join(review_output_root, "confirmed")
        if not os.path.isdir(confirmed_root):
            raise FileNotFoundError(f"Confirmed review folder was not found: {confirmed_root}")

        individual = []
        for name in sorted(os.listdir(confirmed_root)):
            if name.lower().startswith("confirmed_") and name.lower().endswith(".json"):
                individual.extend(_load_review_record_file(os.path.join(confirmed_root, name)))
        jsonl_path = os.path.join(confirmed_root, "records.jsonl")
        records = individual if individual else (_load_review_record_file(jsonl_path) if os.path.isfile(jsonl_path) else [])

    latest: dict[tuple, dict] = {}
    for record in records:
        latest[_record_key(record)] = record  # 나중 것이 이전 것을 덮어씀(입력 순서가 시간 순서라고 가정)
    confirmed = [r for r in latest.values() if r.get("status") == "confirmed"]
    confirmed.sort(key=lambda r: (r["sourceTifName"].lower(), r["globalBox"]["top"], r["globalBox"]["left"]))
    return confirmed


def _leading_digits(value: str) -> str:
    result = []
    for ch in value or "":
        if ch.isdigit():
            result.append(ch)
        else:
            break
    return "".join(result)


def _is_date_like_token(value: str) -> bool:
    return len(_leading_digits(value)) in (4, 6, 8)


def _normalize_date_token(value: str) -> str:
    date_part = _leading_digits(value)
    suffix = value[len(date_part):]
    return ("20" + date_part + suffix) if len(date_part) == 6 else (date_part + suffix)


def _normalize_review_source(record: dict) -> tuple[str, str, str]:
    source_tif_name = record.get("sourceTifName") or os.path.basename(record.get("sourceTifPath", ""))
    stem = os.path.splitext(source_tif_name)[0]
    parts = [p for p in stem.split("_") if p != ""]
    if len(parts) >= 2 and _is_date_like_token(parts[0]):
        source_base_name = record.get("sourceBaseName") or "_".join(parts[1:])
        return _normalize_date_token(parts[0]), source_base_name, source_tif_name

    folder_name = os.path.basename(os.path.dirname(record.get("sourceTifPath", "")))
    capture_date = _normalize_date_token(folder_name) if _is_date_like_token(folder_name) else ""
    return capture_date, record.get("sourceBaseName") or stem, source_tif_name


def _make_relative_if_possible(root: str, path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path) and os.path.abspath(path).lower().startswith(os.path.abspath(root).lower()):
        return os.path.relpath(path, root)
    return path


def build_object_db_from_review_records(review_output_root: str, records: list[dict]) -> dict:
    """C# BuildObjectDbFromReviewRecords 포팅. confirmed 기록들을 object_db.json 구조로 재구성."""
    tiles, objects = [], []
    for record in records:
        tile_id = len(tiles) + 1
        capture_date, source_base_name, source_tif_name = _normalize_review_source(record)
        crop = record["candidateCropBox"]
        box = record["globalBox"]
        local_box = {"left": box["left"] - crop["left"], "top": box["top"] - crop["top"],
                     "right": box["right"] - crop["left"], "bottom": box["bottom"] - crop["top"],
                     "width": box["width"], "height": box["height"]}
        image_path = _make_relative_if_possible(review_output_root, record.get("imagePath", ""))
        label_path = _make_relative_if_possible(review_output_root, record.get("labelPath", ""))

        tiles.append({"tileId": tile_id, "captureDate": capture_date, "sourceBaseName": source_base_name,
                      "sourceTifName": source_tif_name, "tileImagePath": image_path, "labelPath": label_path,
                      "tileBoxInSource": crop, "tileImageWidth": crop["width"], "tileImageHeight": crop["height"]})
        objects.append({"objectId": len(objects) + 1, "captureDate": capture_date, "sourceBaseName": source_base_name,
                        "sourceTifName": source_tif_name, "tileId": tile_id, "classId": 0, "className": "whale",
                        "labelPath": label_path, "labelLineNumber": 1, "localBox": local_box, "globalBox": box,
                        "yolo": {"classId": 0, "centerX": 0.5, "centerY": 0.5, "width": 1.0, "height": 1.0}})

    return {"generatedAt": datetime.now().astimezone().isoformat(timespec="microseconds"),
            "dataRoot": review_output_root, "tileCount": len(tiles), "objectCount": len(objects),
            "excludedLabelFileCount": 0, "tiles": tiles, "objects": objects, "warnings": []}


def load_confirmed_object_db_document(review_output_root: str) -> dict:
    return build_object_db_from_review_records(review_output_root, load_confirmed_records(review_output_root))


def delete_negative(candidate: ReviewCandidate, output_root: str) -> dict:
    result = get_negative_save_result(candidate, output_root)
    _delete_if_exists(result["imagePath"])
    _delete_if_exists(result["labelPath"])
    _delete_if_exists(result["infoPath"])
    records_root = os.path.join(output_root, "negative")
    os.makedirs(records_root, exist_ok=True)
    _append_jsonl(os.path.join(records_root, "records.jsonl"), _build_record("unnegative", candidate, result))
    return result
