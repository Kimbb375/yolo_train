"""9. TXT 보정 반영 — 외부 라벨링 프로그램에서 고친 YOLO TXT를 기준으로 8번 Object DB의
객체 좌표/삭제/추가를 동기화.

C# AdjustedLabelSynchronizer.cs 포팅. TXT가 최종값 - 없는 TXT는 기존 객체 유지(경고만),
빈 TXT는 해당 타일 객체 삭제로 처리, 여러 줄은 객체 추가로 반영.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

from objectdb import CoordinateBox, YoloLabel, _round_away_from_zero


def default_output_path(base_json_path: str) -> str:
    directory = os.path.dirname(base_json_path) or "."
    stem = os.path.splitext(os.path.basename(base_json_path))[0]
    return os.path.join(directory, stem + "_txt_synced.json")


def _find_edited_label_path(labels_root: str, configured_path: str) -> Optional[str]:
    file_name = os.path.basename(configured_path)
    candidates = [
        configured_path if os.path.isabs(configured_path) else os.path.join(labels_root, configured_path),
        os.path.join(labels_root, file_name),
        os.path.join(labels_root, "labels", file_name),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_labels(label_path: str, warnings: list[str]) -> list[tuple[YoloLabel, int]]:
    with open(label_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    result = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        label = YoloLabel.try_parse(line)
        if label is None:
            warnings.append(f"Invalid YOLO label at {label_path}:{line_number}")
            continue
        result.append((label, line_number))
    return result


def _try_convert_to_boxes(label: YoloLabel, tile: dict) -> Optional[tuple[CoordinateBox, CoordinateBox]]:
    tile_width, tile_height = tile["tileImageWidth"], tile["tileImageHeight"]
    if (tile_width <= 0 or tile_height <= 0 or
            not (0 <= label.centerX <= 1) or not (0 <= label.centerY <= 1) or
            not (0 < label.width <= 1) or not (0 < label.height <= 1)):
        return None

    def clamp(value: int, low: int, high: int) -> int:
        return min(high, max(low, value))

    left = clamp(_round_away_from_zero((label.centerX - label.width / 2) * tile_width), 0, tile_width - 1)
    top = clamp(_round_away_from_zero((label.centerY - label.height / 2) * tile_height), 0, tile_height - 1)
    right = clamp(_round_away_from_zero((label.centerX + label.width / 2) * tile_width), left + 1, tile_width)
    bottom = clamp(_round_away_from_zero((label.centerY + label.height / 2) * tile_height), top + 1, tile_height)
    local_box = CoordinateBox(left, top, right, bottom, right - left, bottom - top)
    source_box = tile["tileBoxInSource"]
    global_box = local_box.offset(source_box["left"], source_box["top"])
    return local_box, global_box


@dataclass
class LabelSyncResult:
    outputJsonPath: str
    processedLabelFileCount: int
    originalObjectCount: int
    synchronizedObjectCount: int
    addedObjectCount: int
    removedObjectCount: int
    warnings: list

    def to_display_text(self) -> str:
        lines = [
            "[OK] TXT edits applied to Object DB", "",
            f"Output JSON       : {self.outputJsonPath}",
            f"Processed TXT     : {self.processedLabelFileCount}",
            f"Objects           : {self.originalObjectCount} -> {self.synchronizedObjectCount}",
            f"Added / removed   : {self.addedObjectCount} / {self.removedObjectCount}",
            f"Warnings          : {len(self.warnings)}",
        ]
        for warning in self.warnings[:20]:
            lines.append("- " + warning)
        if len(self.warnings) > 20:
            lines.append(f"- ... {len(self.warnings) - 20} more")
        return "\n".join(lines)


def synchronize(base_json_path: str, labels_root_path: str,
                 output_json_path: Optional[str] = None) -> LabelSyncResult:
    with open(base_json_path, encoding="utf-8") as fh:
        source = json.load(fh)
    if not os.path.isdir(labels_root_path):
        raise FileNotFoundError(f"Edited TXT folder was not found: {labels_root_path}")

    tiles = source.get("tiles", [])
    objects = source.get("objects", [])

    objects_by_tile: dict[int, list[dict]] = {}
    for obj in sorted(objects, key=lambda o: (o["labelLineNumber"], o["objectId"])):
        objects_by_tile.setdefault(obj["tileId"], []).append(obj)
    next_object_id = max((o["objectId"] for o in objects), default=0) + 1

    synchronized_objects: list[dict] = []
    processed_tile_ids: set[int] = set()
    processed_label_files = 0
    old_in_processed_files = 0
    warnings: list[str] = []

    for tile in tiles:
        old_objects = objects_by_tile.get(tile["tileId"], [])
        label_path = _find_edited_label_path(labels_root_path, tile["labelPath"])
        if label_path is None:
            synchronized_objects.extend(old_objects)
            warnings.append(f"TXT not found; kept existing objects for tile {tile['tileId']}: {tile['labelPath']}")
            continue

        processed_label_files += 1
        processed_tile_ids.add(tile["tileId"])
        old_in_processed_files += len(old_objects)
        for label_index, (label, line_number) in enumerate(_read_labels(label_path, warnings)):
            boxes = _try_convert_to_boxes(label, tile)
            if boxes is None:
                warnings.append(f"Skipped out-of-range YOLO label at {label_path}:{line_number}")
                continue
            local_box, global_box = boxes
            template = old_objects[label_index] if label_index < len(old_objects) else None
            if template is not None:
                object_id = template["objectId"]
            else:
                object_id = next_object_id
                next_object_id += 1
            synchronized_objects.append({
                "objectId": object_id,
                "captureDate": tile["captureDate"],
                "sourceBaseName": tile["sourceBaseName"],
                "sourceTifName": tile["sourceTifName"],
                "tileId": tile["tileId"],
                "classId": 0,
                "className": "whale",
                "labelPath": tile["labelPath"],
                "labelLineNumber": line_number,
                "localBox": asdict(local_box),
                "globalBox": asdict(global_box),
                "yolo": {"classId": 0, "centerX": label.centerX, "centerY": label.centerY,
                         "width": label.width, "height": label.height},
            })

    output = dict(source)
    output["tileCount"] = len(tiles)
    output["objectCount"] = len(synchronized_objects)
    output["objects"] = synchronized_objects
    output["warnings"] = list(source.get("warnings", [])) + warnings

    resolved_output_path = output_json_path or default_output_path(base_json_path)
    output_dir = os.path.dirname(resolved_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(resolved_output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    new_in_processed_files = sum(1 for item in synchronized_objects if item["tileId"] in processed_tile_ids)
    added = max(0, new_in_processed_files - old_in_processed_files)
    removed = max(0, old_in_processed_files - new_in_processed_files)

    return LabelSyncResult(
        resolved_output_path, processed_label_files, len(objects), len(synchronized_objects),
        added, removed, warnings)
