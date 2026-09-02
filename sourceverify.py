"""2번 원본 검증: object_db.json을 원본 TIF에서 크롭해 육안 검증 + 박스 추가/삭제/저장.

C# Form1.cs 검증 탭 로직(PreviewSourceObject/FindOverlappingRecords/AddBoxButton/
DeleteObjectButton/SaveCorrectedJsonButton) + ObjectDbJsonStore.cs의
SourceObjectPreviewRenderer 포팅. 마우스 드래그 캔버스 자체는 GUI 레이어(main.py)에 둠.
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image, ImageDraw

import sourceimage

CONTEXT_PIXELS = 180

_COLOR_PRIMARY = (0, 255, 0)
_COLOR_OVERLAP = (255, 165, 0)
_COLOR_OTHER = (0, 191, 255)


def get_corrected_json_path(source_json_path: str) -> str:
    directory = os.path.dirname(source_json_path) or os.getcwd()
    stem = os.path.splitext(os.path.basename(source_json_path))[0]
    suffix = stem if stem.lower().endswith("_new") else stem + "_new"
    return os.path.join(directory, suffix + ".json")


def _is_same_source(a: dict, b: dict) -> bool:
    return (a["captureDate"].lower() == b["captureDate"].lower() and
            a["sourceBaseName"].lower() == b["sourceBaseName"].lower())


def _boxes_intersect(a: dict, b: dict) -> bool:
    left, top = max(a["left"], b["left"]), max(a["top"], b["top"])
    right, bottom = min(a["right"], b["right"]), min(a["bottom"], b["bottom"])
    return right > left and bottom > top


def find_overlapping_records(records: list[dict], record: dict) -> list[dict]:
    return sorted(
        (c for c in records if c["objectId"] != record["objectId"] and _is_same_source(c, record)
         and _boxes_intersect(c["globalBox"], record["globalBox"])),
        key=lambda c: c["objectId"])


def find_same_source_records(records: list[dict], record: dict, excluded_ids: set) -> list[dict]:
    return sorted(
        (c for c in records if c["objectId"] not in excluded_ids and _is_same_source(c, record)),
        key=lambda c: c["objectId"])


def build_overlap_count_map(records: list[dict]) -> dict:
    counts = {r["objectId"]: 0 for r in records}
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record["captureDate"].lower(), record["sourceBaseName"].lower())
        groups.setdefault(key, []).append(record)
    for group in groups.values():
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                if _boxes_intersect(first["globalBox"], second["globalBox"]):
                    counts[first["objectId"]] += 1
                    counts[second["objectId"]] += 1
    return counts


def _get_crop_bounds(image_width: int, image_height: int, box: dict) -> tuple:
    left = max(0, box["left"] - CONTEXT_PIXELS)
    top = max(0, box["top"] - CONTEXT_PIXELS)
    right = min(image_width, box["right"] + CONTEXT_PIXELS)
    bottom = min(image_height, box["bottom"] + CONTEXT_PIXELS)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _draw_object_label(draw: ImageDraw.ImageDraw, crop_bounds: tuple, record: dict,
                        color: tuple, show_boundaries: bool, show_text: bool) -> None:
    box = record["globalBox"]
    crop_left, crop_top, crop_right, crop_bottom = crop_bounds
    crop_width, crop_height = crop_right - crop_left, crop_bottom - crop_top
    local = (box["left"] - crop_left, box["top"] - crop_top, box["right"] - crop_left, box["bottom"] - crop_top)
    if local[2] <= 0 or local[0] >= crop_width or local[3] <= 0 or local[1] >= crop_height:
        return

    if show_boundaries:
        width = max(2, crop_width // 300)
        draw.rectangle(local, outline=color, width=width)

    if show_text:
        label = f"{record['objectId']}: {record.get('className') or 'object'}"
        text_bbox = draw.textbbox((0, 0), label)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        label_top = max(0, local[1] - text_height - 4)
        draw.rectangle((local[0], label_top, local[0] + text_width + 8, label_top + text_height + 4), fill=(0, 0, 0))
        draw.text((local[0] + 4, label_top + 2), label, fill=color)


def render_preview(source_path: str, primary: dict, overlaps: list[dict], other_labels: list[dict],
                    show_boundaries: bool, show_text: bool) -> tuple:
    """C# SourceObjectPreviewRenderer.Render 포팅. 반환: (PIL Image, crop_bounds)."""
    width, height = sourceimage.get_size(source_path)
    crop_bounds = _get_crop_bounds(width, height, primary["globalBox"])
    left, top, right, bottom = crop_bounds
    array = sourceimage.read_full(source_path)[top:bottom, left:right]
    image = Image.fromarray(array).convert("RGB")
    draw = ImageDraw.Draw(image)

    for record in other_labels:
        _draw_object_label(draw, crop_bounds, record, _COLOR_OTHER, show_boundaries, show_text)
    for record in overlaps:
        _draw_object_label(draw, crop_bounds, record, _COLOR_OVERLAP, show_boundaries, show_text)
    _draw_object_label(draw, crop_bounds, primary, _COLOR_PRIMARY, show_boundaries, show_text)

    return image, crop_bounds


def box_to_yolo_label(local_box: dict, image_width: int, image_height: int) -> dict:
    return {
        "classId": 0,
        "centerX": (local_box["left"] + local_box["width"] / 2.0) / image_width,
        "centerY": (local_box["top"] + local_box["height"] / 2.0) / image_height,
        "width": local_box["width"] / image_width,
        "height": local_box["height"] / image_height,
    }


def update_record_box(record: dict, tile: Optional[dict], global_box: dict) -> dict:
    """C# UpdateRecordBox 포팅: 전역 좌표가 바뀌면 타일 기준 로컬 좌표/YOLO 값도 같이 갱신."""
    updated = dict(record)
    updated["globalBox"] = global_box
    if tile is None:
        return updated
    tile_box = tile["tileBoxInSource"]
    local_box = {
        "left": global_box["left"] - tile_box["left"], "top": global_box["top"] - tile_box["top"],
        "right": global_box["right"] - tile_box["left"], "bottom": global_box["bottom"] - tile_box["top"],
        "width": global_box["width"], "height": global_box["height"],
    }
    updated["localBox"] = local_box
    updated["yolo"] = box_to_yolo_label(local_box, tile["tileImageWidth"], tile["tileImageHeight"])
    return updated


def create_added_record(context_record: dict, global_box: dict, next_object_id: int, tile: Optional[dict]) -> dict:
    """C# CreateAddedRecord 포팅: 선택된 개체를 템플릿 삼아 새 박스 개체 생성."""
    template = dict(context_record)
    template["objectId"] = next_object_id
    template["classId"] = 0
    template["className"] = "whale"
    if "yolo" in template:
        template["yolo"] = {**template["yolo"], "classId": 0}
    return update_record_box(template, tile, global_box)


def insert_after(records: list[dict], context_object_id: int, new_record: dict) -> None:
    index = next((i for i, r in enumerate(records) if r["objectId"] == context_object_id), -1)
    if index < 0:
        records.append(new_record)
    else:
        records.insert(index + 1, new_record)


def delete_record(records: list[dict], object_id: int) -> None:
    index = next((i for i, r in enumerate(records) if r["objectId"] == object_id), -1)
    if index >= 0:
        del records[index]


def next_object_id(records: list[dict]) -> int:
    return 1 if not records else max(r["objectId"] for r in records) + 1
