"""3번 학습 타일: object_db.json -> 512/640 학습용 크롭 이미지 + YOLO 라벨.

C# TrainingDatasetBuilder.cs 포팅. 개체 1개당 모서리 4구역 + 중앙 1구역(tl/tr/bl/br/cc)
크롭 생성 규칙은 docs/domain-rules.md §4 참조.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from typing import Optional

from PIL import Image

import sourceimage
from objectdb import CoordinateBox

DEFAULT_OUTPUT_SIZES = (512, 640)

# (zoneName, column, row) — 3x3 그리드 기준
ZONES = (
    ("tl", 0, 0),
    ("tr", 2, 0),
    ("bl", 0, 2),
    ("br", 2, 2),
    ("cc", 1, 1),
)


def _round_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _clamp_to_int(value: float, minimum: int, maximum: int) -> int:
    return min(maximum, max(minimum, _round_away_from_zero(value)))


def _format_six(value: float) -> str:
    """C# "0.######" 포맷 재현: 소수 6자리까지, 끝자리 0/소수점 제거."""
    text = f"{value:.6f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _create_crop_bounds(box: CoordinateBox, source_size: tuple[int, int], output_size: int,
                         zone: tuple[str, int, int], rng: random.Random) -> Optional[tuple[int, int, int, int]]:
    source_width, source_height = source_size
    if source_width < output_size or source_height < output_size:
        return None

    _name, column, row = zone
    box_center_x = (box.left + box.right) / 2.0
    box_center_y = (box.top + box.bottom) / 2.0
    zone_size = output_size / 3.0
    zone_left = column * zone_size
    zone_top = row * zone_size
    zone_right = (column + 1) * zone_size
    zone_bottom = (row + 1) * zone_size

    min_local_x = max(zone_left, box.width / 2.0)
    max_local_x = min(zone_right, output_size - box.width / 2.0)
    min_local_y = max(zone_top, box.height / 2.0)
    max_local_y = min(zone_bottom, output_size - box.height / 2.0)

    local_x = _random_between_or_center(rng, min_local_x, max_local_x, zone_left, zone_right, output_size)
    local_y = _random_between_or_center(rng, min_local_y, max_local_y, zone_top, zone_bottom, output_size)

    left = _clamp_to_int(box_center_x - local_x, 0, source_width - output_size)
    top = _clamp_to_int(box_center_y - local_y, 0, source_height - output_size)
    return left, top, left + output_size, top + output_size


def _random_between_or_center(rng: random.Random, minimum: float, maximum: float,
                               zone_start: float, zone_end: float, output_size: int) -> float:
    if minimum <= maximum:
        return minimum + rng.random() * (maximum - minimum)
    return min(max((zone_start + zone_end) / 2.0, 0), output_size)


def _intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right < left or bottom < top:
        return (0, 0, 0, 0)
    return (left, top, right, bottom)


def _create_labels(crop_bounds: tuple[int, int, int, int], source_objects: list[dict]) -> list[str]:
    crop_left, crop_top, crop_right, crop_bottom = crop_bounds
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    labels = []

    for record in source_objects:
        gbox = record["globalBox"]
        object_rect = (gbox["left"], gbox["top"],
                        max(gbox["left"] + 1, gbox["right"]), max(gbox["top"] + 1, gbox["bottom"]))
        visible = _intersect(crop_bounds, object_rect)
        visible_width = visible[2] - visible[0]
        visible_height = visible[3] - visible[1]
        if visible_width <= 1 or visible_height <= 1:
            continue

        local_left = visible[0] - crop_left
        local_top = visible[1] - crop_top
        center_x = (local_left + visible_width / 2.0) / crop_width
        center_y = (local_top + visible_height / 2.0) / crop_height
        width = visible_width / crop_width
        height = visible_height / crop_height
        labels.append(f"0 {_format_six(center_x)} {_format_six(center_y)} "
                       f"{_format_six(width)} {_format_six(height)}")

    return labels


def _reset_generated_directory(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


@dataclass
class TrainingDatasetBuildResult:
    outputRootPath: str
    outputSizes: list
    imageCount: int
    labelFileCount: int
    warnings: list

    def to_display_text(self) -> str:
        lines = [
            "[OK] training dataset generated", "",
            f"Output images root : {self.outputRootPath}",
            f"Output sizes       : {', '.join(str(s) for s in self.outputSizes)}",
            f"Image files        : {self.imageCount}",
            f"Label files        : {self.labelFileCount}",
            f"Warnings           : {len(self.warnings)}",
        ]
        for warning in self.warnings[:20]:
            lines.append("- " + warning)
        if len(self.warnings) > 20:
            lines.append(f"- ... {len(self.warnings) - 20} more")
        return "\n".join(lines)


def build(object_db_json_path: str, source_root_path: str, output_root_path: str,
          output_sizes: Optional[list[int]] = None, rng: Optional[random.Random] = None) -> TrainingDatasetBuildResult:
    with open(object_db_json_path, encoding="utf-8") as fh:
        document = json.load(fh)

    if not document["objects"]:
        raise ValueError("Object DB contains no objects.")

    selected_sizes = sorted(set(output_sizes)) if output_sizes else list(DEFAULT_OUTPUT_SIZES)
    if not selected_sizes:
        raise ValueError("No output image sizes were selected.")

    os.makedirs(output_root_path, exist_ok=True)
    warnings: list[str] = []
    image_count = 0
    label_count = 0
    rng = rng or random.Random()

    objects_by_source: dict[tuple[str, str], list[dict]] = {}
    for record in document["objects"]:
        key = (record["captureDate"], record["sourceBaseName"])
        objects_by_source.setdefault(key, []).append(record)

    source_info_cache: dict[tuple[str, str], tuple[str, tuple[int, int]]] = {}

    for size in selected_sizes:
        metadata = []
        size_root = os.path.join(output_root_path, str(size))
        images_root = os.path.join(size_root, "images")
        labels_root = os.path.join(size_root, "labels")
        _reset_generated_directory(images_root)
        _reset_generated_directory(labels_root)
        os.makedirs(images_root, exist_ok=True)
        os.makedirs(labels_root, exist_ok=True)
        for path in (os.path.join(size_root, "classes.txt"), os.path.join(images_root, "classes.txt"),
                     os.path.join(labels_root, "classes.txt")):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("whale\n")
        for path in (os.path.join(size_root, "predefined_classes.txt"),
                     os.path.join(images_root, "predefined_classes.txt")):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("whale\n")

        for primary in document["objects"]:
            source_key = (primary["captureDate"], primary["sourceBaseName"])
            source_objects = objects_by_source.get(source_key)
            if not source_objects:
                continue

            if source_key not in source_info_cache:
                try:
                    source_path = sourceimage.resolve(
                        source_root_path, primary["captureDate"], primary["sourceBaseName"], primary["sourceTifName"])
                    source_info_cache[source_key] = (source_path, sourceimage.get_size(source_path))
                except Exception as exc:  # noqa: BLE001 - 원본 스캔이라 개별 소스 실패는 경고로만 남김
                    warnings.append(f"Skipped source {primary['captureDate']}/{primary['sourceTifName']}: {exc}")
                    source_info_cache[source_key] = None  # type: ignore[assignment]

            cache_entry = source_info_cache[source_key]
            if cache_entry is None:
                continue
            source_path, source_size = cache_entry

            box = CoordinateBox(**primary["globalBox"])
            for zone in ZONES:
                zone_name = zone[0]
                crop_bounds = _create_crop_bounds(box, source_size, size, zone, rng)
                if crop_bounds is None or (crop_bounds[2] - crop_bounds[0]) != size or (crop_bounds[3] - crop_bounds[1]) != size:
                    warnings.append(f"Skipped object {primary['objectId']}: source image is smaller than {size}.")
                    continue

                labels = _create_labels(crop_bounds, source_objects)
                if not labels:
                    warnings.append(f"Skipped crop for object {primary['objectId']}: no visible labels in crop.")
                    continue

                left, top, right, bottom = crop_bounds
                crop_array = sourceimage.read_crop(source_path, left, top, right, bottom)
                crop_image = Image.fromarray(crop_array)

                base_name = f"{primary['captureDate']}_{primary['sourceBaseName']}_obj{primary['objectId']:06d}_{size}_{zone_name}"
                image_path = os.path.join(images_root, base_name + ".png")
                label_path = os.path.join(labels_root, base_name + ".txt")
                label_img_path = os.path.join(images_root, base_name + ".txt")
                crop_image.save(image_path, format="PNG")
                label_text = "\n".join(labels) + "\n"
                with open(label_path, "w", encoding="utf-8") as fh:
                    fh.write(label_text)
                with open(label_img_path, "w", encoding="utf-8") as fh:
                    fh.write(label_text)

                metadata.append({
                    "baseName": base_name,
                    "imagePath": os.path.relpath(image_path, size_root),
                    "labelPath": os.path.relpath(label_path, size_root),
                    "captureDate": primary["captureDate"],
                    "sourceBaseName": primary["sourceBaseName"],
                    "sourceTifName": primary["sourceTifName"],
                    "cropBoxInSource": {"left": left, "top": top, "right": right, "bottom": bottom,
                                        "width": size, "height": size},
                    "imageWidth": size,
                    "imageHeight": size,
                    "primaryObjectId": primary["objectId"],
                    "zoneName": zone_name,
                })
                image_count += 1
                label_count += 1

        with open(os.path.join(size_root, "crop_metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)

    return TrainingDatasetBuildResult(output_root_path, selected_sizes, image_count, label_count, warnings)
