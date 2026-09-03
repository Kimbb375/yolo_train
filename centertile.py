"""2.2. 중앙 크롭 학습 데이터 — object_db.json + 원본 TIF -> 개체 1개당 중앙(cc) 크롭 1장만
뽑은 학습용 이미지+YOLO 라벨.

3번(trainingdataset.py)이 개체 1개당 5구역(tl/tr/bl/br/cc) 크롭을 만드는 것과 달리 cc 1장만
만듦 — 뷰어 박스 조절이 느려서 외부 라벨링 툴로 박스를 고치거나 개체를 새로 추가하려는
용도. 같이 나오는 object_db.json은 9번(labelsync.synchronize)의 '8번 기준 JSON'으로 그대로
쓸 수 있음: labels/*.txt를 외부 툴에서 고친 뒤 그 폴더를 9번 '보정 TXT 폴더'로 넣으면 됨.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

from PIL import Image

import sourceimage
from objectdb import CoordinateBox, ObjectDbDocument, TileRecord, TrainingObjectRecord, YoloLabel, \
    _convert_yolo_to_pixel_box
from trainingdataset import DEFAULT_OUTPUT_SIZES, ZONES, _create_crop_bounds, _create_labels, \
    _reset_generated_directory

_CC_ZONE = next(zone for zone in ZONES if zone[0] == "cc")


@dataclass
class CenterTileBuildResult:
    outputRootPath: str
    outputSizes: list
    imageCount: int
    labelFileCount: int
    warnings: list

    def to_display_text(self) -> str:
        lines = [
            "[OK] center-crop training data generated", "",
            f"Output root  : {self.outputRootPath}",
            f"Output sizes : {', '.join(str(s) for s in self.outputSizes)}",
            f"Image files  : {self.imageCount}",
            f"Label files  : {self.labelFileCount}",
            f"Warnings     : {len(self.warnings)}",
            "",
            "각 크기 폴더 밑 object_db.json을 9번 'TXT 보정 반영' 탭의 '8번 기준 JSON'으로,",
            "labels 폴더(수정본)를 '보정 TXT 폴더'로 넣으면 바로 반영됩니다.",
        ]
        for warning in self.warnings[:20]:
            lines.append("- " + warning)
        if len(self.warnings) > 20:
            lines.append(f"- ... {len(self.warnings) - 20} more")
        return "\n".join(lines)


def build(object_db_json_path: str, source_root_path: str, output_root_path: str,
          output_sizes: Optional[list[int]] = None, rng: Optional[random.Random] = None) -> CenterTileBuildResult:
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

    source_info_cache: dict[tuple[str, str], object] = {}

    for size in selected_sizes:
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

        tiles: list[TileRecord] = []
        objects: list[TrainingObjectRecord] = []

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
                    source_info_cache[source_key] = None

            cache_entry = source_info_cache[source_key]
            if cache_entry is None:
                continue
            source_path, source_size = cache_entry

            box = CoordinateBox(**primary["globalBox"])
            crop_bounds = _create_crop_bounds(box, source_size, size, _CC_ZONE, rng)
            if (crop_bounds is None or (crop_bounds[2] - crop_bounds[0]) != size
                    or (crop_bounds[3] - crop_bounds[1]) != size):
                warnings.append(f"Skipped object {primary['objectId']}: source image is smaller than {size}.")
                continue

            label_lines = _create_labels(crop_bounds, source_objects)
            if not label_lines:
                warnings.append(f"Skipped crop for object {primary['objectId']}: no visible labels in crop.")
                continue

            left, top, right, bottom = crop_bounds
            crop_array = sourceimage.read_crop(source_path, left, top, right, bottom)
            crop_image = Image.fromarray(crop_array)

            base_name = f"{primary['captureDate']}_{primary['sourceBaseName']}_obj{primary['objectId']:06d}_{size}_cc"
            image_path = os.path.join(images_root, base_name + ".png")
            label_path = os.path.join(labels_root, base_name + ".txt")
            label_img_path = os.path.join(images_root, base_name + ".txt")
            crop_image.save(image_path, format="PNG")
            label_text = "\n".join(label_lines) + "\n"
            with open(label_path, "w", encoding="utf-8") as fh:
                fh.write(label_text)
            with open(label_img_path, "w", encoding="utf-8") as fh:
                fh.write(label_text)
            image_count += 1
            label_count += 1

            tile_id = len(tiles) + 1
            tiles.append(TileRecord(
                tile_id, primary["captureDate"], primary["sourceBaseName"], primary["sourceTifName"],
                os.path.relpath(image_path, size_root), os.path.relpath(label_path, size_root),
                CoordinateBox(left, top, right, bottom, size, size), size, size))

            for line_number, line in enumerate(label_lines, start=1):
                label = YoloLabel.try_parse(line)
                if label is None:
                    warnings.append(f"Invalid YOLO label generated at {label_path}:{line_number}")
                    continue
                local_box = _convert_yolo_to_pixel_box(label, size, size)
                global_box = local_box.offset(left, top)
                objects.append(TrainingObjectRecord(
                    len(objects) + 1, primary["captureDate"], primary["sourceBaseName"], primary["sourceTifName"],
                    tile_id, 0, "whale", os.path.relpath(label_path, size_root), line_number,
                    local_box, global_box, label))

        db_document = ObjectDbDocument(
            generatedAt=datetime.now().astimezone().isoformat(timespec="microseconds"),
            dataRoot=size_root,
            tileCount=len(tiles),
            objectCount=len(objects),
            excludedLabelFileCount=0,
            tiles=tiles,
            objects=objects,
            warnings=[],
        )
        with open(os.path.join(size_root, "object_db.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(db_document), fh, ensure_ascii=False, indent=2)

    return CenterTileBuildResult(output_root_path, selected_sizes, image_count, label_count, warnings)
