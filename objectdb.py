"""1번 라벨 DB: YOLO 라벨 txt + 타일 이미지 -> object_db.json (원본 TIF 좌표 기준).

C# ObjectDbBuilder.cs 포팅. 필드명은 camelCase 그대로 유지함
(docs/json-schema-contract.md 계약 — dataclasses.asdict()가 그대로 JSON 키가 되게).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from PIL import Image

CLASSES_FILE_NAMES = {"classes.txt", "predefined_classes.txt"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _round_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _leading_date_part(value: str) -> str:
    result = []
    for ch in value:
        if ch.isdigit():
            result.append(ch)
        else:
            break
    return "".join(result)


def _is_numeric_token(value: str) -> bool:
    return value != "" and value.isdigit()


def _is_capture_date_token(value: str) -> bool:
    return len(_leading_date_part(value)) in (6, 8)


def _normalize_capture_date_token(value: str) -> str:
    date_part = _leading_date_part(value)
    suffix = value[len(date_part):]
    return ("20" + date_part + suffix) if len(date_part) == 6 else (date_part + suffix)


def _try_parse_prefixed_coordinate(value: str, prefix: str) -> Optional[int]:
    if len(value) > 1 and value[0].lower() == prefix.lower():
        try:
            return int(value[1:])
        except ValueError:
            return None
    return None


@dataclass
class CoordinateBox:
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int

    @staticmethod
    def from_double(left: float, top: float, right: float, bottom: float) -> "CoordinateBox":
        l = _round_away_from_zero(left)
        t = _round_away_from_zero(top)
        r = _round_away_from_zero(right)
        b = _round_away_from_zero(bottom)
        return CoordinateBox(l, t, r, b, max(0, r - l), max(0, b - t))

    def offset(self, offset_x: int, offset_y: int) -> "CoordinateBox":
        return CoordinateBox(
            self.left + offset_x, self.top + offset_y,
            self.right + offset_x, self.bottom + offset_y,
            self.width, self.height)


@dataclass
class YoloLabel:
    classId: int
    centerX: float
    centerY: float
    width: float
    height: float

    @staticmethod
    def try_parse(line: str) -> Optional["YoloLabel"]:
        parts = line.split()
        if len(parts) != 5:
            return None
        try:
            return YoloLabel(int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            return None


@dataclass
class ParsedTileName:
    sourceBaseName: str
    left: int
    top: int
    right: int
    bottom: int
    captureDate: Optional[str] = None

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


def _is_valid(tile: ParsedTileName) -> bool:
    return tile.left >= 0 and tile.top >= 0 and tile.right >= tile.left and tile.bottom >= tile.top


def _create_centered_tile_name(source_base_name: str, center_x: int, center_y: int,
                                image_width: int, image_height: int, capture_date: str) -> ParsedTileName:
    left = max(0, center_x - image_width // 2)
    top = max(0, center_y - image_height // 2)
    return ParsedTileName(source_base_name, left, top, left + image_width - 1, top + image_height - 1, capture_date)


def try_parse_tile_name(stem: str, image_width: int, image_height: int) -> Optional[ParsedTileName]:
    """4가지 패턴 순서대로 시도 (domain-rules.md §1.1)."""
    parts = [p.strip() for p in stem.split("_")]
    parts = [p for p in parts if p != ""]
    if len(parts) < 4:
        return None

    if len(parts) == 4 and _is_capture_date_token(parts[0]) and _is_numeric_token(parts[1]):
        try:
            center_x, center_y = int(parts[2]), int(parts[3])
            tile = _create_centered_tile_name(parts[1], center_x, center_y, image_width, image_height,
                                               _normalize_capture_date_token(parts[0]))
            if _is_valid(tile):
                return tile
        except ValueError:
            pass

    if len(parts) == 4 and _is_capture_date_token(parts[0]) and _is_numeric_token(parts[1]):
        center_y = _try_parse_prefixed_coordinate(parts[2], "y")
        center_x = _try_parse_prefixed_coordinate(parts[3], "x")
        if center_y is not None and center_x is not None:
            tile = _create_centered_tile_name(parts[1], center_x, center_y, image_width, image_height,
                                               _normalize_capture_date_token(parts[0]))
            if _is_valid(tile):
                return tile

    if parts[-1].lower() == "raw" and len(parts) >= 4:
        try:
            raw_left, raw_top = int(parts[-3]), int(parts[-2])
            source_name = "_".join(parts[:-3])
            if source_name:
                tile = ParsedTileName(source_name, raw_left, raw_top,
                                       raw_left + image_width - 1, raw_top + image_height - 1)
                if _is_valid(tile):
                    return tile
        except ValueError:
            pass

    if len(parts) >= 5:
        try:
            left, top, right, bottom = int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1])
            source_name = "_".join(parts[:-4])
            if source_name:
                tile = ParsedTileName(source_name, left, top, right, bottom)
                if _is_valid(tile):
                    return tile
        except ValueError:
            pass

    return None


def _to_tif_name(parsed: ParsedTileName) -> str:
    source_base_name = parsed.sourceBaseName
    if parsed.captureDate:
        date_part = _leading_date_part(parsed.captureDate)
        suffix = parsed.captureDate[len(date_part):]
        if len(date_part) >= 8:
            return date_part[-6:] + suffix + "_" + source_base_name + ".tif"
    return source_base_name if os.path.splitext(source_base_name)[1] else source_base_name + ".tif"


def _find_capture_date(data_path: str, path: str) -> str:
    relative = os.path.relpath(path, data_path)
    return relative.split(os.sep)[0]


def _is_classes_file(path: str) -> bool:
    return os.path.basename(path).lower() in CLASSES_FILE_NAMES


def _get_mirrored_image_path(data_path: str, label_path: str, image_extension: str) -> Optional[str]:
    relative = os.path.relpath(label_path, data_path)
    parts = relative.split(os.sep)
    label_index = next((i for i, p in enumerate(parts) if p.lower() == "labels"), -1)
    if label_index < 0:
        return None
    parts[label_index] = "images"
    mirrored_relative = os.path.splitext(os.path.join(*parts))[0] + image_extension
    return os.path.join(data_path, mirrored_relative)


def _find_tile_image_path(data_path: str, label_path: str) -> str:
    direct = os.path.splitext(label_path)[0] + ".png"
    if os.path.exists(direct):
        return direct
    for extension in IMAGE_EXTENSIONS:
        mirrored = _get_mirrored_image_path(data_path, label_path, extension)
        if mirrored and os.path.exists(mirrored):
            return mirrored
    return direct


def _convert_yolo_to_pixel_box(label: YoloLabel, tile_width: int, tile_height: int) -> CoordinateBox:
    center_x = label.centerX * tile_width
    center_y = label.centerY * tile_height
    width = label.width * tile_width
    height = label.height * tile_height
    return CoordinateBox.from_double(center_x - width / 2, center_y - height / 2,
                                      center_x + width / 2, center_y + height / 2)


@dataclass
class TileRecord:
    tileId: int
    captureDate: str
    sourceBaseName: str
    sourceTifName: str
    tileImagePath: str
    labelPath: str
    tileBoxInSource: CoordinateBox
    tileImageWidth: int
    tileImageHeight: int


@dataclass
class TrainingObjectRecord:
    objectId: int
    captureDate: str
    sourceBaseName: str
    sourceTifName: str
    tileId: int
    classId: int
    className: str
    labelPath: str
    labelLineNumber: int
    localBox: CoordinateBox
    globalBox: CoordinateBox
    yolo: YoloLabel


@dataclass
class ObjectDbDocument:
    generatedAt: str
    dataRoot: str
    tileCount: int
    objectCount: int
    excludedLabelFileCount: int
    tiles: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class ObjectDbBuildResult:
    outputPath: str
    tileCount: int
    objectCount: int
    excludedLabelFileCount: int
    warnings: list

    def to_display_text(self) -> str:
        lines = [
            "[OK] object DB JSON generated", "",
            f"Output  : {self.outputPath}",
            f"Tiles   : {self.tileCount}",
            f"Objects : {self.objectCount}",
            f"Excluded label files: {self.excludedLabelFileCount}",
            f"Warnings: {len(self.warnings)}",
        ]
        for warning in self.warnings[:20]:
            lines.append("- " + warning)
        if len(self.warnings) > 20:
            lines.append(f"- ... {len(self.warnings) - 20} more")
        return "\n".join(lines)


def build(data_path: str, output_path: Optional[str] = None) -> ObjectDbBuildResult:
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    tiles: list[TileRecord] = []
    objects: list[TrainingObjectRecord] = []
    warnings: list[str] = []
    excluded = 0

    label_paths = []
    for root, _dirs, files in os.walk(data_path):
        for name in files:
            if name.lower().endswith(".txt"):
                label_paths.append(os.path.join(root, name))
    label_paths = [p for p in label_paths if not _is_classes_file(p)]
    label_paths.sort(key=lambda p: p.lower())

    for label_path in label_paths:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        relative_label = os.path.relpath(label_path, data_path)

        if "raw" in stem.lower():
            excluded += 1
            warnings.append(f"Excluded label file because file name contains raw: {relative_label}")
            continue

        image_path = _find_tile_image_path(data_path, label_path)
        if not os.path.exists(image_path):
            excluded += 1
            warnings.append(f"Excluded label file because tile image is missing: {relative_label}")
            continue

        with Image.open(image_path) as img:
            tile_width, tile_height = img.size

        parsed = try_parse_tile_name(stem, tile_width, tile_height)
        if parsed is None:
            excluded += 1
            warnings.append(f"Excluded label file because tile name is invalid: {relative_label}")
            continue

        capture_date = parsed.captureDate or _find_capture_date(data_path, label_path)
        tile_id = len(tiles) + 1
        tif_name = _to_tif_name(parsed)
        tile_box = CoordinateBox(parsed.left, parsed.top, parsed.right, parsed.bottom,
                                  parsed.width, parsed.height)

        tiles.append(TileRecord(
            tile_id, capture_date, parsed.sourceBaseName, tif_name,
            os.path.relpath(image_path, data_path), relative_label,
            tile_box, tile_width, tile_height))

        with open(label_path, encoding="utf-8") as fh:
            lines = fh.readlines()

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            label = YoloLabel.try_parse(line)
            if label is None:
                warnings.append(f"Invalid YOLO label at {relative_label}:{line_number}")
                continue

            normalized = YoloLabel(0, label.centerX, label.centerY, label.width, label.height)
            local_box = _convert_yolo_to_pixel_box(normalized, tile_width, tile_height)
            global_box = local_box.offset(parsed.left, parsed.top)
            objects.append(TrainingObjectRecord(
                len(objects) + 1, capture_date, parsed.sourceBaseName, tif_name,
                tile_id, 0, "whale", relative_label, line_number,
                local_box, global_box, normalized))

    document = ObjectDbDocument(
        generatedAt=datetime.now().astimezone().isoformat(timespec="microseconds"),
        dataRoot=data_path,
        tileCount=len(tiles),
        objectCount=len(objects),
        excludedLabelFileCount=excluded,
        tiles=tiles,
        objects=objects,
        warnings=warnings,
    )

    resolved_output_path = output_path or os.path.join(data_path, "object_db.json")
    output_dir = os.path.dirname(resolved_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(resolved_output_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(document), fh, ensure_ascii=False, indent=2)

    return ObjectDbBuildResult(resolved_output_path, len(tiles), len(objects), excluded, warnings)
