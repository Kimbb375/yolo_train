"""6번 원본 추론: 원본 TIF -> 내부 타일링 -> YOLO 추론 -> candidates.json.

C# InferenceTilingRunner.cs 포팅. 5번(학습)과 같은 패턴으로 trainer/infer_tiles.py를
서브프로세스 없이 같은 프로세스 안에서 import해서 실행함 (trainerscript.py 공용).

범위: C# 원본의 'full'(디스크 타일링) 경로만 포팅함. 'memory' 타일 모드
(trainer/infer_tif_memory.py, 디스크에 타일 안 쓰고 메모리에서 바로 추론)는 아직 미포함
— docs/pyside6-env-setup.md 참고.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from PIL import Image

import sourceimage
import trainerscript
from objectdb import CoordinateBox
from trainingdataset import _format_six

TILE_NAME_TEMPLATE = "{source}__x{left}_y{top}_s{size}"


def _parse_option_text(text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    if not text:
        return options
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        options[key.strip().lower()] = value.strip()
    return options


def _get_str(options: dict, key: str, default: str) -> str:
    return options.get(key, default)


def _get_int(options: dict, key: str, default: int) -> int:
    return int(options[key]) if key in options else default


def _get_double(options: dict, key: str, default: float) -> float:
    return float(options[key]) if key in options else default


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return min(maximum, max(minimum, value))


def _enumerate_starts(length: int, tile_size: int, stride: int):
    last = length - tile_size
    value = 0
    while value < last:
        yield value
        value += stride
    yield last


def enumerate_tiles(source_width: int, source_height: int, tile_size: int, stride: int):
    """C# EnumerateTiles/EnumerateStarts 포팅. 마지막 타일은 항상 우/하단에 맞춰짐(겹침 허용)."""
    if source_width < tile_size or source_height < tile_size:
        return
    for top in _enumerate_starts(source_height, tile_size, stride):
        for left in _enumerate_starts(source_width, tile_size, stride):
            yield left, top


def _enumerate_source_tif_files(source_path: str, recursive: bool = True) -> list[str]:
    found: list[str] = []
    for part in re.split(r"[;\r\n]+", source_path):
        part = part.strip().strip('"')
        if not part:
            continue
        if os.path.isfile(part):
            if part.lower().endswith((".tif", ".tiff")):
                found.append(part)
            continue
        if os.path.isdir(part):
            for extension in ("*.tif", "*.tiff"):
                pattern = os.path.join(part, "**", extension) if recursive else os.path.join(part, extension)
                found.extend(glob.glob(pattern, recursive=recursive))
    return sorted(set(found), key=str.lower)


@dataclass
class TileInfo:
    name: str
    path: str
    sourceTifPath: str
    sourceBaseName: str
    left: int
    top: int
    width: int
    height: int


def _create_tiles_for_source(tif_path: str, tiles_root: str, tile_size: int,
                              bounds_list: list[tuple[int, int]]) -> list[TileInfo]:
    source_base_name = os.path.splitext(os.path.basename(tif_path))[0]
    array = sourceimage.read_full(tif_path)
    tiles = []
    for left, top in bounds_list:
        crop = array[top:top + tile_size, left:left + tile_size]
        name = TILE_NAME_TEMPLATE.format(source=source_base_name, left=left, top=top, size=tile_size)
        path = os.path.join(tiles_root, name + ".png")
        Image.fromarray(crop).save(path, format="PNG")
        tiles.append(TileInfo(name, path, tif_path, source_base_name, left, top, tile_size, tile_size))
    return sorted(tiles, key=lambda t: (t.top, t.left))


def _yolo_to_local_box(center_x: float, center_y: float, width: float, height: float,
                        image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """C# YoloToRectangle 포팅. Math.Round 기본값(banker's rounding) = 파이썬 round()와 동일."""
    pixel_width = width * image_width
    pixel_height = height * image_height
    left = round(center_x * image_width - pixel_width / 2.0)
    top = round(center_y * image_height - pixel_height / 2.0)
    right = round(center_x * image_width + pixel_width / 2.0)
    bottom = round(center_y * image_height + pixel_height / 2.0)
    left = _clamp(left, 0, image_width - 1)
    top = _clamp(top, 0, image_height - 1)
    right = _clamp(right, left + 1, image_width)
    bottom = _clamp(bottom, top + 1, image_height)
    return left, top, right, bottom


def read_candidates(tile_infos: list[TileInfo], labels_root: str) -> list[dict]:
    tile_map = {tile.name.lower(): tile for tile in tile_infos}
    candidates: list[dict] = []
    if not os.path.isdir(labels_root):
        return candidates

    for label_path in sorted(glob.glob(os.path.join(labels_root, "*.txt")), key=str.lower):
        tile_name = os.path.splitext(os.path.basename(label_path))[0]
        tile = tile_map.get(tile_name.lower())
        if tile is None:
            continue

        with open(label_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 6:
                    continue
                class_id = int(parts[0])
                center_x, center_y, width, height, confidence = (float(p) for p in parts[1:6])
                local_left, local_top, local_right, local_bottom = _yolo_to_local_box(
                    center_x, center_y, width, height, tile.width, tile.height)
                global_box = CoordinateBox(
                    tile.left + local_left, tile.top + local_top,
                    tile.left + local_right, tile.top + local_bottom,
                    local_right - local_left, local_bottom - local_top)
                candidates.append({
                    "sourceBaseName": tile.sourceBaseName,
                    "sourceTifName": os.path.basename(tile.sourceTifPath),
                    "sourceTifPath": tile.sourceTifPath,
                    "tileName": tile.name,
                    "tileLeft": tile.left,
                    "tileTop": tile.top,
                    "classId": class_id,
                    "className": "whale",
                    "confidence": confidence,
                    "globalBox": global_box,
                })
    return candidates


def _iou(a: CoordinateBox, b: CoordinateBox) -> float:
    left, top = max(a.left, b.left), max(a.top, b.top)
    right, bottom = min(a.right, b.right), min(a.bottom, b.bottom)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a.width * a.height + b.width * b.height - intersection
    return 0.0 if union <= 0 else intersection / union


def merge_candidates(candidates: list[dict], merge_iou: float) -> list[dict]:
    """C# MergeCandidates 포팅: 소스별로 confidence 내림차순 그리디 NMS."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["sourceTifPath"].lower()].append(candidate)

    merged: list[dict] = []
    for group in grouped.values():
        kept: list[dict] = []
        for candidate in sorted(group, key=lambda c: -c["confidence"]):
            if any(_iou(existing["globalBox"], candidate["globalBox"]) >= merge_iou for existing in kept):
                continue
            kept.append(candidate)
        merged.extend(kept)

    merged.sort(key=lambda c: (c["sourceTifName"].lower(), c["globalBox"].top, c["globalBox"].left))
    for index, candidate in enumerate(merged, start=1):
        candidate["candidateId"] = index
    return merged


def _build_candidate_yolo_label(box: CoordinateBox, crop_bounds: tuple[int, int, int, int]) -> str:
    left, top, right, bottom = crop_bounds
    visible_left, visible_top = max(box.left, left), max(box.top, top)
    visible_right, visible_bottom = min(box.right, right), min(box.bottom, bottom)
    width = max(1, visible_right - visible_left)
    height = max(1, visible_bottom - visible_top)
    crop_width, crop_height = right - left, bottom - top
    center_x = (visible_left - left + width / 2.0) / crop_width
    center_y = (visible_top - top + height / 2.0) / crop_height
    return (f"0 {_format_six(center_x)} {_format_six(center_y)} "
            f"{_format_six(width / crop_width)} {_format_six(height / crop_height)}")


def save_candidate_assets(candidates: list[dict], run_root: str, options: dict) -> None:
    """C# SaveCandidateAssets 포팅 + 확장: candidateImagePath 등을 candidates.json 본문에도 채워 넣음
    (C# 원본은 이 경로들을 info.json 사이드카에만 쓰고 candidates.json 본문에는 없음 —
    7/8번이 파일명 역추적 없이 바로 쓸 수 있게 여기서는 처음부터 포함시킴. §알아둘 것 참고)."""
    if not candidates:
        return

    candidates_root = os.path.join(run_root, "candidates")
    os.makedirs(candidates_root, exist_ok=True)
    with open(os.path.join(candidates_root, "classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("whale\n")
    with open(os.path.join(candidates_root, "predefined_classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("whale\n")

    crop_size = _get_int(options, "candidate_crop", _get_int(options, "tile", 640))
    context = _get_int(options, "candidate_context", 120)
    candidate_view = _get_str(options, "candidate_view", "tile")
    size_cache: dict[str, tuple[int, int]] = {}
    array_cache: dict[str, object] = {}

    for candidate in candidates:
        source_path = candidate["sourceTifPath"]
        if source_path not in size_cache:
            size_cache[source_path] = sourceimage.get_size(source_path)
        source_width, source_height = size_cache[source_path]
        box: CoordinateBox = candidate["globalBox"]

        if candidate_view.lower() == "tile":
            requested = min(source_width, source_height, crop_size)
            left = _clamp(candidate["tileLeft"], 0, source_width - requested)
            top = _clamp(candidate["tileTop"], 0, source_height - requested)
        else:
            requested = min(source_width, source_height,
                             max(crop_size, box.width + context * 2, box.height + context * 2))
            center_x, center_y = (box.left + box.right) / 2.0, (box.top + box.bottom) / 2.0
            left = _clamp(round(center_x - requested / 2.0), 0, source_width - requested)
            top = _clamp(round(center_y - requested / 2.0), 0, source_height - requested)
        crop_bounds = (left, top, left + requested, top + requested)

        if source_path not in array_cache:
            array_cache[source_path] = sourceimage.read_full(source_path)
        crop_array = array_cache[source_path][top:top + requested, left:left + requested]

        stem = f"cand{candidate['candidateId']:06d}_{candidate['sourceBaseName']}_conf{candidate['confidence']:.3f}"
        image_path = os.path.join(candidates_root, stem + ".png")
        label_path = os.path.join(candidates_root, stem + ".txt")
        info_path = os.path.join(candidates_root, stem + ".json")
        Image.fromarray(crop_array).save(image_path, format="PNG")
        with open(label_path, "w", encoding="utf-8") as fh:
            fh.write(_build_candidate_yolo_label(box, crop_bounds) + "\n")

        crop_box_dict = {"left": left, "top": top, "right": left + requested, "bottom": top + requested,
                          "width": requested, "height": requested}
        info = {
            "candidateId": candidate["candidateId"], "sourceTifPath": source_path,
            "confidence": candidate["confidence"], "cropBox": crop_box_dict,
            "globalBox": {"left": box.left, "top": box.top, "right": box.right, "bottom": box.bottom,
                          "width": box.width, "height": box.height},
            "candidateView": candidate_view, "imagePath": image_path, "labelPath": label_path,
        }
        with open(info_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=2)

        candidate["candidateImagePath"] = image_path
        candidate["candidateLabelPath"] = label_path
        candidate["candidateInfoPath"] = info_path
        candidate["candidateCropBox"] = crop_box_dict


def _reset_directory(path: str) -> None:
    import shutil
    if os.path.isdir(path):
        shutil.rmtree(path)


@dataclass
class InferenceRunResult:
    runRootPath: str
    tilesRootPath: str
    candidateJsonPath: str
    tileCount: int
    candidateCount: int

    def to_display_text(self) -> str:
        return ("[OK] inference completed\n\n"
                f"Run root       : {self.runRootPath}\n"
                f"Candidate JSON : {self.candidateJsonPath}\n"
                f"Tiles          : {self.tileCount}\n"
                f"Candidates     : {self.candidateCount}")


def _candidate_to_dict(candidate: dict) -> dict:
    box: CoordinateBox = candidate["globalBox"]
    return {
        "candidateId": candidate["candidateId"],
        "sourceBaseName": candidate["sourceBaseName"],
        "sourceTifName": candidate["sourceTifName"],
        "sourceTifPath": candidate["sourceTifPath"],
        "tileName": candidate["tileName"],
        "tileLeft": candidate["tileLeft"],
        "tileTop": candidate["tileTop"],
        "classId": candidate["classId"],
        "className": candidate["className"],
        "confidence": candidate["confidence"],
        "globalBox": {"left": box.left, "top": box.top, "right": box.right, "bottom": box.bottom,
                      "width": box.width, "height": box.height},
        "candidateImagePath": candidate.get("candidateImagePath", ""),
        "candidateLabelPath": candidate.get("candidateLabelPath", ""),
        "candidateInfoPath": candidate.get("candidateInfoPath", ""),
        "candidateCropBox": candidate.get("candidateCropBox"),
    }


def run(source_tif_folder: str, output_root: str, model_path: str, run_name: Optional[str],
        option_text: str) -> InferenceRunResult:
    options = _parse_option_text(option_text)
    tile_size = _get_int(options, "tile", 640)
    overlap = _get_double(options, "overlap", 0.2)
    stride = max(1, round(tile_size * (1.0 - overlap)))
    merge_iou = _get_double(options, "merge_iou", 0.5)
    limit = _get_int(options, "limit", 0)
    recursive = _get_str(options, "recursive", "1").lower() not in ("0", "false")

    run_name = (run_name or "").strip() or f"infer_{datetime.now():%Y%m%d_%H%M%S}"
    run_root = os.path.join(output_root, run_name)
    tiles_root = os.path.join(run_root, "tiles")
    predict_root = os.path.join(run_root, "predict")
    _reset_directory(run_root)
    os.makedirs(tiles_root, exist_ok=True)
    os.makedirs(predict_root, exist_ok=True)

    tif_paths = _enumerate_source_tif_files(source_tif_folder, recursive)
    tile_infos: list[TileInfo] = []
    for tif_path in tif_paths:
        width, height = sourceimage.get_size(tif_path)
        bounds_list = list(enumerate_tiles(width, height, tile_size, stride))
        if limit > 0:
            remaining = limit - len(tile_infos)
            if remaining <= 0:
                break
            bounds_list = bounds_list[:remaining]
        if not bounds_list:
            continue
        tile_infos.extend(_create_tiles_for_source(tif_path, tiles_root, tile_size, bounds_list))
        if limit > 0 and len(tile_infos) >= limit:
            break

    if not tile_infos:
        raise ValueError("No tiles were generated.")

    predict_args = [
        "--model", model_path,
        "--source", tiles_root,
        "--project", predict_root,
        "--name", "predict",
        "--imgsz", _get_str(options, "imgsz", _get_str(options, "tile", "640")),
        "--conf", _get_str(options, "conf", "0.1"),
        "--iou", _get_str(options, "iou", "0.6"),
        "--device", _get_str(options, "device", "0"),
        "--batch", _get_str(options, "batch", "auto"),
        "--max_det", _get_str(options, "max_det", "300"),
    ]
    trainerscript.run("infer_tiles", predict_args)

    labels_root = os.path.join(predict_root, "predict", "labels")
    candidates = read_candidates(tile_infos, labels_root)
    candidates = merge_candidates(candidates, merge_iou)
    save_candidate_assets(candidates, run_root, options)

    document = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "sourceTifFolderPath": source_tif_folder,
        "modelPath": model_path,
        "runRootPath": run_root,
        "options": option_text,
        "tileCount": len(tile_infos),
        "candidateCount": len(candidates),
        "candidates": [_candidate_to_dict(c) for c in candidates],
    }
    candidate_json_path = os.path.join(run_root, "candidates.json")
    with open(candidate_json_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, ensure_ascii=False, indent=2)

    return InferenceRunResult(run_root, tiles_root, candidate_json_path, len(tile_infos), len(candidates))
