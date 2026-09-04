"""원본 TIF 소스 이미지 탐색/읽기. C# SourceImageResolver.cs 포팅 (docs/domain-rules.md §1.4)."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import tifffile


def _leading_date_part(value: str) -> str:
    result = []
    for ch in value:
        if ch.isdigit():
            result.append(ch)
        else:
            break
    return "".join(result)


def _date_folder_candidates(capture_date: str) -> Iterable[str]:
    if capture_date:
        yield capture_date

    date_token = capture_date or ""
    date_part = _leading_date_part(date_token)
    suffix = date_token[len(date_part):]
    digits = "".join(ch for ch in date_part if ch.isdigit())

    if len(digits) >= 8:
        yield digits[-6:]
        yield digits[-4:]
        if suffix:
            yield digits[-6:] + suffix
            yield digits[-4:] + suffix
    elif len(digits) == 6:
        yield digits
        yield digits[-4:]
        if suffix:
            yield digits + suffix
            yield digits[-4:] + suffix

    if not suffix and len(digits) >= 6:
        yield digits[-6:] + "(AM)"
        yield digits[-6:] + "(PM)"
    if not suffix and len(digits) >= 4:
        yield digits[-4:] + "(AM)"
        yield digits[-4:] + "(PM)"


def _source_file_name_candidates(source_base_name: str, source_tif_name: str, capture_date: str) -> Iterable[str]:
    base_names = [os.path.splitext(source_tif_name)[0], source_base_name]

    date_token = capture_date or ""
    date_part = _leading_date_part(date_token)
    suffix = date_token[len(date_part):]
    digits = "".join(ch for ch in date_part if ch.isdigit())

    if len(digits) >= 8:
        base_names.append(digits[-6:] + "_" + source_base_name)
        if suffix:
            base_names.append(digits[-6:] + suffix + "_" + source_base_name)
        else:
            base_names.append(digits[-6:] + "(AM)_" + source_base_name)
            base_names.append(digits[-6:] + "(PM)_" + source_base_name)
    elif len(digits) == 6:
        base_names.append(digits + "_" + source_base_name)
        if suffix:
            base_names.append(digits + suffix + "_" + source_base_name)
        else:
            base_names.append(digits + "(AM)_" + source_base_name)
            base_names.append(digits + "(PM)_" + source_base_name)

    seen = set()
    for name in base_names:
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        for extension in (".tif", ".tiff", ".TIF", ".TIFF"):
            yield name + extension


def resolve(source_root_path: str, capture_date: str, source_base_name: str, source_tif_name: str) -> str:
    if not source_root_path:
        raise ValueError("Source root path is empty.")

    seen = set()

    def try_candidate(path: str) -> str | None:
        key = path.lower()
        if key in seen:
            return None
        seen.add(key)
        return path if os.path.exists(path) else None

    for date_folder in _date_folder_candidates(capture_date):
        date_dir = os.path.join(source_root_path, date_folder)
        for file_name in _source_file_name_candidates(source_base_name, source_tif_name, capture_date):
            found = try_candidate(os.path.join(date_dir, file_name))
            if found:
                return found

    for file_name in _source_file_name_candidates(source_base_name, source_tif_name, capture_date):
        found = try_candidate(os.path.join(source_root_path, file_name))
        if found:
            return found

    raise FileNotFoundError(
        f"Source TIF was not found. Source root: {source_root_path}, "
        f"date: {capture_date}, image: {source_tif_name}")


def get_size(path: str) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        return page.imagewidth, page.imagelength


def read_full(path: str) -> np.ndarray:
    return tifffile.imread(path)


# ponytail: memmap only works for uncompressed/non-tiled/native-byte-order TIFFs (tifffile raises
# ValueError otherwise) - falls back to the old full-decode-then-slice path for anything else.
# Memmap lets the OS page in only the rows actually touched by the slice instead of decoding the
# whole (often multi-gigapixel survey) image on every single crop request - this was the viewer's
# main slowness source (2번 원본 검증 뷰어).
def read_crop(path: str, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    try:
        mapped = tifffile.memmap(path, mode="r")
    except ValueError:
        return read_full(path)[top:bottom, left:right]
    return np.array(mapped[top:bottom, left:right])
