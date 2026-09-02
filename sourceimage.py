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


# ponytail: 전체 이미지를 메모리에 다 읽은 뒤 슬라이스함 (윈도우/타일 단위 부분 읽기 아님).
# 실제 원본 TIF(수만x수만 픽셀급)로 아직 검증 못 해서 성능 최적화는 보류함.
# 느리면 tifffile.TiffFile(path).aszarr() + zarr 슬라이싱으로 창 단위 읽기로 바꿀 것.
def read_full(path: str) -> np.ndarray:
    return tifffile.imread(path)


def read_crop(path: str, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    return read_full(path)[top:bottom, left:right]
