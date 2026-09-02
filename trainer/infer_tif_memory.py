import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
SENTINEL = object()


def parse_options(text):
    values = {}
    if not text:
        return values
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid option: {item}")
        key, value = item.split("=", 1)
        values[key.strip().lstrip("-")] = value.strip()
    return values


def get_float(options, key, default):
    return float(options.get(key, default))


def get_int(options, key, default):
    return int(float(options.get(key, default)))


def get_str(options, key, default):
    return str(options.get(key, default))


def get_bool(options, key, default):
    value = str(options.get(key, default)).strip().lower()
    return value in ("1", "true", "yes", "y", "on")


def split_source_paths(source_text):
    text = str(source_text).strip()
    if not text:
        return []
    # If source is a text file containing paths (e.g. @file_list.txt or file_list.txt)
    target_text_file = None
    if text.startswith("@"):
        target_text_file = Path(text[1:].strip().strip('"'))
    elif (text.endswith(".txt") or text.endswith(".lst")) and Path(text.strip('"')).is_file():
        target_text_file = Path(text.strip('"'))

    if target_text_file is not None and target_text_file.is_file():
        lines = target_text_file.read_text(encoding="utf-8").splitlines()
        return [Path(line.strip().strip('"')) for line in lines if line.strip() and not line.strip().startswith("#")]

    return [Path(item.strip().strip('"')) for item in text.replace("\r", ";").replace("\n", ";").split(";") if item.strip()]


def find_tif_paths(source_root, recursive):
    paths = []
    for source in split_source_paths(source_root):
        if source.is_file():
            if source.suffix.lower() in (".tif", ".tiff"):
                paths.append(source)
            continue

        if source.is_dir():
            iterator = source.rglob("*") if recursive else source.iterdir()
            paths.extend([p for p in iterator if p.is_file() and p.suffix.lower() in (".tif", ".tiff")])

    # De-duplicate while preserving sorted order
    seen = set()
    unique_paths = []
    for p in sorted(paths, key=lambda p: str(p).lower()):
        resolved = str(p.resolve()).lower()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(p)

    return unique_paths


def tile_starts(length, tile, stride):
    if length < tile:
        return []
    values = list(range(0, length - tile, stride))
    values.append(length - tile)
    return sorted(set(values))


def image_stats(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": 0, "p1": 0, "mean": 0, "p99": 0, "max": 0}
    return {
        "min": float(np.min(finite)),
        "p1": float(np.percentile(finite, 1)),
        "mean": float(np.mean(finite)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def scale_to_uint8(arr, preprocess):
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        if preprocess == "byteswap_shift":
            return (arr.byteswap() >> 8).astype(np.uint8)
        if preprocess == "minmax":
            arr32 = arr.astype(np.float32)
            lo = float(np.min(arr32))
            hi = float(np.max(arr32))
            if hi <= lo:
                hi = lo + 1.0
            return np.clip((arr32 - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        if preprocess == "percentile":
            arr32 = arr.astype(np.float32)
            finite = arr32[np.isfinite(arr32)]
            if finite.size == 0:
                return np.zeros(arr.shape, dtype=np.uint8)
            lo, hi = np.percentile(finite, (1, 99))
            if hi <= lo:
                hi = lo + 1.0
            return np.clip((arr32 - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        return (arr >> 8).astype(np.uint8)

    arr32 = arr.astype(np.float32)
    finite = arr32[np.isfinite(arr32)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr32 - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def to_uint8_rgb(image, preprocess="shift", invert=False):
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    raw_dtype = str(arr.dtype)
    raw_shape = list(arr.shape)
    raw_stats = image_stats(arr.astype(np.float32, copy=False))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    arr = scale_to_uint8(arr, preprocess)
    if invert:
        arr = 255 - arr
    arr = np.ascontiguousarray(arr)
    return arr, {
        "rawDtype": raw_dtype,
        "rawShape": raw_shape,
        "rawStats": raw_stats,
        "preprocess": preprocess,
        "invert": invert,
        "uint8Stats": image_stats(arr.astype(np.float32, copy=False)),
    }


def load_tif(tif_path, preprocess, invert):
    started = time.perf_counter()
    image, stats = to_uint8_rgb(tifffile.imread(tif_path), preprocess, invert)
    return {
        "path": tif_path,
        "image": image,
        "stats": stats,
        "loadSeconds": time.perf_counter() - started,
    }


def start_loader(tif_paths, prefetch, preprocess, invert):
    work_queue = queue.Queue(maxsize=max(1, prefetch))

    def worker():
        try:
            for tif_path in tif_paths:
                print(f"Prefetch loading {tif_path.name}", flush=True)
                work_queue.put(load_tif(tif_path, preprocess, invert))
        except Exception as exc:
            work_queue.put(exc)
        finally:
            work_queue.put(SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return work_queue, thread


def box_to_record(source_path, tile_name, tile_left, tile_top, cls_id, conf, xyxy):
    left = int(round(float(xyxy[0]))) + tile_left
    top = int(round(float(xyxy[1]))) + tile_top
    right = int(round(float(xyxy[2]))) + tile_left
    bottom = int(round(float(xyxy[3]))) + tile_top
    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    return {
        "candidateId": 0,
        "sourceBaseName": source_path.stem,
        "sourceTifName": source_path.name,
        "sourceTifPath": str(source_path),
        "tileName": tile_name,
        "tileLeft": tile_left,
        "tileTop": tile_top,
        "classId": int(cls_id),
        "className": "whale",
        "confidence": float(conf),
        "globalBox": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": right - left,
            "height": bottom - top,
        },
    }


def touches_internal_tile_edge(xyxy, tile_left, tile_top, tile_size, image_width, image_height, margin):
    if margin <= 0:
        return False
    local_left = float(xyxy[0])
    local_top = float(xyxy[1])
    local_right = float(xyxy[2])
    local_bottom = float(xyxy[3])
    touches_left = local_left <= margin and tile_left > 0
    touches_top = local_top <= margin and tile_top > 0
    touches_right = local_right >= tile_size - margin and tile_left + tile_size < image_width
    touches_bottom = local_bottom >= tile_size - margin and tile_top + tile_size < image_height
    return touches_left or touches_top or touches_right or touches_bottom


def iou(first, second):
    a = first["globalBox"]
    b = second["globalBox"]
    left = max(a["left"], b["left"])
    top = max(a["top"], b["top"])
    right = min(a["right"], b["right"])
    bottom = min(a["bottom"], b["bottom"])
    inter = max(0, right - left) * max(0, bottom - top)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return 0.0 if union <= 0 else inter / union


def merge_candidates(candidates, merge_iou, start_id):
    kept = []
    for cand in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if any(iou(existing, cand) >= merge_iou for existing in kept):
            continue
        kept.append(cand)

    kept.sort(key=lambda item: (item["sourceTifName"].lower(), item["globalBox"]["top"], item["globalBox"]["left"]))
    for idx, cand in enumerate(kept, start=start_id):
        cand["candidateId"] = idx
    return kept


def yolo_label_for_box(box, crop_left, crop_top, crop_width, crop_height):
    visible_left = max(box["left"], crop_left)
    visible_top = max(box["top"], crop_top)
    visible_right = min(box["right"], crop_left + crop_width)
    visible_bottom = min(box["bottom"], crop_top + crop_height)
    width = max(1, visible_right - visible_left)
    height = max(1, visible_bottom - visible_top)
    local_left = visible_left - crop_left
    local_top = visible_top - crop_top
    center_x = (local_left + width / 2.0) / crop_width
    center_y = (local_top + height / 2.0) / crop_height
    return f"0 {center_x:.6f} {center_y:.6f} {width / crop_width:.6f} {height / crop_height:.6f}"


def save_candidate_assets(candidates, image, run_root, crop_size, context, candidate_view):
    candidate_root = run_root / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "classes.txt").write_text("whale\n", encoding="utf-8")
    (candidate_root / "predefined_classes.txt").write_text("whale\n", encoding="utf-8")
    height, width = image.shape[:2]
    for cand in candidates:
        box = cand["globalBox"]
        if candidate_view == "tile":
            requested = min(int(crop_size), width, height)
            crop_left = max(0, min(int(cand["tileLeft"]), width - requested))
            crop_top = max(0, min(int(cand["tileTop"]), height - requested))
        else:
            center_x = (box["left"] + box["right"]) / 2.0
            center_y = (box["top"] + box["bottom"]) / 2.0
            requested = max(crop_size, box["width"] + context * 2, box["height"] + context * 2)
            requested = min(int(requested), width, height)
            crop_left = int(round(center_x - requested / 2.0))
            crop_top = int(round(center_y - requested / 2.0))
            crop_left = max(0, min(crop_left, width - requested))
            crop_top = max(0, min(crop_top, height - requested))
        crop = np.ascontiguousarray(image[crop_top : crop_top + requested, crop_left : crop_left + requested, :])

        stem = f"cand{cand['candidateId']:06d}_{Path(cand['sourceTifPath']).stem}_conf{cand['confidence']:.3f}"
        image_path = candidate_root / f"{stem}.png"
        label_path = candidate_root / f"{stem}.txt"
        info_path = candidate_root / f"{stem}.json"
        crop_image = Image.fromarray(crop)
        crop_image.save(image_path)
        label_path.write_text(yolo_label_for_box(box, crop_left, crop_top, requested, requested) + "\n", encoding="utf-8")
        asset_info = {
            "candidateId": cand["candidateId"],
            "sourceTifPath": cand["sourceTifPath"],
            "confidence": cand["confidence"],
            "cropBox": {
                "left": crop_left,
                "top": crop_top,
                "right": crop_left + requested,
                "bottom": crop_top + requested,
                "width": requested,
                "height": requested,
            },
            "globalBox": box,
            "candidateView": candidate_view,
            "imagePath": str(image_path),
            "labelPath": str(label_path),
        }
        info_path.write_text(json.dumps(asset_info, ensure_ascii=False, indent=2), encoding="utf-8")
        cand["candidateImagePath"] = str(image_path)
        cand["candidateLabelPath"] = str(label_path)
        cand["candidateInfoPath"] = str(info_path)
        cand["candidateCropBox"] = asset_info["cropBox"]


def write_documents(run_root, source_root, model_path, options_text, tile_count, all_candidates, per_tif_records, progress):
    document = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceTifFolderPath": str(source_root),
        "modelPath": str(Path(model_path)),
        "runRootPath": str(run_root),
        "options": options_text,
        "tileCount": tile_count,
        "candidateCount": len(all_candidates),
        "candidates": all_candidates,
    }
    (run_root / "candidates.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    by_tif_root = run_root / "candidates_by_tif"
    by_tif_root.mkdir(parents=True, exist_ok=True)
    for record in per_tif_records:
        path = by_tif_root / f"{Path(record['sourceTifPath']).stem}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def append_raw_candidates(raw_path, candidates):
    if not candidates:
        return
    with raw_path.open("a", encoding="utf-8") as handle:
        for cand in candidates:
            handle.write(json.dumps(cand, ensure_ascii=False) + "\n")


def write_progress(run_root, progress):
    (run_root / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def run_batch(model, tiles, meta, args, candidates, raw_path=None):
    if not tiles:
        return {"added": 0, "filteredEdge": 0}
    # Ultralytics treats ndarray inputs as OpenCV-style BGR, while saved PNG/PIL
    # inputs are RGB. Training images are loaded from files, so mirror that path.
    predict_tiles = [np.ascontiguousarray(tile[..., ::-1]) for tile in tiles]
    results = model.predict(
        source=predict_tiles,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=None if str(args.device).lower() == "auto" else args.device,
        batch=len(tiles),
        max_det=args.max_det,
        save=False,
        save_txt=False,
        verbose=False,
    )
    added = []
    filtered_edge = 0
    for result, info in zip(results, meta):
        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, conf, cls_id in zip(xyxy, confs, classes):
            if args.edge_filter and touches_internal_tile_edge(
                box,
                info["x"],
                info["y"],
                info["tile_size"],
                info["image_width"],
                info["image_height"],
                args.edge_margin,
            ):
                filtered_edge += 1
                continue
            cand = box_to_record(info["source_path"], info["tile_name"], info["x"], info["y"], cls_id, conf, box)
            candidates.append(cand)
            added.append(cand)
    if raw_path is not None:
        append_raw_candidates(raw_path, added)
    return {"added": len(added), "filteredEdge": filtered_edge}


def process_loaded_tif(model, loaded, args, tile, stride, batch, limit_remaining, progress_every, raw_path, run_root, run_progress):
    tif_path = loaded["path"]
    image = loaded["image"]
    height, width = image.shape[:2]
    xs = tile_starts(width, tile, stride)
    ys = tile_starts(height, tile, stride)
    expected_tiles = len(xs) * len(ys)
    candidates = []
    batch_tiles = []
    batch_meta = []
    processed = 0
    debug_saved = 0
    filtered_edge = 0
    started = time.perf_counter()
    stop = False

    for y in ys:
        for x in xs:
            if limit_remaining is not None and processed >= limit_remaining:
                stop = True
                break
            tile_array = np.ascontiguousarray(image[y : y + tile, x : x + tile, :])
            tile_name = f"{tif_path.stem}__x{x}_y{y}_s{tile}"
            if args.debug_tiles > 0 and debug_saved < args.debug_tiles:
                debug_root = run_root / "debug_tiles" / tif_path.stem
                debug_root.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile_array).save(debug_root / f"{tile_name}.png")
                debug_saved += 1
            batch_tiles.append(tile_array)
            batch_meta.append({
                "source_path": tif_path,
                "tile_name": tile_name,
                "x": x,
                "y": y,
                "tile_size": tile,
                "image_width": width,
                "image_height": height,
            })
            processed += 1
            if len(batch_tiles) >= batch:
                batch_result = run_batch(model, batch_tiles, batch_meta, args, candidates, raw_path)
                filtered_edge += batch_result["filteredEdge"]
                batch_tiles.clear()
                batch_meta.clear()
                if processed % progress_every == 0:
                    print(
                        f"{tif_path.name}: processed {processed}/{expected_tiles} tiles, "
                        f"raw_candidates={len(candidates)}, edge_filtered={filtered_edge}",
                        flush=True,
                    )
                    run_progress.update(
                        {
                            "updatedAt": datetime.now(timezone.utc).isoformat(),
                            "currentFile": tif_path.name,
                            "currentFileProcessedTiles": processed,
                            "currentFileExpectedTiles": expected_tiles,
                            "currentFileRawCandidates": len(candidates),
                            "currentFileEdgeFiltered": filtered_edge,
                            "phase": "infer",
                        }
                    )
                    write_progress(run_root, run_progress)
        if stop:
            break

    batch_result = run_batch(model, batch_tiles, batch_meta, args, candidates, raw_path)
    filtered_edge += batch_result["filteredEdge"]
    return {
        "sourcePath": tif_path,
        "image": image,
        "width": width,
        "height": height,
        "expectedTiles": expected_tiles,
        "tileCount": processed,
        "rawCandidates": candidates,
        "edgeFilteredCount": filtered_edge,
        "inferSeconds": time.perf_counter() - started,
        "limitReached": stop,
        "debugTileCount": debug_saved,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run_name", default="infer_whale")
    parser.add_argument("--options", default="")
    args = parser.parse_args()

    options = parse_options(args.options)
    tile = get_int(options, "tile", 640)
    overlap = get_float(options, "overlap", 0.2)
    stride = max(1, int(round(tile * (1.0 - overlap))))
    batch_value = get_str(options, "batch", "8")
    batch = 8 if batch_value.lower() == "auto" else max(1, int(float(batch_value)))
    args.imgsz = get_int(options, "imgsz", tile)
    args.conf = get_float(options, "conf", 0.1)
    args.iou = get_float(options, "iou", 0.6)
    args.device = get_str(options, "device", "0")
    args.max_det = get_int(options, "max_det", 300)
    args.edge_filter = get_bool(options, "edge_filter", True)
    args.edge_margin = get_int(options, "edge_margin", 32)
    limit = get_int(options, "limit", 0)
    merge_iou = get_float(options, "merge_iou", 0.5)
    crop_size = get_int(options, "candidate_crop", tile)
    crop_context = get_int(options, "candidate_context", 120)
    candidate_view = get_str(options, "candidate_view", "tile").lower()
    if candidate_view not in ("tile", "center"):
        raise ValueError("candidate_view must be tile or center")
    prefetch = get_int(options, "prefetch", 3)
    progress_every = max(1, get_int(options, "progress_every", 100))
    args.debug_tiles = get_int(options, "debug_tiles", 0)
    recursive = get_bool(options, "recursive", True)
    preprocess = get_str(options, "preprocess", "shift").lower()
    if preprocess not in ("shift", "percentile", "minmax", "byteswap_shift"):
        raise ValueError("preprocess must be shift, percentile, minmax, or byteswap_shift")
    invert = get_bool(options, "invert", False)

    resume = get_bool(options, "resume", True)

    from ultralytics import YOLO

    source_root = Path(args.source)
    run_root = Path(args.output) / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "candidates").mkdir(parents=True, exist_ok=True)
    (run_root / "candidates_by_tif").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    raw_candidates_path = run_root / "raw_candidates.jsonl"
    if not resume or not raw_candidates_path.exists():
        raw_candidates_path.write_text("", encoding="utf-8")

    all_tif_paths = find_tif_paths(source_root, recursive)
    if not all_tif_paths:
        raise FileNotFoundError(f"No TIF files were found under {source_root} (recursive={recursive})")

    all_candidates = []
    per_tif_records = []
    total_tiles = 0
    next_candidate_id = 1
    processed_stems = set()

    # Load existing progress if resuming
    by_tif_root = run_root / "candidates_by_tif"
    if resume and by_tif_root.is_dir():
        for tif_p in all_tif_paths:
            record_path = by_tif_root / f"{tif_p.stem}.json"
            if record_path.is_file():
                try:
                    record_data = json.loads(record_path.read_text(encoding="utf-8"))
                    per_tif_records.append(record_data)
                    cands = record_data.get("candidates", [])
                    all_candidates.extend(cands)
                    total_tiles += record_data.get("tileCount", 0)
                    for c in cands:
                        cid = c.get("candidateId", 0)
                        if isinstance(cid, int) and cid >= next_candidate_id:
                            next_candidate_id = cid + 1
                    processed_stems.add(tif_p.stem.lower())
                except Exception:
                    pass

    tif_paths = [p for p in all_tif_paths if p.stem.lower() not in processed_stems]
    completed_files = len(processed_stems)

    run_started = time.perf_counter()
    progress = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "completedFiles": completed_files,
        "totalFiles": len(all_tif_paths),
        "tileCount": total_tiles,
        "candidateCount": len(all_candidates),
        "rawCandidatesPath": str(raw_candidates_path),
        "completed": len(tif_paths) == 0,
        "phase": "start",
    }
    write_documents(run_root, source_root, args.model, args.options, total_tiles, all_candidates, per_tif_records, progress)

    print(f"Run root: {run_root}", flush=True)
    print(f"Source input: {args.source}", flush=True)
    if completed_files > 0:
        if tif_paths:
            next_num = completed_files + 1
            print(f"[RESUME] Found {completed_files} already completed files in '{run_root.name}'. Resuming from file #{next_num}/{len(all_tif_paths)} ({len(tif_paths)} files remaining).", flush=True)
        else:
            print(f"[RESUME] All {len(all_tif_paths)} files in '{run_root.name}' are already completed.", flush=True)
    print(
        f"Tiling memory: tile={tile}, overlap={overlap}, stride={stride}, batch={batch}, "
        f"prefetch={prefetch}, recursive={recursive}, files={len(all_tif_paths)}, "
        f"preprocess={preprocess}, invert={invert}, edge_filter={args.edge_filter}, "
        f"edge_margin={args.edge_margin}",
        flush=True,
    )

    if tif_paths:
        work_queue, loader = start_loader(tif_paths, prefetch, preprocess, invert)
        while True:
            loaded = work_queue.get()
            if loaded is SENTINEL:
                break
            if isinstance(loaded, Exception):
                raise loaded

            tif_path = loaded["path"]
            progress.update(
                {
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "currentFile": tif_path.name,
                    "phase": "loaded",
                    "lastLoadSeconds": loaded["loadSeconds"],
                    "lastImageStats": loaded["stats"],
                }
            )
            write_progress(run_root, progress)
            print(
                f"Processing {tif_path.name} loaded_in={loaded['loadSeconds']:.2f}s "
                f"stats={json.dumps(loaded['stats']['uint8Stats'])}",
                flush=True,
            )
            remaining = None if limit <= 0 else max(0, limit - total_tiles)
            processed = process_loaded_tif(
                model,
                loaded,
                args,
                tile,
                stride,
                batch,
                remaining,
                progress_every,
                raw_candidates_path,
                run_root,
                progress,
            )
            merged = merge_candidates(processed["rawCandidates"], merge_iou, next_candidate_id)
            next_candidate_id += len(merged)
            save_candidate_assets(merged, processed["image"], run_root, crop_size, crop_context, candidate_view)
            total_tiles += processed["tileCount"]
            all_candidates.extend(merged)
            completed_files += 1

            tif_record = {
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "sourceTifPath": str(tif_path),
                "sourceTifName": tif_path.name,
                "width": processed["width"],
                "height": processed["height"],
                "tileCount": processed["tileCount"],
                "rawCandidateCount": len(processed["rawCandidates"]),
                "edgeFilteredCount": processed["edgeFilteredCount"],
                "candidateCount": len(merged),
                "loadSeconds": loaded["loadSeconds"],
                "inferSeconds": processed["inferSeconds"],
                "debugTileCount": processed["debugTileCount"],
                "imageStats": loaded["stats"],
                "candidates": merged,
            }
            per_tif_records.append(tif_record)
            progress.update(
                {
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "completedFiles": completed_files,
                    "totalFiles": len(all_tif_paths),
                    "tileCount": total_tiles,
                    "candidateCount": len(all_candidates),
                    "lastCompleted": tif_path.name,
                    "elapsedSeconds": time.perf_counter() - run_started,
                    "phase": "saved_tif",
                }
            )
            write_documents(run_root, source_root, args.model, args.options, total_tiles, all_candidates, per_tif_records, progress)
            print(
                f"{tif_path.name}: {processed['width']}x{processed['height']}, "
                f"tiles={processed['tileCount']}, raw={len(processed['rawCandidates'])}, "
                f"edge_filtered={processed['edgeFilteredCount']}, "
                f"merged={len(merged)}, infer={processed['inferSeconds']:.2f}s",
                flush=True,
            )
            print(f"Saved intermediate candidates: {run_root / 'candidates.json'}", flush=True)

            del processed
            del loaded
            if limit > 0 and total_tiles >= limit:
                break

        loader.join(timeout=1)
    final_progress = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "completedFiles": completed_files,
        "totalFiles": len(tif_paths),
        "tileCount": total_tiles,
        "candidateCount": len(all_candidates),
        "completed": True,
        "rawCandidatesPath": str(raw_candidates_path),
        "phase": "complete",
        "elapsedSeconds": time.perf_counter() - run_started,
    }
    write_documents(run_root, source_root, args.model, args.options, total_tiles, all_candidates, per_tif_records, final_progress)
    print(f"Candidates: {len(all_candidates)}", flush=True)
    print(f"Candidate JSON: {run_root / 'candidates.json'}", flush=True)


if __name__ == "__main__":
    main()
