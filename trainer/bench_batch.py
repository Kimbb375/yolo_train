"""6번 배치 크기 자동 최적화: 이 GPU/모델/타일 크기로 실제 model.predict를 돌려가며
배치를 1,2,4,8...로 늘려서 "진짜 OOM이 날 때까지" 밀어붙임(처리량 향상폭으로 중간에
멈추지 않음 - 속도 정체와 메모리 한계는 다른 문제라 속도만 보고 멈추면 실제로 훨씬
큰 배치가 안전하게 도는데도 낮은 값을 추천하게 됨). OOM 지점을 찾으면 그 직전 성공한
배치와의 사이를 이진 탐색으로 좁혀서 실제 한계에 더 가깝게 만든 뒤, 안전 마진(20%)을
빼서 --output에 추천 배치를 JSON으로 저장.

trainerscript.run()을 통해 다른 trainer/*.py처럼 같은 프로세스 안에서 호출됨.
"""

import argparse
import json
import time
from typing import Optional

import numpy as np
from ultralytics import YOLO

CANDIDATE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)


def _make_dummy_tiles(count: int, tile: int) -> list:
    return [np.random.randint(0, 255, (tile, tile, 3), dtype=np.uint8) for _ in range(count)]


def _measure_throughput(model, tile: int, imgsz: int, device: str, batch: int, repeats: int = 2) -> float:
    images = _make_dummy_tiles(batch, tile)
    model.predict(images, imgsz=imgsz, device=device, batch=batch, verbose=False)  # 워밍업(첫 호출은 커널 컴파일 등으로 느림)
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict(images, imgsz=imgsz, device=device, batch=batch, verbose=False)
    elapsed = time.perf_counter() - started
    return (batch * repeats) / elapsed if elapsed > 0 else 0.0


def _probe(model, tile: int, imgsz: int, device: str, batch: int, deadline: float, trials: list) -> Optional[float]:
    """batch를 실제로 돌려보고 성공하면 처리량, OOM 등 실패면 None을 반환(trials에도 기록)."""
    if time.perf_counter() >= deadline:
        return None
    print(f"Trying batch={batch}...", flush=True)
    try:
        throughput = _measure_throughput(model, tile, imgsz, device, batch)
    except Exception as exc:  # noqa: BLE001 - CUDA OOM 등: 이 배치는 실패로 기록
        print(f"batch={batch} failed: {exc}", flush=True)
        trials.append({"batch": batch, "tilesPerSec": None, "error": str(exc)})
        return None
    print(f"batch={batch}: {throughput:.1f} tiles/sec", flush=True)
    trials.append({"batch": batch, "tilesPerSec": throughput, "error": None})
    return throughput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--time-budget", type=float, default=45.0)
    args = parser.parse_args()

    model = YOLO(args.model)
    candidates = [b for b in CANDIDATE_BATCHES if b <= args.max_batch]
    deadline = time.perf_counter() + args.time_budget

    trials: list = []
    last_good_batch = 1
    last_good_throughput = 0.0
    failed_batch = None

    # 1단계: 2배씩 늘려가며 실제로 OOM(또는 다른 실패)이 날 때까지 밀어붙임 - 처리량이
    # 정체돼도 멈추지 않음(속도 정체 != 메모리 한계, 실제로 훨씬 큰 배치도 안전하게 도는
    # 경우가 많음).
    for batch in candidates:
        throughput = _probe(model, args.tile, args.imgsz, args.device, batch, deadline, trials)
        if throughput is None:
            if trials and trials[-1]["batch"] == batch and trials[-1]["error"] is not None:
                failed_batch = batch
            break
        last_good_batch, last_good_throughput = batch, throughput

    # 2단계: OOM 지점을 찾았으면 마지막 성공값과의 사이를 이진 탐색으로 좁혀서 실제 한계에
    # 더 가깝게 만듦 (2배씩 뛰는 간격이 커서, 예를 들어 16은 되고 32는 터지면 진짜 한계는
    # 그 사이 어딘가 - 그 값을 못 찾으면 16만 보고 과소 추천하게 됨).
    if failed_batch is not None:
        low, high = last_good_batch, failed_batch
        for _ in range(4):
            if high - low <= 1 or time.perf_counter() >= deadline:
                break
            mid = low + (high - low) // 2
            throughput = _probe(model, args.tile, args.imgsz, args.device, mid, deadline, trials)
            if throughput is None:
                high = mid
            else:
                low, last_good_throughput = mid, throughput
        last_good_batch = low

    # 실제로 돌아간 최대 배치 바로 아래로 안전 마진(20%) - 검출 개수/NMS 등 실제 운영 데이터는
    # 벤치마크용 더미 이미지보다 메모리를 더 쓸 수 있어서, 한계치를 그대로 쓰면 위험함.
    recommended_batch = max(1, int(last_good_batch * 0.8)) if last_good_batch > 1 else last_good_batch

    result = {
        "recommendedBatch": recommended_batch,
        "bestMeasuredBatch": last_good_batch,
        "bestThroughputTilesPerSec": last_good_throughput,
        "trials": trials,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"Recommended batch: {recommended_batch} (max working batch: {last_good_batch} @ "
          f"{last_good_throughput:.1f} tiles/sec)", flush=True)


if __name__ == "__main__":
    main()
