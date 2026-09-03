"""6번 배치 크기 자동 최적화: 이 GPU/모델/타일 크기로 실제 model.predict를 돌려가며
배치를 1,2,4,8...로 늘려서 처리량(tiles/sec)을 재고, 향상폭이 작아지는 지점(포화) 또는
OOM 직전까지 찾음. 마지막으로 안전 마진(20%)을 빼서 --output에 추천 배치를 JSON으로 저장.

trainerscript.run()을 통해 다른 trainer/*.py처럼 같은 프로세스 안에서 호출됨.
"""

import argparse
import json
import time

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

    trials = []
    best_batch = 1
    best_throughput = 0.0
    for batch in candidates:
        if time.perf_counter() >= deadline:
            print(f"Time budget ({args.time_budget}s) reached, stopping search.", flush=True)
            break
        print(f"Trying batch={batch}...", flush=True)
        try:
            throughput = _measure_throughput(model, args.tile, args.imgsz, args.device, batch)
        except Exception as exc:  # noqa: BLE001 - CUDA OOM 등: 이 배치는 실패로 기록하고 탐색 종료
            print(f"batch={batch} failed: {exc}", flush=True)
            trials.append({"batch": batch, "tilesPerSec": None, "error": str(exc)})
            break
        print(f"batch={batch}: {throughput:.1f} tiles/sec", flush=True)
        trials.append({"batch": batch, "tilesPerSec": throughput, "error": None})
        # 이전 최고 대비 20% 미만 향상이면 이미 포화 - 더 큰 배치는 메모리만 더 쓰고 이득 적음.
        if best_throughput > 0 and throughput < best_throughput * 1.2:
            if throughput > best_throughput:
                best_batch, best_throughput = batch, throughput
            break
        if throughput > best_throughput:
            best_batch, best_throughput = batch, throughput

    # OOM/포화 지점 바로 아래로 안전 마진(20%) - 실제 운영 중엔 검출 개수/NMS 등으로 메모리 사용량이
    # 벤치마크보다 더 튈 수 있어서, 측정된 최댓값을 그대로 쓰면 나중에 터질 수 있음.
    recommended_batch = max(1, int(best_batch * 0.8)) if best_batch > 1 else best_batch

    result = {
        "recommendedBatch": recommended_batch,
        "bestMeasuredBatch": best_batch,
        "bestThroughputTilesPerSec": best_throughput,
        "trials": trials,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"Recommended batch: {recommended_batch} (measured best: {best_batch} @ "
          f"{best_throughput:.1f} tiles/sec)", flush=True)


if __name__ == "__main__":
    main()
