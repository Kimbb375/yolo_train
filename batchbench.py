"""6번 배치 크기 자동 최적화 — trainer/bench_batch.py를 이 프로세스 안에서 실행해 이
GPU/모델/타일 크기에서 안전한 배치 값을 찾아옴 (컴퓨터 사양마다 batch 최적값이 달라서,
사용자가 직접 시행착오로 맞추는 대신 실제로 측정해서 정해줌)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

import trainerscript


@dataclass
class BatchBenchResult:
    recommendedBatch: int
    bestMeasuredBatch: int
    bestThroughputTilesPerSec: float
    trials: list

    def to_display_text(self) -> str:
        lines = [
            f"[OK] 배치 벤치마크 완료 — 추천 batch={self.recommendedBatch}",
            f"(측정 최고 batch={self.bestMeasuredBatch} @ {self.bestThroughputTilesPerSec:.1f} tiles/sec, "
            f"안전 마진 20% 적용됨)",
        ]
        for trial in self.trials:
            if trial.get("error"):
                lines.append(f"  batch={trial['batch']}: 실패 ({trial['error']})")
            else:
                lines.append(f"  batch={trial['batch']}: {trial['tilesPerSec']:.1f} tiles/sec")
        return "\n".join(lines)


def run(model_path: str, tile: int, imgsz: int, device: str) -> BatchBenchResult:
    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "bench_result.json")
        trainerscript.run("bench_batch", [
            "--model", model_path, "--tile", str(tile), "--imgsz", str(imgsz),
            "--device", device, "--output", output_path])
        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)
    return BatchBenchResult(**data)
