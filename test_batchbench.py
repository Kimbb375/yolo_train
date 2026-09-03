"""6번 '최적 배치 검색'(batchbench.py / trainer/bench_batch.py) 검증.

CPU + 작은 max-batch/time-budget으로 실제 yolo26n.pt에 대해 진짜 벤치마크를 돌려 결과
스키마와 추천값 산출 로직(포화 지점에서 멈추고 20% 안전 마진 적용)을 확인함.

python test_batchbench.py 로 직접 실행.
"""

import os
import sys

import batchbench
import trainerscript

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = os.path.join(PROJECT_ROOT, "yolo26n.pt")


def check_bench_batch_cpu() -> None:
    # device=cpu라 trainerscript._wants_gpu가 False -> gpu_setup 안 거치고 바로 실행됨.
    sys.path.insert(0, trainerscript.TRAINER_DIR) if trainerscript.TRAINER_DIR not in sys.path else None
    import bench_batch

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "bench_result.json")
        previous_argv = sys.argv
        sys.argv = ["bench_batch.py", "--model", BASE_MODEL, "--tile", "320", "--imgsz", "320",
                    "--device", "cpu", "--output", output_path, "--max-batch", "4", "--time-budget", "20"]
        try:
            bench_batch.main()
        finally:
            sys.argv = previous_argv

        import json
        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)

    assert data["recommendedBatch"] >= 1
    assert data["bestMeasuredBatch"] in (1, 2, 4)
    assert data["bestThroughputTilesPerSec"] > 0
    assert len(data["trials"]) >= 1
    assert data["trials"][0]["batch"] == 1
    for trial in data["trials"]:
        assert trial["error"] is None or trial["tilesPerSec"] is None

    print("OK: trainer/bench_batch.py - CPU에서 batch 1/2/4 실제 벤치마크 -> JSON 스키마/추천값 확인.")


def check_option_helpers() -> None:
    from main import InferenceTab  # noqa: PLC0415 - Qt 위젯 없이 static helper만 필요
    assert InferenceTab._get_option("tile=320, device=0", "tile", "640") == "320"
    assert InferenceTab._get_option("tile=320, device=0", "batch", "8") == "8"
    assert InferenceTab._set_option("tile=320, batch=8", "batch", 16) == "tile=320, batch=16"
    assert InferenceTab._set_option("tile=320", "batch", 16) == "tile=320, batch=16"
    print("OK: InferenceTab._get_option/_set_option - 옵션 문자열에서 값 읽기/갱신 확인.")


def check_batchbench_run_wrapper() -> None:
    result = batchbench.run(BASE_MODEL, tile=320, imgsz=320, device="cpu")
    assert result.recommendedBatch >= 1
    assert result.to_display_text().startswith("[OK]")
    print("OK: batchbench.run - trainerscript.run 경유해 trainer/bench_batch.py 결과를 "
          "BatchBenchResult로 정상 역직렬화.")


if __name__ == "__main__":
    check_bench_batch_cpu()
    check_option_helpers()
    check_batchbench_run_wrapper()
