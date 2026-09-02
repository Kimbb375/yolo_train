"""4번 YOLO 정렬: 3번 출력 -> train/val/test/predict YOLO 표준 구조.

C# YoloDatasetOrganizer.cs 포팅. 같은 개체에서 나온 5개 zone 크롭(tl/tr/bl/br/cc)은
하나의 그룹으로 묶여서 분할됨 (개체 단위 분할, 데이터 누수 방지 — docs/domain-rules.md §5).
"""

from __future__ import annotations

import glob
import os
import random
import re
import shutil
from dataclasses import dataclass, field

_GROUP_KEY_PATTERN = re.compile(r"^(?P<group>.+_obj\d{6})_\d+_[a-z]{2}$", re.IGNORECASE)


def _reset_directory(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _group_key(base_name: str) -> str:
    match = _GROUP_KEY_PATTERN.match(base_name)
    return match.group("group") if match else base_name


@dataclass
class _SplitDefinition:
    name: str
    ratio: int


@dataclass
class YoloSplitSummary:
    name: str
    groupCount: int
    imageCount: int
    labelCount: int


@dataclass
class YoloDatasetOrganizeResult:
    sourceSizeRootPath: str
    targetSizeRootPath: str
    groupCount: int
    sampleCount: int
    splits: list

    def to_display_text(self) -> str:
        lines = [
            "[OK] YOLO dataset arranged", "",
            f"Source  : {self.sourceSizeRootPath}",
            f"Target  : {self.targetSizeRootPath}",
            f"Groups  : {self.groupCount}",
            f"Samples : {self.sampleCount}", "",
        ]
        for split in self.splits:
            lines.append(f"{split.name:<7} groups={split.groupCount:>5} "
                          f"images={split.imageCount:>6} labels={split.labelCount:>6}")
        lines.append("")
        lines.append("YOLO YAML: dataset.yaml")
        return "\n".join(lines)


def _build_dataset_yaml(target_size_root: str) -> str:
    root = target_size_root.replace("\\", "/")
    return (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: whale\n"
    )


def organize(source_output_root_path: str, target_yolo_root_path: str, image_size: int,
             train_ratio: int, validation_ratio: int, test_ratio: int, predict_ratio: int,
             seed: int) -> YoloDatasetOrganizeResult:
    ratios = (train_ratio, validation_ratio, test_ratio, predict_ratio)
    if image_size <= 0:
        raise ValueError("Image size must be positive.")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios cannot be negative.")
    if sum(ratios) <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")

    source_size_root = os.path.join(source_output_root_path, str(image_size))
    source_images_root = os.path.join(source_size_root, "images")
    source_labels_root = os.path.join(source_size_root, "labels")
    if not os.path.isdir(source_images_root) or not os.path.isdir(source_labels_root):
        raise FileNotFoundError(f"3rd step output was not found: {source_size_root}")

    target_size_root = os.path.join(target_yolo_root_path, str(image_size))
    _reset_directory(target_size_root)

    splits = [
        _SplitDefinition("train", train_ratio),
        _SplitDefinition("val", validation_ratio),
        _SplitDefinition("test", test_ratio),
        _SplitDefinition("predict", predict_ratio),
    ]
    for split in splits:
        os.makedirs(os.path.join(target_size_root, "images", split.name), exist_ok=True)
        os.makedirs(os.path.join(target_size_root, "labels", split.name), exist_ok=True)

    samples = []
    for image_path in sorted(glob.glob(os.path.join(source_images_root, "*.png"))):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_file_name = base_name + ".txt"
        label_path = os.path.join(source_labels_root, label_file_name)
        if not os.path.isfile(label_path):
            continue
        samples.append({
            "imagePath": image_path,
            "labelPath": label_path,
            "baseName": base_name,
            "fileName": os.path.basename(image_path),
            "labelFileName": label_file_name,
            "groupKey": _group_key(base_name),
        })
    if not samples:
        raise ValueError(f"No paired image/label samples were found in {source_size_root}.")

    groups_by_key: dict[str, list[dict]] = {}
    for sample in samples:
        groups_by_key.setdefault(sample["groupKey"].lower(), []).append(sample)
    groups = [
        (key, sorted(items, key=lambda s: s["baseName"].lower()))
        for key, items in sorted(groups_by_key.items())
    ]

    rng = random.Random(seed)
    rng.shuffle(groups)

    total_ratio = sum(split.ratio for split in splits)
    accumulated = {split.name: 0 for split in splits}
    targets = {split.name: len(groups) * split.ratio / total_ratio for split in splits}

    counts = {split.name: {"groupCount": 0, "imageCount": 0, "labelCount": 0} for split in splits}

    for _key, group_samples in groups:
        candidates = [split for split in splits if split.ratio > 0]
        chosen = sorted(candidates, key=lambda s: (-(targets[s.name] - accumulated[s.name]), -s.ratio))[0]

        for sample in group_samples:
            target_image = os.path.join(target_size_root, "images", chosen.name, sample["fileName"])
            target_label = os.path.join(target_size_root, "labels", chosen.name, sample["labelFileName"])
            shutil.copy2(sample["imagePath"], target_image)
            shutil.copy2(sample["labelPath"], target_label)
            counts[chosen.name]["imageCount"] += 1
            counts[chosen.name]["labelCount"] += 1

        counts[chosen.name]["groupCount"] += 1
        accumulated[chosen.name] += 1

    with open(os.path.join(target_size_root, "classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("whale\n")
    with open(os.path.join(target_size_root, "predefined_classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("whale\n")
    with open(os.path.join(target_size_root, "dataset.yaml"), "w", encoding="utf-8") as fh:
        fh.write(_build_dataset_yaml(target_size_root))

    split_summaries = [
        YoloSplitSummary(split.name, counts[split.name]["groupCount"],
                          counts[split.name]["imageCount"], counts[split.name]["labelCount"])
        for split in splits
    ]
    return YoloDatasetOrganizeResult(source_size_root, target_size_root, len(groups), len(samples), split_summaries)
