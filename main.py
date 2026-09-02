import io
import os
import re
import sys
from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import json

import compare
import inference
import objectdb
import review
import sourceimage
import sourceverify
import trainingdataset
import training
import appversion
import updatecheck
import yolodataset

AUGMENTATION_PRESETS = {
    "None": "degrees=0, translate=0, scale=0, fliplr=0, flipud=0, hsv_h=0, hsv_s=0, hsv_v=0, "
            "mosaic=0, mixup=0, copy_paste=0, patience=30",
    "Light": "degrees=5, translate=0.03, scale=0.15, fliplr=0.5, flipud=0.5, hsv_h=0, hsv_s=0.1, "
             "hsv_v=0.1, mosaic=0.1, mixup=0, copy_paste=0, patience=30",
    "Default": "degrees=10, translate=0.05, scale=0.25, fliplr=0.5, flipud=0.5, hsv_h=0, hsv_s=0.2, "
               "hsv_v=0.2, mosaic=0.2, mixup=0, copy_paste=0, patience=30",
    "Strong": "degrees=20, translate=0.1, scale=0.4, fliplr=0.5, flipud=0.5, hsv_h=0.01, hsv_s=0.35, "
              "hsv_v=0.35, mosaic=0.5, mixup=0.05, copy_paste=0, patience=40",
}


class LabelDbTab(QWidget):
    """1. 라벨 DB — YOLO 라벨+타일 이미지 -> object_db.json"""

    def __init__(self) -> None:
        super().__init__()

        self.source_input = QLineEdit()
        self.output_input = QLineEdit()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        form = QGridLayout()
        form.addWidget(QLabel("YOLO 라벨 루트 폴더"), 0, 0)
        form.addWidget(self.source_input, 0, 1)
        form.addWidget(self._browse_button(self._pick_source), 0, 2)

        form.addWidget(QLabel("저장할 JSON 경로"), 1, 0)
        form.addWidget(self.output_input, 1, 1)
        form.addWidget(self._browse_button(self._pick_output), 1, 2)

        build_row = QHBoxLayout()
        build_button = QPushButton("DB 생성")
        build_button.clicked.connect(self._on_build_clicked)
        build_row.addWidget(build_button)
        build_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(build_row)
        layout.addWidget(self.log, stretch=1)

    @staticmethod
    def _browse_button(handler) -> QPushButton:
        button = QPushButton("찾기...")
        button.clicked.connect(handler)
        return button

    def _pick_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "YOLO 라벨 루트 폴더 선택")
        if path:
            self.source_input.setText(path)

    def _pick_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "저장할 JSON 경로", "object_db.json", "JSON (*.json)")
        if path:
            self.output_input.setText(path)

    def _on_build_clicked(self) -> None:
        source = self.source_input.text().strip()
        output = self.output_input.text().strip() or None
        if not source:
            self.log.setPlainText("[오류] YOLO 라벨 루트 폴더를 지정하세요.")
            return

        try:
            result = objectdb.build(source, output)
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.log.setPlainText(f"[오류] {exc}")
            return

        self.log.setPlainText(result.to_display_text())
        self.output_input.setText(result.outputPath)


class TrainingTileTab(QWidget):
    """3. 학습 타일 — object_db.json + 원본 TIF -> 512/640 학습 크롭 + YOLO 라벨"""

    def __init__(self) -> None:
        super().__init__()

        self.object_db_input = QLineEdit()
        self.source_root_input = QLineEdit()
        self.output_root_input = QLineEdit()
        self.size_512_checkbox = QCheckBox("512")
        self.size_640_checkbox = QCheckBox("640")
        self.size_640_checkbox.setChecked(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        form = QGridLayout()
        form.addWidget(QLabel("object JSON"), 0, 0)
        form.addWidget(self.object_db_input, 0, 1)
        form.addWidget(self._browse_button(self._pick_object_db), 0, 2)

        form.addWidget(QLabel("원본 TIF 루트 폴더"), 1, 0)
        form.addWidget(self.source_root_input, 1, 1)
        form.addWidget(self._browse_button(self._pick_source_root), 1, 2)

        form.addWidget(QLabel("출력 폴더"), 2, 0)
        form.addWidget(self.output_root_input, 2, 1)
        form.addWidget(self._browse_button(self._pick_output_root), 2, 2)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("출력 크기"))
        size_row.addWidget(self.size_512_checkbox)
        size_row.addWidget(self.size_640_checkbox)
        size_row.addStretch(1)

        build_row = QHBoxLayout()
        build_button = QPushButton("학습 타일 생성")
        build_button.clicked.connect(self._on_build_clicked)
        build_row.addWidget(build_button)
        build_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(size_row)
        layout.addLayout(build_row)
        layout.addWidget(self.log, stretch=1)

    @staticmethod
    def _browse_button(handler) -> QPushButton:
        button = QPushButton("찾기...")
        button.clicked.connect(handler)
        return button

    def _pick_object_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "object JSON 선택", "", "JSON (*.json)")
        if path:
            self.object_db_input.setText(path)

    def _pick_source_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "원본 TIF 루트 폴더 선택")
        if path:
            self.source_root_input.setText(path)

    def _pick_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "출력 폴더 선택")
        if path:
            self.output_root_input.setText(path)

    def _on_build_clicked(self) -> None:
        object_db_path = self.object_db_input.text().strip()
        source_root = self.source_root_input.text().strip()
        output_root = self.output_root_input.text().strip()
        sizes = [size for size, checked in ((512, self.size_512_checkbox.isChecked()),
                                             (640, self.size_640_checkbox.isChecked())) if checked]

        if not object_db_path or not source_root or not output_root:
            self.log.setPlainText("[오류] object JSON, 원본 TIF 루트, 출력 폴더를 모두 지정하세요.")
            return

        try:
            result = trainingdataset.build(object_db_path, source_root, output_root, sizes or None)
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.log.setPlainText(f"[오류] {exc}")
            return

        self.log.setPlainText(result.to_display_text())


class YoloOrganizeTab(QWidget):
    """4. YOLO 정렬 — 3번 출력 -> train/val/test/predict YOLO 표준 구조"""

    def __init__(self) -> None:
        super().__init__()

        self.source_root_input = QLineEdit()
        self.target_root_input = QLineEdit()
        self.image_size_input = QSpinBox()
        self.image_size_input.setRange(32, 4096)
        self.image_size_input.setValue(640)
        self.train_ratio_input = self._ratio_spinbox(70)
        self.val_ratio_input = self._ratio_spinbox(15)
        self.test_ratio_input = self._ratio_spinbox(15)
        self.predict_ratio_input = self._ratio_spinbox(0)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(1234)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        form = QGridLayout()
        form.addWidget(QLabel("3번 출력 루트"), 0, 0)
        form.addWidget(self.source_root_input, 0, 1)
        form.addWidget(self._browse_button(self._pick_source_root), 0, 2)

        form.addWidget(QLabel("YOLO 출력 루트"), 1, 0)
        form.addWidget(self.target_root_input, 1, 1)
        form.addWidget(self._browse_button(self._pick_target_root), 1, 2)

        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("크기"))
        ratio_row.addWidget(self.image_size_input)
        ratio_row.addWidget(QLabel("train"))
        ratio_row.addWidget(self.train_ratio_input)
        ratio_row.addWidget(QLabel("val"))
        ratio_row.addWidget(self.val_ratio_input)
        ratio_row.addWidget(QLabel("test"))
        ratio_row.addWidget(self.test_ratio_input)
        ratio_row.addWidget(QLabel("predict"))
        ratio_row.addWidget(self.predict_ratio_input)
        ratio_row.addWidget(QLabel("seed"))
        ratio_row.addWidget(self.seed_input)
        ratio_row.addStretch(1)

        build_row = QHBoxLayout()
        build_button = QPushButton("YOLO 정렬 실행")
        build_button.clicked.connect(self._on_build_clicked)
        build_row.addWidget(build_button)
        build_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(ratio_row)
        layout.addLayout(build_row)
        layout.addWidget(self.log, stretch=1)

    @staticmethod
    def _ratio_spinbox(default: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(0, 1000)
        spinbox.setValue(default)
        return spinbox

    @staticmethod
    def _browse_button(handler) -> QPushButton:
        button = QPushButton("찾기...")
        button.clicked.connect(handler)
        return button

    def _pick_source_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "3번 출력 루트 선택")
        if path:
            self.source_root_input.setText(path)

    def _pick_target_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "YOLO 출력 루트 선택")
        if path:
            self.target_root_input.setText(path)

    def _on_build_clicked(self) -> None:
        source_root = self.source_root_input.text().strip()
        target_root = self.target_root_input.text().strip()
        if not source_root or not target_root:
            self.log.setPlainText("[오류] 3번 출력 루트와 YOLO 출력 루트를 모두 지정하세요.")
            return

        try:
            result = yolodataset.organize(
                source_root, target_root, self.image_size_input.value(),
                self.train_ratio_input.value(), self.val_ratio_input.value(),
                self.test_ratio_input.value(), self.predict_ratio_input.value(),
                self.seed_input.value())
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.log.setPlainText(f"[오류] {exc}")
            return

        self.log.setPlainText(result.to_display_text())


class _StreamToSignal(io.TextIOBase):
    """print()/logger 출력을 Qt 시그널로 넘겨서 다른 스레드에서 로그 창에 표시함."""

    def __init__(self, signal: Signal) -> None:
        super().__init__()
        self._signal = signal

    def write(self, text: str) -> int:
        if text:
            self._signal.emit(text)
        return len(text)

    def flush(self) -> None:
        pass


class BackgroundCallWorker(QThread):
    """stdout을 Qt 시그널로 리다이렉션하며 함수 하나를 백그라운드 스레드에서 실행 (5/6번 공용)."""

    output = Signal(str)
    finished_ok = Signal(object)
    finished_error = Signal(str)

    def __init__(self, func, *args) -> None:
        super().__init__()
        self._func = func
        self._args = args

    def run(self) -> None:
        stream = _StreamToSignal(self.output)
        previous_stdout, previous_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stream, stream
        try:
            result = self._func(*self._args)
        except Exception as exc:  # noqa: BLE001 - 백그라운드 스레드, 예외 텍스트 그대로 UI에 전달
            self.finished_error.emit(str(exc))
            return
        finally:
            sys.stdout, sys.stderr = previous_stdout, previous_stderr
        self.finished_ok.emit(result)


class TrainingTab(QWidget):
    """5. 학습 — trainer/train.py를 서브프로세스 없이 같은 프로세스 안에서 실행."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BackgroundCallWorker | None = None

        self.dataset_input = QPlainTextEdit()
        self.dataset_input.setPlaceholderText("YOLO 데이터셋 폴더 (여러 개면 줄바꿈으로 구분)")
        self.dataset_input.setFixedHeight(60)
        self.model_input = QLineEdit()
        self.project_input = QLineEdit()
        self.name_input = QLineEdit("yolo_whale")
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(32, 4096)
        self.imgsz_input.setValue(640)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100000)
        self.epochs_input.setValue(100)
        self.batch_input = QLineEdit("auto")
        self.device_input = QLineEdit("auto")
        self.workers_input = QSpinBox()
        self.workers_input.setRange(0, 64)
        self.augmentation_preset = QComboBox()
        self.augmentation_preset.addItems(list(AUGMENTATION_PRESETS.keys()))
        self.augmentation_preset.setCurrentText("Default")
        self.augmentation_input = QLineEdit(AUGMENTATION_PRESETS["Default"])
        self.augmentation_preset.currentTextChanged.connect(
            lambda name: self.augmentation_input.setText(AUGMENTATION_PRESETS[name]))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        form = QGridLayout()
        form.addWidget(QLabel("YOLO 데이터셋 폴더"), 0, 0)
        form.addWidget(self.dataset_input, 0, 1)
        form.addWidget(self._browse_button(self._pick_dataset), 0, 2)

        form.addWidget(QLabel("초기 모델 pt"), 1, 0)
        form.addWidget(self.model_input, 1, 1)
        form.addWidget(self._browse_button(self._pick_model), 1, 2)

        form.addWidget(QLabel("runs 출력 폴더"), 2, 0)
        form.addWidget(self.project_input, 2, 1)
        form.addWidget(self._browse_button(self._pick_project), 2, 2)

        param_row = QHBoxLayout()
        for label, widget in (("이름", self.name_input), ("imgsz", self.imgsz_input),
                               ("epochs", self.epochs_input), ("batch", self.batch_input),
                               ("device", self.device_input), ("workers", self.workers_input)):
            param_row.addWidget(QLabel(label))
            param_row.addWidget(widget)

        aug_row = QHBoxLayout()
        aug_row.addWidget(QLabel("증강"))
        aug_row.addWidget(self.augmentation_preset)
        aug_row.addWidget(self.augmentation_input, stretch=1)

        self.start_button = QPushButton("학습 시작")
        self.start_button.clicked.connect(self._on_start_clicked)
        build_row = QHBoxLayout()
        build_row.addWidget(self.start_button)
        build_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(param_row)
        layout.addLayout(aug_row)
        layout.addLayout(build_row)
        layout.addWidget(self.log, stretch=1)

    @staticmethod
    def _browse_button(handler) -> QPushButton:
        button = QPushButton("찾기...")
        button.clicked.connect(handler)
        return button

    def _pick_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "YOLO 데이터셋 폴더 선택")
        if path:
            existing = self.dataset_input.toPlainText().strip()
            self.dataset_input.setPlainText((existing + "\n" + path).strip() if existing else path)

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "초기 모델 pt 선택", "", "PyTorch model (*.pt)")
        if path:
            self.model_input.setText(path)

    def _pick_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "runs 출력 폴더 선택")
        if path:
            self.project_input.setText(path)

    def _on_start_clicked(self) -> None:
        dataset_roots = [line.strip() for line in self.dataset_input.toPlainText().splitlines() if line.strip()]
        model_path = self.model_input.text().strip()
        project = self.project_input.text().strip()
        if not dataset_roots or not model_path or not project:
            self.log.setPlainText("[오류] 데이터셋 폴더, 초기 모델, runs 출력 폴더를 모두 지정하세요.")
            return

        try:
            args = training.build_train_args(
                dataset_roots, model_path, self.imgsz_input.value(), self.epochs_input.value(),
                self.batch_input.text().strip() or "auto", self.device_input.text().strip() or "auto",
                project, self.name_input.text().strip() or "yolo_whale", self.workers_input.value(),
                self.augmentation_input.text())
        except ValueError as exc:
            self.log.setPlainText(f"[오류] {exc}")
            return

        self.log.setPlainText("학습 시작...\n")
        self.start_button.setEnabled(False)
        self._worker = BackgroundCallWorker(training.run, args)
        self._worker.output.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    def _on_finished_ok(self, _result) -> None:
        self._append_log("\n[OK] 학습 종료.\n")
        self.start_button.setEnabled(True)

    def _on_finished_error(self, message: str) -> None:
        self._append_log(f"\n[오류] {message}\n")
        self.start_button.setEnabled(True)


class InferenceTab(QWidget):
    """6. 원본 추론 — 원본 TIF -> 내부 타일링 -> YOLO 추론 -> candidates.json"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BackgroundCallWorker | None = None
        self._selected_files: list[str] = []

        self.source_input = QLineEdit()
        self.source_input.textEdited.connect(self._on_source_edited_by_user)
        self.model_input = QLineEdit()
        self.output_input = QLineEdit()
        self.name_input = QLineEdit()
        self.options_input = QLineEdit(
            "tile_mode=memory, resume=1, tile=640, overlap=0.2, conf=0.1, iou=0.6, imgsz=640, batch=auto, "
            "device=0, max_det=300, merge_iou=0.5, candidate_crop=640, candidate_context=120, candidate_view=tile")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        source_buttons = QHBoxLayout()
        source_buttons.addWidget(self._browse_button(self._pick_source, "폴더 선택..."))
        source_buttons.addWidget(self._browse_button(self._pick_source_files, "파일 선택(다중)..."))

        form = QGridLayout()
        form.addWidget(QLabel("원본 TIF 폴더/파일"), 0, 0)
        form.addWidget(self.source_input, 0, 1)
        form.addLayout(source_buttons, 0, 2)

        form.addWidget(QLabel("모델 pt"), 1, 0)
        form.addWidget(self.model_input, 1, 1)
        form.addWidget(self._browse_button(self._pick_model), 1, 2)

        form.addWidget(QLabel("출력 폴더"), 2, 0)
        form.addWidget(self.output_input, 2, 1)
        form.addWidget(self._browse_button(self._pick_output), 2, 2)

        form.addWidget(QLabel("실행 이름"), 3, 0)
        form.addWidget(self.name_input, 3, 1)

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("옵션"))
        options_row.addWidget(self.options_input, stretch=1)

        self.start_button = QPushButton("추론 시작")
        self.start_button.clicked.connect(self._on_start_clicked)
        build_row = QHBoxLayout()
        build_row.addWidget(self.start_button)
        build_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(options_row)
        layout.addLayout(build_row)
        layout.addWidget(self.log, stretch=1)

    @staticmethod
    def _browse_button(handler, label: str = "찾기...") -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(handler)
        return button

    def _pick_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "원본 TIF 폴더 선택")
        if path:
            self._selected_files = []
            self.source_input.setText(path)

    def _pick_source_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "원본 TIF 파일 선택 (여러 개 가능)", "", "TIF images (*.tif *.tiff)")
        if not paths:
            return
        self._selected_files = paths
        self.source_input.setText(self._summarize_selected_files(paths))

    def _on_source_edited_by_user(self, _text: str) -> None:
        # 파일 다중선택 후 요약 텍스트가 표시된 상태에서 사용자가 직접 입력창을 고치면
        # 선택된 파일 목록은 더 이상 유효하지 않음 -> 입력창 텍스트를 그대로 씀.
        self._selected_files = []

    @staticmethod
    def _summarize_selected_files(paths: list[str]) -> str:
        names = sorted(os.path.basename(p) for p in paths)
        first_digits = re.match(r"\d+", names[0])
        label = first_digits.group(0) if first_digits else names[0]
        return f"{label}번부터 총 {len(names)}개 tif 선택됨"

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "모델 pt 선택", "", "PyTorch model (*.pt)")
        if path:
            self.model_input.setText(path)

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "출력 폴더 선택")
        if path:
            self.output_input.setText(path)

    def _on_start_clicked(self) -> None:
        source = ";".join(self._selected_files) if self._selected_files else self.source_input.text().strip()
        model_path = self.model_input.text().strip()
        output_root = self.output_input.text().strip()
        if not source or not model_path or not output_root:
            self.log.setPlainText("[오류] 원본 TIF, 모델, 출력 폴더를 모두 지정하세요.")
            return

        self.log.setPlainText("추론 시작...\n")
        self.start_button.setEnabled(False)
        self._worker = BackgroundCallWorker(
            inference.run, source, output_root, model_path,
            self.name_input.text().strip() or None, self.options_input.text())
        self._worker.output.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    def _on_finished_ok(self, result) -> None:
        self._append_log("\n" + result.to_display_text() + "\n")
        self.start_button.setEnabled(True)

    def _on_finished_error(self, message: str) -> None:
        self._append_log(f"\n[오류] {message}\n")
        self.start_button.setEnabled(True)


class CandidateImageLabel(QLabel):
    """후보 크롭 이미지 표시 영역. 클릭해서 포커스를 줘야 A/D/Space/X 키가 먹음(§알아둘 것)."""

    def __init__(self) -> None:
        super().__init__("후보를 불러오세요.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 480)
        self.setStyleSheet("background-color: #222; color: #ccc;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.setFocus()
        super().mousePressEvent(event)


class ReviewTab(QWidget):
    """7. 후보 검수 — candidates.json을 A/D로 넘기며 Space(고래 확정)/X(고래 아님)로 분류."""

    def __init__(self) -> None:
        super().__init__()
        self._candidates: list[review.ReviewCandidate] = []
        self._index = 0

        self.candidate_json_input = QLineEdit()
        self.output_root_input = QLineEdit()
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("예: conf>=0.3, width>=20")
        self.filter_input.textChanged.connect(self._update_filter_status_label)
        self.filter_status_label = QLabel()
        self.filter_status_label.setStyleSheet("color: #666;")
        self.jump_input = QSpinBox()
        self.jump_input.setRange(1, 1000)
        self.jump_input.setValue(1)
        self.image_label = CandidateImageLabel()
        self.info_label = QLabel("-")
        self.status_label = QLabel("")

        form = QGridLayout()
        form.addWidget(QLabel("candidates.json"), 0, 0)
        form.addWidget(self.candidate_json_input, 0, 1)
        form.addWidget(self._browse_button(self._pick_candidate_json), 0, 2)

        form.addWidget(QLabel("검수 출력 폴더"), 1, 0)
        form.addWidget(self.output_root_input, 1, 1)
        form.addWidget(self._browse_button(self._pick_output_root), 1, 2)

        load_row = QHBoxLayout()
        load_button = QPushButton("불러오기")
        load_button.clicked.connect(self._on_load_clicked)
        load_row.addWidget(load_button)
        load_row.addWidget(QLabel("필터"))
        load_row.addWidget(self.filter_input, stretch=1)
        apply_filter_button = QPushButton("필터 적용")
        apply_filter_button.clicked.connect(self._on_apply_filter_clicked)
        load_row.addWidget(apply_filter_button)
        load_row.addWidget(QLabel("점프"))
        load_row.addWidget(self.jump_input)

        nav_row = QHBoxLayout()
        prev_button = QPushButton("<- 이전 (A)")
        prev_button.clicked.connect(self._go_prev)
        next_button = QPushButton("다음 (D) ->")
        next_button.clicked.connect(self._go_next)
        self.confirm_button = QPushButton("고래 확정 (Space)")
        self.confirm_button.clicked.connect(self._toggle_confirmed)
        self.negative_button = QPushButton("고래 아님 (X)")
        self.negative_button.clicked.connect(self._toggle_negative)
        export_button = QPushButton("통합 JSON 내보내기 (object_db_new.json)")
        export_button.clicked.connect(self._on_export_clicked)
        nav_row.addWidget(prev_button)
        nav_row.addWidget(next_button)
        nav_row.addWidget(self.confirm_button)
        nav_row.addWidget(self.negative_button)
        nav_row.addStretch(1)
        nav_row.addWidget(export_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(load_row)
        layout.addWidget(self.filter_status_label)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.info_label)
        layout.addLayout(nav_row)
        layout.addWidget(self.status_label)

        for key, handler in (("A", self._go_prev), ("D", self._go_next),
                              ("Space", self._toggle_confirmed), ("X", self._toggle_negative)):
            shortcut = QShortcut(QKeySequence(key), self.image_label)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(handler)

        self._update_filter_status_label()

    def _update_filter_status_label(self) -> None:
        # conf/area/width/height는 강제되는 기본값이 없음 - 필터 비우면 전체 표시가 기본값.
        # "지금 몇 이상만 보고 있는지"를 항상 보이게 표기.
        text = self.filter_input.text().strip()
        if not text:
            self.filter_status_label.setText("필터 기준 — 기본값: 없음 (전체 표시) / 현재 설정: 기본값과 동일")
            return
        try:
            review.parse_filters(text)
        except ValueError:
            self.filter_status_label.setText(f"필터 기준 — 기본값: 없음 (전체 표시) / 현재 설정: {text} [형식 오류]")
            return
        self.filter_status_label.setText(f"필터 기준 — 기본값: 없음 (전체 표시) / 현재 설정: {text}")

    @staticmethod
    def _browse_button(handler) -> QPushButton:
        button = QPushButton("찾기...")
        button.clicked.connect(handler)
        return button

    def _pick_candidate_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "candidates.json 선택", "", "JSON (*.json)")
        if path:
            self.candidate_json_input.setText(path)

    def _pick_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "검수 출력 폴더 선택")
        if path:
            self.output_root_input.setText(path)

    def _on_load_clicked(self) -> None:
        candidate_json_path = self.candidate_json_input.text().strip()
        if not candidate_json_path:
            self.status_label.setText("[오류] candidates.json 경로를 지정하세요.")
            return
        try:
            all_candidates = review.load_candidates(candidate_json_path)
            self._candidates = review.apply_filters(all_candidates, self.filter_input.text())
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.status_label.setText(f"[오류] {exc}")
            return
        self._index = 0
        self.status_label.setText(f"{len(all_candidates)}개 중 {len(self._candidates)}개 표시 (필터 적용됨)")
        self.image_label.setFocus()
        self._refresh()

    def _on_apply_filter_clicked(self) -> None:
        if not self.candidate_json_input.text().strip():
            return
        self._on_load_clicked()

    def _output_root(self) -> str:
        return self.output_root_input.text().strip()

    def _go_prev(self) -> None:
        if not self._candidates:
            return
        self._index = max(0, self._index - self.jump_input.value())
        self._refresh()

    def _go_next(self) -> None:
        if not self._candidates:
            return
        self._index = min(len(self._candidates) - 1, self._index + self.jump_input.value())
        self._refresh()

    def _current(self) -> Optional[review.ReviewCandidate]:
        if not self._candidates:
            return None
        return self._candidates[self._index]

    def _toggle_confirmed(self) -> None:
        candidate = self._current()
        if candidate is None or not self._output_root():
            return
        if review.is_confirmed(candidate, self._output_root()):
            review.delete_confirmed(candidate, self._output_root())
        else:
            review.save_confirmed(candidate, self._output_root())
        self._refresh()

    def _toggle_negative(self) -> None:
        candidate = self._current()
        if candidate is None or not self._output_root():
            return
        if review.is_negative(candidate, self._output_root()):
            review.delete_negative(candidate, self._output_root())
        else:
            review.save_negative(candidate, self._output_root())
        self._refresh()

    def _refresh(self) -> None:
        candidate = self._current()
        if candidate is None:
            self.image_label.setText("표시할 후보가 없습니다.")
            self.info_label.setText("-")
            return

        pixmap = QPixmap(candidate.candidateImagePath)
        if pixmap.isNull():
            self.image_label.setText(f"이미지를 불러올 수 없음: {candidate.candidateImagePath}")
        else:
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

        output_root = self._output_root()
        status = []
        if output_root and review.is_confirmed(candidate, output_root):
            status.append("CONFIRMED")
        if output_root and review.is_negative(candidate, output_root):
            status.append("NEGATIVE")
        status_text = "/".join(status) if status else "미분류"

        box = candidate.globalBox
        self.info_label.setText(
            f"[{self._index + 1}/{len(self._candidates)}] cand{candidate.candidateId:06d} "
            f"{candidate.sourceTifName} conf={candidate.confidence:.3f} "
            f"box=({box.left},{box.top})-({box.right},{box.bottom}) [{status_text}]")

        self.confirm_button.setText("고래 확정 취소 (Space)" if status_text.startswith("CONFIRMED") else "고래 확정 (Space)")
        self.negative_button.setText("고래 아님 취소 (X)" if "NEGATIVE" in status else "고래 아님 (X)")

    def _on_export_clicked(self) -> None:
        if not self._output_root():
            self.status_label.setText("[오류] 검수 출력 폴더를 지정하세요.")
            return
        try:
            path = review.export_confirmed_object_db(self._output_root())
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.status_label.setText(f"[오류] {exc}")
            return
        self.status_label.setText(f"통합 JSON 저장됨: {path}")


_COMPARE_STATUS_COLORS = {
    "MATCH": "#e2f5e6", "MISSED": "#ffeed6", "NEW": "#deeeff",
}
_COMPARE_COLUMNS = ("", "상태", "개체", "날짜", "영상", "IoU", "원본 좌표")


class CompareTab(QWidget):
    """8. 매칭/선별 — 왼쪽 기준 데이터 vs 오른쪽 검수 결과를 비교해 NEW/MATCH 표시,
    체크한 개체만 새 object_db.json으로 export."""

    def __init__(self) -> None:
        super().__init__()
        self._left_document: Optional[dict] = None
        self._right_document: Optional[dict] = None

        self.left_input = QLineEdit()
        self.right_input = QLineEdit()
        self.output_input = QLineEdit()
        self.iou_input = QDoubleSpinBox()
        self.iou_input.setRange(0.0, 1.0)
        self.iou_input.setSingleStep(0.05)
        self.iou_input.setValue(0.5)
        self.status_label = QLabel("")

        form = QGridLayout()
        form.addWidget(QLabel("왼쪽 기준 데이터 (object JSON 또는 폴더)"), 0, 0)
        form.addWidget(self.left_input, 0, 1)
        form.addWidget(self._browse_file_button(self.left_input, "object JSON 선택"), 0, 2)
        form.addWidget(self._browse_folder_button(self.left_input, "학습 이미지/검수 폴더 선택"), 0, 3)

        form.addWidget(QLabel("오른쪽 검수 결과 (JSON 또는 폴더)"), 1, 0)
        form.addWidget(self.right_input, 1, 1)
        form.addWidget(self._browse_file_button(self.right_input, "object JSON 선택"), 1, 2)
        form.addWidget(self._browse_folder_button(self.right_input, "검수 결과 폴더 선택"), 1, 3)

        form.addWidget(QLabel("export 출력 경로"), 2, 0)
        form.addWidget(self.output_input, 2, 1)
        form.addWidget(self._browse_folder_button(self.output_input, "export 폴더 선택"), 2, 2)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("IoU 기준"))
        top_row.addWidget(self.iou_input)
        load_button = QPushButton("뷰어 로드")
        load_button.clicked.connect(self._on_load_clicked)
        top_row.addWidget(load_button)
        top_row.addStretch(1)

        select_row = QHBoxLayout()
        for label, handler in (("전체", lambda: self._set_checks(lambda status: True)),
                                ("해제", lambda: self._set_checks(lambda status: False)),
                                ("신규", lambda: self._set_checks(lambda status: status == "NEW")),
                                ("일치", lambda: self._set_checks(lambda status: status == "MATCH"))):
            button = QPushButton(label)
            button.clicked.connect(handler)
            select_row.addWidget(button)
        export_button = QPushButton("체크 내보내기")
        export_button.clicked.connect(self._on_export_clicked)
        select_row.addWidget(export_button)
        select_row.addStretch(1)

        self.left_table = self._make_table()
        self.right_table = self._make_table()
        tables_row = QHBoxLayout()
        tables_row.addWidget(self.left_table)
        tables_row.addWidget(self.right_table)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(top_row)
        layout.addLayout(select_row)
        layout.addLayout(tables_row, stretch=1)
        layout.addWidget(self.status_label)

    @staticmethod
    def _make_table() -> QTableWidget:
        table = QTableWidget(0, len(_COMPARE_COLUMNS))
        table.setHorizontalHeaderLabels(_COMPARE_COLUMNS)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        return table

    @staticmethod
    def _browse_file_button(target: QLineEdit, caption: str) -> QPushButton:
        button = QPushButton("파일...")

        def handler() -> None:
            path, _ = QFileDialog.getOpenFileName(None, caption, "", "JSON (*.json)")
            if path:
                target.setText(path)
        button.clicked.connect(handler)
        return button

    @staticmethod
    def _browse_folder_button(target: QLineEdit, caption: str) -> QPushButton:
        button = QPushButton("폴더...")

        def handler() -> None:
            path = QFileDialog.getExistingDirectory(None, caption)
            if path:
                target.setText(path)
        button.clicked.connect(handler)
        return button

    def _populate_table(self, table: QTableWidget, rows: list[dict]) -> None:
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            checkbox_item.setCheckState(Qt.CheckState.Unchecked)
            checkbox_item.setData(Qt.ItemDataRole.UserRole, row)
            table.setItem(row_index, 0, checkbox_item)

            values = (row["status"], str(row["objectId"]), row["date"], row["image"],
                      "" if row["iou"] is None else f"{row['iou']:.2f}", row["box"])
            for column_offset, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                item.setBackground(_hex_color(_COMPARE_STATUS_COLORS.get(row["status"], "#ffffff")))
                table.setItem(row_index, column_offset, item)
        table.resizeColumnsToContents()

    def _on_load_clicked(self) -> None:
        left_path = self.left_input.text().strip()
        right_path = self.right_input.text().strip()
        if not left_path or not right_path:
            self.status_label.setText("[오류] 왼쪽/오른쪽 데이터 경로를 모두 지정하세요.")
            return

        try:
            self._left_document = compare.load_compare_document(left_path)
            self._right_document = compare.load_compare_document(right_path)
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.status_label.setText(f"[오류] {exc}")
            return

        threshold = self.iou_input.value()
        left_objects = self._left_document["objects"]
        right_objects = self._right_document["objects"]
        left_rows = compare.build_compare_rows(left_objects, right_objects, True, threshold)
        right_rows = compare.build_compare_rows(right_objects, left_objects, False, threshold)
        self._populate_table(self.left_table, left_rows)
        self._populate_table(self.right_table, right_rows)

        matched = sum(1 for row in left_rows if row["status"] == "MATCH")
        self.status_label.setText(
            f"Base objects : {len(left_objects)}  |  Review/new : {len(right_objects)}  |  "
            f"Matched : {matched}  |  Missed : {len(left_objects) - matched}  |  "
            f"New : {sum(1 for row in right_rows if row['status'] == 'NEW')}  |  IoU threshold : {threshold:.2f}")

    def _set_checks(self, predicate) -> None:
        for row_index in range(self.right_table.rowCount()):
            item = self.right_table.item(row_index, 0)
            row = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if predicate(row["status"]) else Qt.CheckState.Unchecked)
        self.status_label.setText(f"Checked review rows: {len(self._get_checked_records())}")

    def _get_checked_records(self) -> list[dict]:
        records = []
        for row_index in range(self.right_table.rowCount()):
            item = self.right_table.item(row_index, 0)
            if item.checkState() == Qt.CheckState.Checked:
                records.append(item.data(Qt.ItemDataRole.UserRole)["record"])
        return records

    def _on_export_clicked(self) -> None:
        if self._right_document is None:
            self.status_label.setText("[오류] 먼저 뷰어를 로드하세요.")
            return
        selected = self._get_checked_records()
        if not selected:
            self.status_label.setText("[오류] 체크된 항목이 없습니다.")
            return

        output_text = self.output_input.text().strip()
        if not output_text:
            self.status_label.setText("[오류] export 출력 경로를 지정하세요.")
            return
        output_json_path = output_text if output_text.lower().endswith((".json", ".jsonl")) \
            else os.path.join(output_text, "object_db_selected.json")

        try:
            exported = compare.export_selected(self._right_document, selected, output_json_path)
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.status_label.setText(f"[오류] {exc}")
            return
        self.status_label.setText(f"[OK] checked review rows exported\nSelected objects: {len(selected)}\nOutput JSON: {exported}")


def _hex_color(value: str) -> QColor:
    return QColor(value)


def _pil_to_qpixmap(image) -> QPixmap:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())  # copy(): data 버퍼 수명 분리


class PreviewCanvas(QLabel):
    """2번 검증 캔버스 — '박스 추가' 무장 상태에서 마우스 드래그로 새 박스를 그림.
    C# SourceCropPictureBox_MouseDown/Move/Up/Paint + TryMapControlPointToImagePoint 포팅."""

    boxDrawn = Signal(int, int, int, int)  # 이미지 좌표계 left, top, right, bottom

    def __init__(self) -> None:
        super().__init__("먼저 개체를 선택하세요.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(520, 520)
        self.setStyleSheet("background-color: #111; color: #ccc;")
        self._image_pixmap: Optional[QPixmap] = None
        self._armed = False
        self._dragging = False
        self._start: Optional[tuple] = None
        self._end: Optional[tuple] = None

    def set_armed(self, armed: bool) -> None:
        self._armed = armed
        self.setCursor(Qt.CursorShape.CrossCursor if armed else Qt.CursorShape.ArrowCursor)

    def set_image(self, image) -> None:
        self._image_pixmap = _pil_to_qpixmap(image) if image is not None else None
        self._rescale()

    def clear_image(self, message: str) -> None:
        self._image_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(message)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self) -> None:
        if self._image_pixmap is None or self._image_pixmap.isNull():
            return
        scaled = self._image_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled)

    def _display_rect(self) -> QRect:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return QRect()
        x = (self.width() - pixmap.width()) // 2
        y = (self.height() - pixmap.height()) // 2
        return QRect(x, y, pixmap.width(), pixmap.height())

    def _map_to_image(self, point: QPoint) -> Optional[tuple]:
        if self._image_pixmap is None or self._image_pixmap.isNull():
            return None
        rect = self._display_rect()
        if rect.width() <= 0 or rect.height() <= 0 or not rect.contains(point):
            return None
        x = round((point.x() - rect.left()) * self._image_pixmap.width() / rect.width())
        y = round((point.y() - rect.top()) * self._image_pixmap.height() / rect.height())
        x = max(0, min(x, self._image_pixmap.width()))
        y = max(0, min(y, self._image_pixmap.height()))
        return x, y

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._armed and event.button() == Qt.MouseButton.LeftButton:
            point = self._map_to_image(event.position().toPoint())
            if point is not None:
                self._dragging = True
                self._start = point
                self._end = point
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._dragging:
            point = self._map_to_image(event.position().toPoint())
            if point is not None:
                self._end = point
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._dragging:
            self._dragging = False
            point = self._map_to_image(event.position().toPoint())
            if point is not None:
                self._end = point
            left, right = sorted((self._start[0], self._end[0]))
            top, bottom = sorted((self._start[1], self._end[1]))
            self.update()
            if right - left >= 3 and bottom - top >= 3:
                self.boxDrawn.emit(left, top, right, bottom)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if self._dragging and self._start and self._end and self._image_pixmap:
            rect = self._display_rect()
            if rect.width() <= 0:
                return
            scale_x = rect.width() / self._image_pixmap.width()
            scale_y = rect.height() / self._image_pixmap.height()
            left, right = sorted((self._start[0], self._end[0]))
            top, bottom = sorted((self._start[1], self._end[1]))
            display_rect = QRect(rect.left() + round(left * scale_x), rect.top() + round(top * scale_y),
                                  round((right - left) * scale_x), round((bottom - top) * scale_y))
            painter = QPainter(self)
            pen = QPen(QColor("deepskyblue"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(display_rect)


_VERIFY_COLUMNS = ("개체", "날짜", "영상", "라벨", "겹침", "원본 좌표")


class SourceVerifyTab(QWidget):
    """2. 원본 검증 — object_db.json을 원본 TIF에서 크롭해 육안 검증, 박스 추가/삭제/저장."""

    def __init__(self) -> None:
        super().__init__()
        self._document: Optional[dict] = None
        self._records: list[dict] = []
        self._tiles_by_id: dict = {}
        self._current_record: Optional[dict] = None
        self._current_crop_bounds: Optional[tuple] = None

        self.object_json_input = QLineEdit()
        self.corrected_json_input = QLineEdit()
        self.source_root_input = QLineEdit()
        self.show_boundary_checkbox = QCheckBox("박스 표시")
        self.show_boundary_checkbox.setChecked(True)
        self.show_text_checkbox = QCheckBox("글자 표시")
        self.show_text_checkbox.setChecked(True)
        self.table = QTableWidget(0, len(_VERIFY_COLUMNS))
        self.table.setHorizontalHeaderLabels(_VERIFY_COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellClicked.connect(self._on_row_clicked)
        self.canvas = PreviewCanvas()
        self.canvas.boxDrawn.connect(self._on_box_drawn)
        self.status_label = QLabel("")

        form = QGridLayout()
        form.addWidget(QLabel("object JSON"), 0, 0)
        form.addWidget(self.object_json_input, 0, 1)
        form.addWidget(self._browse_file_button(self.object_json_input, "object JSON 선택"), 0, 2)

        form.addWidget(QLabel("수정본 JSON 저장 경로"), 1, 0)
        form.addWidget(self.corrected_json_input, 1, 1)
        form.addWidget(self._browse_save_button(self.corrected_json_input, "수정본 JSON 저장 경로 선택"), 1, 2)

        form.addWidget(QLabel("원본 TIF 루트 폴더"), 2, 0)
        form.addWidget(self.source_root_input, 2, 1)
        form.addWidget(self._browse_folder_button(self.source_root_input, "원본 TIF 루트 폴더 선택"), 2, 2)

        top_row = QHBoxLayout()
        load_button = QPushButton("불러오기")
        load_button.clicked.connect(self._on_load_clicked)
        top_row.addWidget(load_button)
        top_row.addWidget(self.show_boundary_checkbox)
        top_row.addWidget(self.show_text_checkbox)
        top_row.addStretch(1)

        self.add_box_button = QPushButton("박스 추가")
        self.add_box_button.clicked.connect(self._on_add_box_clicked)
        delete_button = QPushButton("개체 삭제")
        delete_button.clicked.connect(self._on_delete_clicked)
        save_button = QPushButton("수정본 JSON 저장")
        save_button.clicked.connect(self._on_save_clicked)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.add_box_button)
        edit_row.addWidget(delete_button)
        edit_row.addWidget(save_button)
        edit_row.addStretch(1)

        self.show_boundary_checkbox.toggled.connect(self._refresh_preview)
        self.show_text_checkbox.toggled.connect(self._refresh_preview)

        body_row = QHBoxLayout()
        body_row.addWidget(self.table, stretch=1)
        right_column = QVBoxLayout()
        right_column.addLayout(edit_row)
        right_column.addWidget(self.canvas, stretch=1)
        body_row.addLayout(right_column, stretch=2)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(top_row)
        layout.addLayout(body_row, stretch=1)
        layout.addWidget(self.status_label)

    @staticmethod
    def _browse_file_button(target: QLineEdit, caption: str) -> QPushButton:
        button = QPushButton("찾기...")

        def handler() -> None:
            path, _ = QFileDialog.getOpenFileName(None, caption, "", "JSON (*.json)")
            if path:
                target.setText(path)
        button.clicked.connect(handler)
        return button

    @staticmethod
    def _browse_save_button(target: QLineEdit, caption: str) -> QPushButton:
        button = QPushButton("찾기...")

        def handler() -> None:
            path, _ = QFileDialog.getSaveFileName(None, caption, "", "JSON (*.json)")
            if path:
                target.setText(path)
        button.clicked.connect(handler)
        return button

    @staticmethod
    def _browse_folder_button(target: QLineEdit, caption: str) -> QPushButton:
        button = QPushButton("찾기...")

        def handler() -> None:
            path = QFileDialog.getExistingDirectory(None, caption)
            if path:
                target.setText(path)
        button.clicked.connect(handler)
        return button

    def _on_load_clicked(self) -> None:
        path = self.object_json_input.text().strip()
        if not path:
            self.status_label.setText("[오류] object JSON 경로를 지정하세요.")
            return
        try:
            with open(path, encoding="utf-8") as fh:
                self._document = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.status_label.setText(f"[오류] {exc}")
            return

        self._records = list(self._document["objects"])
        self._tiles_by_id = {tile["tileId"]: tile for tile in self._document["tiles"]}
        if not self.corrected_json_input.text().strip():
            self.corrected_json_input.setText(sourceverify.get_corrected_json_path(path))
        self._populate_table()
        self.status_label.setText(f"Loaded {len(self._records)} objects.")

    def _populate_table(self) -> None:
        overlap_counts = sourceverify.build_overlap_count_map(self._records)
        self.table.setRowCount(len(self._records))
        for row_index, record in enumerate(self._records):
            box = record["globalBox"]
            overlap_count = overlap_counts.get(record["objectId"], 0)
            values = (str(record["objectId"]), record["captureDate"], record["sourceTifName"],
                      record.get("className", ""), str(overlap_count),
                      f"{box['left']},{box['top']},{box['right']},{box['bottom']}")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if overlap_count > 0:
                    item.setBackground(_hex_color("#fff8d6"))
                item.setData(Qt.ItemDataRole.UserRole, record)
                self.table.setItem(row_index, column, item)
        self.table.resizeColumnsToContents()

    def _on_row_clicked(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        record = item.data(Qt.ItemDataRole.UserRole)
        self._select_record(record)

    def _select_record(self, record: dict) -> None:
        self._current_record = record
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        record = self._current_record
        if record is None:
            return
        source_root = self.source_root_input.text().strip()
        if not source_root:
            self.status_label.setText("[오류] 원본 TIF 루트 폴더를 지정하세요.")
            return

        try:
            source_path = sourceimage.resolve(source_root, record["captureDate"], record["sourceBaseName"],
                                               record["sourceTifName"])
            overlaps = sourceverify.find_overlapping_records(self._records, record)
            excluded = {r["objectId"] for r in overlaps} | {record["objectId"]}
            others = sourceverify.find_same_source_records(self._records, record, excluded)
            image, crop_bounds = sourceverify.render_preview(
                source_path, record, overlaps, others,
                self.show_boundary_checkbox.isChecked(), self.show_text_checkbox.isChecked())
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.canvas.clear_image(f"미리보기 실패: {exc}")
            self.status_label.setText(f"[오류] {exc}")
            return

        self._current_crop_bounds = crop_bounds
        self.canvas.set_image(image)
        box = record["globalBox"]
        self.status_label.setText(
            f"Object {record['objectId']}, Date: {record['captureDate']}, Image: {record['sourceTifName']}, "
            f"Global box: {box['left']},{box['top']},{box['right']},{box['bottom']}, Overlaps: {len(overlaps)}")

    def _on_add_box_clicked(self) -> None:
        if self._current_record is None:
            self.status_label.setText("먼저 개체를 선택한 뒤 박스 추가를 누르세요.")
            return
        self.canvas.set_armed(True)
        self.add_box_button.setText("그리기 대기")
        self.status_label.setText("미리보기 이미지 위에서 새 박스를 드래그하세요.")

    def _on_box_drawn(self, left: int, top: int, right: int, bottom: int) -> None:
        self.canvas.set_armed(False)
        self.add_box_button.setText("박스 추가")
        context_record = self._current_record
        if context_record is None or self._current_crop_bounds is None:
            return

        crop_left, crop_top = self._current_crop_bounds[0], self._current_crop_bounds[1]
        global_box = {"left": crop_left + left, "top": crop_top + top,
                      "right": crop_left + right, "bottom": crop_top + bottom,
                      "width": right - left, "height": bottom - top}

        tile = self._tiles_by_id.get(context_record["tileId"])
        new_id = sourceverify.next_object_id(self._records)
        added = sourceverify.create_added_record(context_record, global_box, new_id, tile)
        sourceverify.insert_after(self._records, context_record["objectId"], added)
        self._populate_table()
        self._select_record(added)
        self.status_label.setText(f"Added object {added['objectId']}. Save corrected JSON when editing is complete.")

    def _on_delete_clicked(self) -> None:
        record = self._current_record
        if record is None:
            return
        overlaps = sourceverify.find_overlapping_records(self._records, record)
        next_selection_id = overlaps[0]["objectId"] if overlaps else None
        sourceverify.delete_record(self._records, record["objectId"])
        self._current_record = None
        self.canvas.clear_image("먼저 개체를 선택하세요.")
        self._populate_table()
        if next_selection_id is not None:
            next_record = next((r for r in self._records if r["objectId"] == next_selection_id), None)
            if next_record is not None:
                self._select_record(next_record)
        self.status_label.setText(f"Deleted object {record['objectId']}. Save corrected JSON when editing is complete.")

    def _on_save_clicked(self) -> None:
        if self._document is None:
            self.status_label.setText("[오류] 개체 JSON을 먼저 불러오세요.")
            return
        output_path = self.corrected_json_input.text().strip()
        if not output_path:
            self.status_label.setText("[오류] 수정본 JSON 저장 경로를 지정하세요.")
            return

        document = {**self._document, "objectCount": len(self._records), "objects": self._records}
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(document, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.status_label.setText(f"[오류] {exc}")
            return
        self.status_label.setText(f"Saved corrected JSON: {output_path}")


class UpdateBanner(QLabel):
    """업데이트 알림 배너 — 새 버전 있을 때만 나타남(없으면 높이 0, 자리 안 차지)."""

    def __init__(self) -> None:
        super().__init__()
        self.setOpenExternalLinks(True)
        self.setStyleSheet("background-color: #fff3cd; color: #664d03; padding: 6px;")
        self.hide()

    def show_message(self, html: str) -> None:
        self.setText(html)
        self.show()


def main() -> int:
    app = QApplication(sys.argv)
    window = QMainWindow()
    version_label = appversion.BUILD_VERSION or "dev"
    window.setWindowTitle(f"Training Data Extractor (PySide6 pilot) - {version_label}")

    tabs = QTabWidget()
    tabs.addTab(LabelDbTab(), "1. 라벨 DB")
    tabs.addTab(SourceVerifyTab(), "2. 원본 검증")
    tabs.addTab(TrainingTileTab(), "3. 학습 타일")
    tabs.addTab(YoloOrganizeTab(), "4. YOLO 정렬")
    tabs.addTab(TrainingTab(), "5. 학습")
    tabs.addTab(InferenceTab(), "6. 원본 추론")
    tabs.addTab(ReviewTab(), "7. 후보 검수")
    tabs.addTab(CompareTab(), "8. 매칭/선별")

    banner = UpdateBanner()
    central = QWidget()
    central_layout = QVBoxLayout(central)
    central_layout.setContentsMargins(0, 0, 0, 0)
    central_layout.addWidget(banner)
    central_layout.addWidget(tabs, stretch=1)
    window.setCentralWidget(central)
    window.resize(820, 560)
    window.show()

    update_worker = BackgroundCallWorker(updatecheck.check_for_update)
    update_worker.finished_ok.connect(lambda message: banner.show_message(message) if message else None)
    window._update_worker = update_worker  # QThread가 GC되지 않게 참조 유지
    update_worker.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
