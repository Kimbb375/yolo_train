from __future__ import annotations

import io
import itertools
import os
import re
import socket
import subprocess
import sys
import threading
from typing import Optional

from PySide6.QtCore import QObject, QPoint, QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QGroupBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import json

import batchbench
import centertile
import compare
import controlserver
import gpu_setup
import inference
import labelsync
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
        self._worker: BackgroundCallWorker | None = None

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
        self.build_button = QPushButton("DB 생성")
        self.build_button.clicked.connect(self._on_build_clicked)
        build_row.addWidget(self.build_button)
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

        # 라벨 폴더가 크면(타일 이미지 수천 장) 메인 스레드에서 그대로 돌릴 경우 창이 멈춘
        # 것처럼 보임 - 백그라운드로 뺌.
        self.log.setPlainText("DB 생성 중...\n")
        self.build_button.setEnabled(False)
        self._worker = BackgroundCallWorker(objectdb.build, source, output)
        self._worker.output.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    def _on_finished_ok(self, result) -> None:
        self.log.appendPlainText("\n" + result.to_display_text())
        self.output_input.setText(result.outputPath)
        self.build_button.setEnabled(True)

    def _on_finished_error(self, message: str) -> None:
        self.log.appendPlainText(f"\n[오류] {message}")
        self.build_button.setEnabled(True)


class TrainingTileTab(QWidget):
    """3. 학습 타일 — object_db.json + 원본 TIF -> 512/640 학습 크롭 + YOLO 라벨"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BackgroundCallWorker | None = None

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
        self.build_button = QPushButton("학습 타일 생성")
        self.build_button.clicked.connect(self._on_build_clicked)
        build_row.addWidget(self.build_button)
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

        # object 개수/원본 TIF 크기(특히 네트워크 공유 폴더)에 따라 수 분 걸릴 수 있어서
        # 메인 스레드에서 그냥 돌리면 그동안 창이 "응답 없음"처럼 멈춰 보임 - 백그라운드로 뺌.
        self.log.setPlainText("학습 타일 생성 중...\n")
        self.build_button.setEnabled(False)
        self._worker = BackgroundCallWorker(trainingdataset.build, object_db_path, source_root,
                                             output_root, sizes or None)
        self._worker.output.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    def _on_finished_ok(self, result) -> None:
        self.log.appendPlainText("\n" + result.to_display_text())
        self.build_button.setEnabled(True)

    def _on_finished_error(self, message: str) -> None:
        self.log.appendPlainText(f"\n[오류] {message}")
        self.build_button.setEnabled(True)


class CenterTileTab(QWidget):
    """2.2. 중앙 크롭 (보정용) — object_db.json + 원본 TIF -> 개체당 cc 크롭 1장 + YOLO 라벨.
    외부 라벨링 툴에서 박스 조절/추가 후 9번 TXT 보정 반영에 바로 넣을 수 있는 object_db.json도 같이 출력."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BackgroundCallWorker | None = None

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
        self.build_button = QPushButton("중앙 크롭 생성")
        self.build_button.clicked.connect(self._on_build_clicked)
        build_row.addWidget(self.build_button)
        build_row.addStretch(1)

        guide = QLabel(
            "3번(학습 타일)과 달리 개체당 중앙(cc) 크롭 1장만 만듭니다 - 외부 라벨링 프로그램에서 "
            "박스를 조절하거나 새 개체를 추가하기 쉽게 하려는 용도.\n"
            "출력 폴더의 object_db.json은 9번 'TXT 보정 반영' 탭의 '8번 기준 JSON'으로, "
            "labels 폴더(수정본)는 '보정 TXT 폴더'로 그대로 사용하면 됩니다.")
        guide.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(size_row)
        layout.addLayout(build_row)
        layout.addWidget(guide)
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

        self.log.setPlainText("중앙 크롭 생성 중...\n")
        self.build_button.setEnabled(False)
        self._worker = BackgroundCallWorker(centertile.build, object_db_path, source_root,
                                             output_root, sizes or None)
        self._worker.output.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    def _on_finished_ok(self, result) -> None:
        self.log.appendPlainText("\n" + result.to_display_text())
        self.build_button.setEnabled(True)

    def _on_finished_error(self, message: str) -> None:
        self.log.appendPlainText(f"\n[오류] {message}")
        self.build_button.setEnabled(True)


class YoloOrganizeTab(QWidget):
    """4. YOLO 정렬 — 3번 출력 -> train/val/test/predict YOLO 표준 구조"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BackgroundCallWorker | None = None

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
        self.build_button = QPushButton("YOLO 정렬 실행")
        self.build_button.clicked.connect(self._on_build_clicked)
        build_row.addWidget(self.build_button)
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

        self.log.setPlainText("YOLO 정렬 실행 중...\n")
        self.build_button.setEnabled(False)
        self._worker = BackgroundCallWorker(
            yolodataset.organize, source_root, target_root, self.image_size_input.value(),
            self.train_ratio_input.value(), self.val_ratio_input.value(),
            self.test_ratio_input.value(), self.predict_ratio_input.value(), self.seed_input.value())
        self._worker.output.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    def _on_finished_ok(self, result) -> None:
        self.log.appendPlainText("\n" + result.to_display_text())
        self.build_button.setEnabled(True)

    def _on_finished_error(self, message: str) -> None:
        self.log.appendPlainText(f"\n[오류] {message}")
        self.build_button.setEnabled(True)


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


class BackgroundProcessWorker(QThread):
    """실제 서브프로세스(torchrun)를 띄우고 stdout을 Qt 시그널로 스트리밍 (멀티노드 DDP 전용).

    BackgroundCallWorker와 달리 같은 프로세스 안에서 함수를 부르지 않고 진짜 별도 OS 프로세스를
    실행함 - training.build_multinode_command의 이유 참고."""

    output = Signal(str)
    finished_ok = Signal(object)
    finished_error = Signal(str)

    def __init__(self, command: list[str]) -> None:
        super().__init__()
        self._command = command
        self._process: subprocess.Popen | None = None

    def run(self) -> None:
        try:
            self._process = subprocess.Popen(
                self._command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001 - 프로세스 시작 자체 실패
            self.finished_error.emit(str(exc))
            return

        assert self._process.stdout is not None
        for line in self._process.stdout:
            self.output.emit(line)
        exit_code = self._process.wait()
        if exit_code == 0:
            self.finished_ok.emit(None)
        else:
            self.finished_error.emit(f"Training process exited with code {exit_code}.")

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()


class TrainingTab(QWidget):
    """5. 학습 — trainer/train.py를 서브프로세스 없이 같은 프로세스 안에서 실행."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BackgroundCallWorker | None = None

        # 6번(원본 추론)과 같은 방식으로 왼쪽 ControlPanel에서 고른 PC에 학습을 원격
        # 배포함(단, 멀티 노드 DDP는 노드마다 순번/마스터 IP를 직접 맞춰야 해서 원격 배포
        # 대상에서 제외 - 각 참여 PC에서 로컬로 직접 실행하는 기존 방식 그대로 씀).
        self._remote_agent_id: Optional[str] = None
        self._states: dict[Optional[str], _JobView] = {}
        self._active_remote_ids: set[str] = set()
        self._poll_timer: Optional[QTimer] = None
        self.target_label = QLabel()
        CONTROL_CONTEXT.target_changed.connect(self._on_control_target_changed)

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

        self.multinode_checkbox = QCheckBox("멀티 노드(DDP)")
        self.node_count_input = QSpinBox()
        self.node_count_input.setRange(1, 32)
        self.node_count_input.setValue(2)
        self.node_rank_input = QSpinBox()
        self.node_rank_input.setRange(0, 31)
        self.master_addr_input = QLineEdit()
        self.master_addr_input.setPlaceholderText("마스터 노드 IP (예: 192.168.0.10)")
        self.master_port_input = QLineEdit("29500")

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

        multinode_row = QHBoxLayout()
        multinode_row.addWidget(self.multinode_checkbox)
        multinode_row.addWidget(QLabel("노드 수"))
        multinode_row.addWidget(self.node_count_input)
        multinode_row.addWidget(QLabel("순번"))
        multinode_row.addWidget(self.node_rank_input)
        multinode_row.addWidget(self.master_addr_input, stretch=1)
        multinode_row.addWidget(QLabel("포트"))
        multinode_row.addWidget(self.master_port_input)

        self.start_button = QPushButton("학습 시작")
        self.start_button.clicked.connect(self._on_start_clicked)
        self.network_test_button = QPushButton("네트워크 테스트")
        self.network_test_button.setToolTip(
            "YOLO 학습 없이 위 노드 수/순번/마스터 IP/포트 설정으로 두 노드가 실제로 "
            "통신되는지만 빠르게 확인. 양쪽 컴퓨터에서 순번만 다르게 해서 같이 눌러야 함.")
        self.network_test_button.clicked.connect(self._on_network_test_clicked)
        build_row = QHBoxLayout()
        build_row.addWidget(self.start_button)
        build_row.addWidget(self.network_test_button)
        build_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.target_label)
        layout.addLayout(form)
        layout.addLayout(param_row)
        layout.addLayout(aug_row)
        layout.addLayout(multinode_row)
        layout.addLayout(build_row)
        layout.addWidget(self.log, stretch=1)

        self._on_control_target_changed(CONTROL_CONTEXT.selected_agent_id)

    def _on_control_target_changed(self, agent_id: Optional[str]) -> None:
        self._remote_agent_id = agent_id
        if agent_id is None:
            self.target_label.setText("제어 대상: 이 PC (로컬 실행)")
        else:
            self.target_label.setText(f"제어 대상: {agent_id} (원격 - 멀티 노드 DDP는 각 PC에서 로컬로 실행하세요)")
        # 멀티 노드는 노드마다 이 PC 자신의 순번/마스터 IP로 직접 실행해야 해서 원격 배포
        # 대상에서 제외함(네트워크 테스트도 마찬가지 이유).
        self.multinode_checkbox.setEnabled(agent_id is None)
        self.network_test_button.setEnabled(agent_id is None)
        self._render_target(agent_id)

    def _get_state(self, target_key: Optional[str]) -> _JobView:
        return self._states.setdefault(target_key, _JobView())

    def _render_target(self, target_key: Optional[str]) -> None:
        state = self._get_state(target_key)
        # 로컬은 원본 스트림을 조각(줄바꿈 안 끝난 것 포함) 그대로 이어붙인 것이라 구분자
        # 없이 합침. 원격은 서버가 이미 줄 단위로 잘라 보내준 것이라 줄바꿈으로 합침.
        text = "".join(state.summary) if target_key is None else "\n".join(state.summary)
        self.log.setPlainText(text)
        self.start_button.setEnabled(not state.running)

    @staticmethod
    def _browse_button(handler) -> QPushButton:
        button = QPushButton("찾기...")
        button.clicked.connect(handler)
        return button

    def _browse_remote(self, pick_files: bool, file_suffix: str = "") -> Optional[str]:
        """원격 대상 선택 중이면 그 워커 PC 경로를 탐색하는 창을 띄움 - 반환값이 None이 아니면
        (빈 문자열 포함 취소 제외) 그 경로를 그대로 씀. 원격이 아니면 None(로컬 QFileDialog로)."""
        if self._remote_agent_id is None:
            return None
        server = CONTROL_CONTEXT.server
        if server is None:
            return ""
        dialog = RemoteBrowseDialog(self, server, self._remote_agent_id, pick_files, file_suffix)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_path():
            return dialog.selected_path()
        return ""

    def _pick_dataset(self) -> None:
        remote = self._browse_remote(pick_files=False)
        path = remote if remote is not None else QFileDialog.getExistingDirectory(self, "YOLO 데이터셋 폴더 선택")
        if path:
            existing = self.dataset_input.toPlainText().strip()
            self.dataset_input.setPlainText((existing + "\n" + path).strip() if existing else path)

    def _pick_model(self) -> None:
        remote = self._browse_remote(pick_files=True, file_suffix=".pt")
        if remote is not None:
            if remote:
                self.model_input.setText(remote)
            return
        path, _ = QFileDialog.getOpenFileName(self, "초기 모델 pt 선택", "", "PyTorch model (*.pt)")
        if path:
            self.model_input.setText(path)

    def _pick_project(self) -> None:
        remote = self._browse_remote(pick_files=False)
        path = remote if remote is not None else QFileDialog.getExistingDirectory(self, "runs 출력 폴더 선택")
        if path:
            self.project_input.setText(path)

    def _on_start_clicked(self) -> None:
        dataset_roots = [line.strip() for line in self.dataset_input.toPlainText().splitlines() if line.strip()]
        model_path = self.model_input.text().strip()
        project = self.project_input.text().strip()
        if not dataset_roots or not model_path or not project:
            self.log.setPlainText("[오류] 데이터셋 폴더, 초기 모델, runs 출력 폴더를 모두 지정하세요.")
            return

        multinode = self.multinode_checkbox.isChecked()
        device = self.device_input.text().strip() or "auto"
        batch = self.batch_input.text().strip() or "auto"
        if multinode and "," in device:
            self.log.setPlainText("[오류] 멀티 노드에서는 device에 이 머신의 로컬 GPU 하나만 지정하세요 (예: 0).")
            return
        if multinode and batch.lower() == "auto":
            self.log.setPlainText("[오류] 멀티 노드에서는 batch를 숫자로 직접 지정하세요 (AutoBatch는 DDP 밖에서만 동작).")
            return
        if multinode and not self.master_addr_input.text().strip():
            self.log.setPlainText("[오류] 멀티 노드 마스터 노드 IP를 입력하세요.")
            return
        if multinode and self._remote_agent_id is not None:
            self.log.setPlainText("[오류] 멀티 노드 DDP는 원격 배포 대상이 아닙니다 - 이 PC(로컬)를 선택한 뒤 각 참여 PC에서 직접 실행하세요.")
            return

        try:
            args = training.build_train_args(
                dataset_roots, model_path, self.imgsz_input.value(), self.epochs_input.value(),
                batch, device, project, self.name_input.text().strip() or "yolo_whale",
                self.workers_input.value(), self.augmentation_input.text())
        except ValueError as exc:
            self.log.setPlainText(f"[오류] {exc}")
            return

        target_key = self._remote_agent_id
        self._states[target_key] = _JobView()
        self._states[target_key].running = True
        self.network_test_button.setEnabled(False)
        self._render_target(target_key)

        if target_key is not None:
            self._start_remote_training(target_key, args)
            return

        self._route_incoming(None, "학습 시작...\n")
        if multinode:
            command = training.build_multinode_command(
                args, self.node_count_input.value(), self.node_rank_input.value(),
                self.master_addr_input.text().strip(), self.master_port_input.text().strip() or "29500")
            self._worker = BackgroundProcessWorker(command)
        else:
            self._worker = BackgroundCallWorker(training.run, args)
        self._worker.output.connect(lambda text: self._route_incoming(None, text))
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _start_remote_training(self, agent_id: str, args: list[str]) -> None:
        server = CONTROL_CONTEXT.server
        if server is None:
            self._route_incoming(agent_id, "[오류] 서버가 꺼져 있습니다.")
            self._get_state(agent_id).running = False
            if agent_id == self._remote_agent_id:
                self.start_button.setEnabled(True)
            return

        self._route_incoming(agent_id, f"[{agent_id}]로 원격 학습 명령 전송...")
        server.queue_command(agent_id, {
            "type": "start_training", "args": args, "name": self.name_input.text().strip() or "yolo_whale"})
        self._active_remote_ids.add(agent_id)
        if self._poll_timer is None:
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(1000)
            self._poll_timer.timeout.connect(self._poll_all_remote)
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _poll_all_remote(self) -> None:
        server = CONTROL_CONTEXT.server
        if server is None or not self._active_remote_ids:
            if self._poll_timer is not None:
                self._poll_timer.stop()
            return

        agents_by_id = {a["agentId"]: a for a in server.snapshot()["agents"]}
        finished_ids = []
        for agent_id in list(self._active_remote_ids):
            agent = agents_by_id.get(agent_id)
            if agent is None:
                continue
            state = self._get_state(agent_id)
            log_tail = agent["logTail"]
            if state.log_seen > len(log_tail):
                state.log_seen = 0
            for line in log_tail[state.log_seen:]:
                self._route_incoming(agent_id, line)
            state.log_seen = len(log_tail)

            if agent["currentJob"] is None and agent["progress"].startswith(("완료:", "실패:")):
                self._route_incoming(agent_id, agent["progress"])
                state.running = False
                if agent_id == self._remote_agent_id:
                    self.start_button.setEnabled(True)
                    self.network_test_button.setEnabled(True)
                finished_ids.append(agent_id)
        for agent_id in finished_ids:
            self._active_remote_ids.discard(agent_id)
        if not self._active_remote_ids and self._poll_timer is not None:
            self._poll_timer.stop()

    def _route_incoming(self, target_key: Optional[str], text: str) -> None:
        state = self._get_state(target_key)
        state.summary.append(text)
        if target_key != self._remote_agent_id:
            return
        if target_key is None:
            self.log.moveCursor(self.log.textCursor().MoveOperation.End)
            self.log.insertPlainText(text)
        else:
            self.log.appendPlainText(text)

    def _on_network_test_clicked(self) -> None:
        if not self.master_addr_input.text().strip():
            self.log.setPlainText("[오류] 마스터 노드 IP를 입력하세요.")
            return

        self._route_incoming(None, "네트워크 테스트 시작 (양쪽 컴퓨터 다 눌러야 함)...\n")
        self.start_button.setEnabled(False)
        self.network_test_button.setEnabled(False)
        command = training.build_network_test_command(
            self.node_count_input.value(), self.node_rank_input.value(),
            self.master_addr_input.text().strip(), self.master_port_input.text().strip() or "29500")
        self._worker = BackgroundProcessWorker(command)
        self._worker.output.connect(lambda text: self._route_incoming(None, text))
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _on_finished_ok(self, _result) -> None:
        self._route_incoming(None, "\n[OK] 완료.\n")
        self._get_state(None).running = False
        if self._remote_agent_id is None:
            self.start_button.setEnabled(True)
            self.network_test_button.setEnabled(True)

    def _on_finished_error(self, message: str) -> None:
        self._route_incoming(None, f"\n[오류] {message}\n")
        self._get_state(None).running = False
        if self._remote_agent_id is None:
            self.start_button.setEnabled(True)
            self.network_test_button.setEnabled(True)


class _JobView:
    """InferenceTab이 제어 대상(로컬=None 또는 agent_id)별로 따로 들고 있는 로그/진행 상태.
    PC를 바꿔 클릭해도 그 PC 걸로 시작한 작업의 로그/진행률이 섞이거나 사라지지 않게
    분리해두는 용도(사용자 요청: "작업하는 프로세스는 선택하는 pc마다 다르게 표기")."""

    def __init__(self) -> None:
        self.summary: list[str] = []
        self.load: list[str] = []
        self.detail: list[str] = []
        self.progress_value = 0
        self.progress_max = 0
        self.running = False
        self.log_seen = 0  # 원격 전용: server의 logTail 중 어디까지 이미 반영했는지


class InferenceTab(QWidget):
    """6. 원본 추론 — 원본 TIF -> 내부 타일링 -> YOLO 추론 -> candidates.json"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BackgroundCallWorker | None = None
        self._gpu_worker: BackgroundCallWorker | None = None
        self._tensorrt_worker: BackgroundCallWorker | None = None
        self._bench_worker: BackgroundCallWorker | None = None
        self._selected_files: list[str] = []

        # 왼쪽 ControlPanel에서 다른 PC를 클릭하면 이 탭이 그 PC를 원격으로 조작하는 모드로
        # 바뀜(로컬 BackgroundCallWorker 대신 controlserver에 명령을 큐잉하고 로그를 폴링함).
        # PC별 로그/진행 상태는 _states에 따로 보관하고(None=로컬), 화면엔 그 중 현재 선택된
        # 대상 것만 렌더링함 - 다른 PC를 보고 있어도 폴링 자체는 계속되어 진행이 안 멈춤.
        self._remote_agent_id: Optional[str] = None
        self._states: dict[Optional[str], _JobView] = {}
        self._active_remote_ids: set[str] = set()
        self._poll_timer: Optional[QTimer] = None
        self.target_label = QLabel()
        CONTROL_CONTEXT.target_changed.connect(self._on_control_target_changed)

        self.gpu_status_label = QLabel()
        self.gpu_install_button = QPushButton("GPU torch 설치")
        self.gpu_install_button.clicked.connect(self._on_gpu_install_clicked)
        self._refresh_gpu_status()

        # 6-1(TensorRT 테스트)에서 engine=1로 첫 실행할 때 자동 설치되긴 하지만, 모델을 굳이
        # 돌리기 전에 미리 설치해두고 싶다는 요청 - GPU torch 설치 버튼과 같은 패턴.
        self.tensorrt_status_label = QLabel()
        self.tensorrt_install_button = QPushButton("TensorRT 설치")
        self.tensorrt_install_button.clicked.connect(self._on_tensorrt_install_clicked)
        self._refresh_tensorrt_status()

        self.optimize_button = QPushButton("최적 배치 검색")
        self.optimize_button.setToolTip(
            "이 GPU/모델/타일 크기로 실제 벤치마크를 돌려서 안전한 최대 batch 값을 찾고 "
            "옵션에 자동 반영합니다 (수십 초 소요).")
        self.optimize_button.clicked.connect(self._on_optimize_clicked)

        self.source_input = QLineEdit()
        self.source_input.textEdited.connect(self._on_source_edited_by_user)
        self.model_input = QLineEdit()
        self.output_input = QLineEdit()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(
            "비워두면 매번 새 폴더(시간 기준) - 중단 후 이어서 하려면 지난번과 같은 이름을 입력하세요")
        self.options_input = QLineEdit(
            "tile_mode=memory, resume=1, tile=640, overlap=0.2, conf=0.1, iou=0.6, imgsz=640, batch=auto, "
            "device=0, max_det=300, merge_iou=0.5, candidate_crop=640, candidate_context=120, candidate_view=tile")

        # C# InferenceTilingRunner 로그를 3개 창으로 분리해서 보여주던 걸 그대로 포팅:
        # Summary(요약/에러/완료) / Load-Prefetch(tif 로딩) / Infer-Save(타일 추론+저장 진행).
        self._line_buffer = ""
        self.summary_log = QPlainTextEdit()
        self.summary_log.setReadOnly(True)
        self.load_log = QPlainTextEdit()
        self.load_log.setReadOnly(True)
        self.detail_log = QPlainTextEdit()
        self.detail_log.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("전체 진행: %v / %m 파일 (%p%)")

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

        detail_split = QSplitter(Qt.Orientation.Horizontal)
        detail_split.addWidget(self._log_group("Load / Prefetch", self.load_log))
        detail_split.addWidget(self._log_group("Infer / Save", self.detail_log))

        log_split = QSplitter(Qt.Orientation.Vertical)
        log_split.addWidget(self._log_group("Summary", self.summary_log))
        log_split.addWidget(detail_split)
        log_split.setSizes([190, 300])

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.gpu_status_label)
        gpu_row.addWidget(self.gpu_install_button)
        gpu_row.addWidget(self.tensorrt_status_label)
        gpu_row.addWidget(self.tensorrt_install_button)
        gpu_row.addWidget(self.optimize_button)
        gpu_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.target_label)
        layout.addLayout(gpu_row)
        layout.addLayout(form)
        layout.addLayout(options_row)
        layout.addLayout(build_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(log_split, stretch=1)

        self._on_control_target_changed(CONTROL_CONTEXT.selected_agent_id)

    def _on_control_target_changed(self, agent_id: Optional[str]) -> None:
        self._remote_agent_id = agent_id
        if agent_id is None:
            self.target_label.setText("제어 대상: 이 PC (로컬 실행)")
        else:
            self.target_label.setText(f"제어 대상: {agent_id} (원격 - 이 PC의 GPU 상태/최적화는 적용 안 됨)")
        # GPU 설치/배치 최적화는 이 PC의 로컬 하드웨어를 대상으로 하는 기능이라 원격 대상일
        # 때는 의미가 없음(잘못 이해하고 중앙 PC에서 눌러버리는 걸 막으려고 비활성화).
        self.gpu_install_button.setEnabled(agent_id is None)
        self.tensorrt_install_button.setEnabled(agent_id is None)
        self.optimize_button.setEnabled(agent_id is None)
        self._render_target(agent_id)

    def _get_state(self, target_key: Optional[str]) -> _JobView:
        return self._states.setdefault(target_key, _JobView())

    def _render_target(self, target_key: Optional[str]) -> None:
        state = self._get_state(target_key)
        self.summary_log.setPlainText("\n".join(state.summary))
        self.load_log.setPlainText("\n".join(state.load))
        self.detail_log.setPlainText("\n".join(state.detail))
        self.progress_bar.setMaximum(state.progress_max or 1)
        self.progress_bar.setValue(state.progress_value)
        self.start_button.setEnabled(not state.running)

    @staticmethod
    def _log_group(title: str, content: QPlainTextEdit) -> QGroupBox:
        group = QGroupBox(title)
        box_layout = QVBoxLayout(group)
        box_layout.addWidget(content)
        return group

    @staticmethod
    def _browse_button(handler, label: str = "찾기...") -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(handler)
        return button

    def _browse_remote(self, target_input: QLineEdit, pick_files: bool, file_suffix: str = "") -> bool:
        """원격 대상 선택 중이면 그 워커 PC 경로를 탐색하는 창을 띄움 - 중앙 PC의
        QFileDialog는 중앙 PC 자신의 디스크만 보여줘서 워커 경로 입력시 "지정된 경로를
        찾을 수 없음" 에러가 났었음(사용자 보고). 반환값 True면 원격 처리로 대신함."""
        if self._remote_agent_id is None:
            return False
        server = CONTROL_CONTEXT.server
        if server is None:
            return True
        dialog = RemoteBrowseDialog(self, server, self._remote_agent_id, pick_files, file_suffix)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_path():
            target_input.setText(dialog.selected_path())
        return True

    def _pick_source(self) -> None:
        if self._browse_remote(self.source_input, pick_files=False):
            self._selected_files = []
            return
        path = QFileDialog.getExistingDirectory(self, "원본 TIF 폴더 선택")
        if path:
            self._selected_files = []
            self.source_input.setText(path)

    def _pick_source_files(self) -> None:
        # 다중 파일 선택은 원격 다이얼로그가 지원 안 함(단일 항목만 고름) - 원격 대상일 땐
        # 폴더 선택(_pick_source)으로 대신 지정하도록 안내.
        if self._remote_agent_id is not None:
            self.summary_log.appendPlainText(
                "\n[안내] 원격 대상에서는 다중 파일 선택 대신 폴더 선택을 쓰세요.")
            return
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
        if self._browse_remote(self.model_input, pick_files=True, file_suffix=".pt"):
            return
        path, _ = QFileDialog.getOpenFileName(self, "모델 pt 선택", "", "PyTorch model (*.pt)")
        if path:
            self.model_input.setText(path)

    def _pick_output(self) -> None:
        if self._browse_remote(self.output_input, pick_files=False):
            return
        path = QFileDialog.getExistingDirectory(self, "출력 폴더 선택")
        if path:
            self.output_input.setText(path)

    def _refresh_gpu_status(self) -> None:
        state = gpu_setup.status()
        text = {
            "available": "GPU torch: 설치되어 있음 (사용 가능)",
            "unavailable": "GPU torch: 설치했지만 이 PC에서 CUDA를 못 찾음 - CPU로 진행됨",
            "not_installed": "GPU torch: 설치 필요 (device=0 등으로 추론하려면 먼저 설치하세요)",
        }[state]
        self.gpu_status_label.setText(text)
        self.gpu_install_button.setVisible(state != "available")
        self.gpu_install_button.setText("GPU 재설치 시도" if state == "unavailable" else "GPU torch 설치")

    def _on_gpu_install_clicked(self) -> None:
        self.gpu_install_button.setEnabled(False)
        self.summary_log.appendPlainText("\nGPU torch 설치 시작...")
        self._gpu_worker = BackgroundCallWorker(lambda: gpu_setup.ensure_cuda_torch(force=True))
        self._gpu_worker.output.connect(self._on_worker_output)
        self._gpu_worker.finished_ok.connect(self._on_gpu_install_finished)
        self._gpu_worker.finished_error.connect(self._on_gpu_install_finished)
        self._gpu_worker.start()

    def _on_gpu_install_finished(self, _result=None) -> None:
        self.gpu_install_button.setEnabled(True)
        self._refresh_gpu_status()

    def _refresh_tensorrt_status(self) -> None:
        state = gpu_setup.tensorrt_status()
        text = {
            "available": "TensorRT: 설치되어 있음 (사용 가능)",
            "unavailable": "TensorRT: 이전 설치 시도 실패 - 재시도 가능",
            "not_installed": "TensorRT: 설치 필요 (6-1에서 engine=1 쓰려면 먼저 설치하세요)",
        }[state]
        self.tensorrt_status_label.setText(text)
        self.tensorrt_install_button.setVisible(state != "available")
        self.tensorrt_install_button.setText("TensorRT 재설치 시도" if state == "unavailable" else "TensorRT 설치")

    def _on_tensorrt_install_clicked(self) -> None:
        self.tensorrt_install_button.setEnabled(False)
        self.summary_log.appendPlainText("\nTensorRT 설치 시작...")
        self._tensorrt_worker = BackgroundCallWorker(lambda: gpu_setup.ensure_tensorrt(force=True))
        self._tensorrt_worker.output.connect(self._on_worker_output)
        self._tensorrt_worker.finished_ok.connect(self._on_tensorrt_install_finished)
        self._tensorrt_worker.finished_error.connect(self._on_tensorrt_install_finished)
        self._tensorrt_worker.start()

    def _on_tensorrt_install_finished(self, _result=None) -> None:
        self.tensorrt_install_button.setEnabled(True)
        self._refresh_tensorrt_status()

    @staticmethod
    def _get_option(options_text: str, key: str, default: str) -> str:
        for part in options_text.split(","):
            part = part.strip()
            if "=" in part and part.split("=", 1)[0].strip() == key:
                return part.split("=", 1)[1].strip()
        return default

    @staticmethod
    def _set_option(options_text: str, key: str, value) -> str:
        parts = [p.strip() for p in options_text.split(",") if p.strip()]
        for i, part in enumerate(parts):
            if part.split("=", 1)[0].strip() == key:
                parts[i] = f"{key}={value}"
                break
        else:
            parts.append(f"{key}={value}")
        return ", ".join(parts)

    def _on_optimize_clicked(self) -> None:
        model_path = self.model_input.text().strip()
        if not model_path:
            self.summary_log.appendPlainText("\n[오류] 먼저 모델 pt를 지정하세요.")
            return

        options_text = self.options_input.text()
        tile = int(float(self._get_option(options_text, "tile", "640")))
        imgsz = int(float(self._get_option(options_text, "imgsz", "640")))
        device = self._get_option(options_text, "device", "0")

        self.optimize_button.setEnabled(False)
        self.summary_log.appendPlainText("\n최적 배치 크기 탐색 중 (실제 GPU/CPU로 벤치마크, 수십 초 소요)...")
        self._bench_worker = BackgroundCallWorker(batchbench.run, model_path, tile, imgsz, device)
        self._bench_worker.output.connect(self._on_worker_output)
        self._bench_worker.finished_ok.connect(self._on_bench_finished)
        self._bench_worker.finished_error.connect(self._on_bench_error)
        self._bench_worker.start()

    def _on_bench_finished(self, result: batchbench.BatchBenchResult) -> None:
        self.optimize_button.setEnabled(True)
        self.options_input.setText(self._set_option(self.options_input.text(), "batch", result.recommendedBatch))
        self.summary_log.appendPlainText("\n" + result.to_display_text())

    def _on_bench_error(self, message: str) -> None:
        self.optimize_button.setEnabled(True)
        self.summary_log.appendPlainText(f"\n[오류] 배치 벤치마크 실패: {message}")

    def _on_start_clicked(self) -> None:
        source = ";".join(self._selected_files) if self._selected_files else self.source_input.text().strip()
        model_path = self.model_input.text().strip()
        output_root = self.output_input.text().strip()
        if not source or not model_path or not output_root:
            self.summary_log.setPlainText("[오류] 원본 TIF, 모델, 출력 폴더를 모두 지정하세요.")
            return

        target_key = self._remote_agent_id
        self._states[target_key] = _JobView()
        self._states[target_key].running = True
        self._line_buffer = ""
        self._render_target(target_key)  # 화면(현재 보고 있는 대상=target_key) 비우고 진행바 리셋

        if target_key is not None:
            self._start_remote(target_key, source, model_path, output_root)
            return

        self._route_incoming(None, "추론 시작...")
        self._worker = BackgroundCallWorker(
            inference.run, source, output_root, model_path,
            self.name_input.text().strip() or None, self.options_input.text())
        self._worker.output.connect(self._on_worker_output)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _start_remote(self, agent_id: str, source: str, model_path: str, output_root: str) -> None:
        server = CONTROL_CONTEXT.server
        if server is None:
            self._route_incoming(agent_id, "[오류] 서버가 꺼져 있습니다.")
            self._get_state(agent_id).running = False
            if agent_id == self._remote_agent_id:
                self.start_button.setEnabled(True)
            return

        self._route_incoming(agent_id, f"[{agent_id}]로 원격 추론 명령 전송...")
        server.queue_command(agent_id, {
            "type": "start_job", "source": source, "model": model_path, "output": output_root,
            "runName": self.name_input.text().strip() or None, "options": self.options_input.text(),
            "mirror": bool(CONTROL_CONTEXT.central_output_root),
        })
        self._active_remote_ids.add(agent_id)
        if self._poll_timer is None:
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(1000)
            self._poll_timer.timeout.connect(self._poll_all_remote)
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _poll_all_remote(self) -> None:
        server = CONTROL_CONTEXT.server
        if server is None or not self._active_remote_ids:
            if self._poll_timer is not None:
                self._poll_timer.stop()
            return

        agents_by_id = {a["agentId"]: a for a in server.snapshot()["agents"]}
        finished_ids = []
        for agent_id in list(self._active_remote_ids):
            agent = agents_by_id.get(agent_id)
            if agent is None:
                continue
            state = self._get_state(agent_id)
            log_tail = agent["logTail"]
            if state.log_seen > len(log_tail):
                state.log_seen = 0  # 서버가 오래된 로그를 정리함(500줄 상한) - 처음부터 다시 표시
            for line in log_tail[state.log_seen:]:
                self._route_incoming(agent_id, line)
            state.log_seen = len(log_tail)

            if agent["currentJob"] is None and agent["progress"].startswith(("완료:", "실패:")):
                self._route_incoming(agent_id, agent["progress"])
                state.running = False
                if agent_id == self._remote_agent_id:
                    self.start_button.setEnabled(True)
                finished_ids.append(agent_id)
        for agent_id in finished_ids:
            self._active_remote_ids.discard(agent_id)
        if not self._active_remote_ids and self._poll_timer is not None:
            self._poll_timer.stop()

    def _on_worker_output(self, text: str) -> None:
        # print()는 라인 조각 단위로 여러 번 emit되므로 줄바꿈 기준으로 모아서 줄 단위로 분류함.
        self._line_buffer += text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._route_incoming(None, line)

    def _route_incoming(self, target_key: Optional[str], line: str) -> None:
        """target_key(로컬=None/원격=agent_id)의 로그 버퍼에 항상 쌓고, 지금 화면에 보이는
        대상과 같을 때만 위젯에도 바로 반영함 - 다른 PC를 보고 있을 땐 조용히 버퍼에만
        쌓였다가, 그 PC로 다시 전환하면 _render_target이 한꺼번에 그려줌."""
        state = self._get_state(target_key)
        category = self._log_category_for(line)
        getattr(state, category).append(line)
        match = re.search(r"\[FILE PROGRESS\] (\d+)/(\d+)", line)
        if match:
            total = int(match.group(2))
            if total > 0:
                state.progress_max = total
                state.progress_value = min(int(match.group(1)), total)

        if target_key == self._remote_agent_id:
            self._log_widget_for(category).appendPlainText(line)
            if match and state.progress_max > 0:
                self.progress_bar.setMaximum(state.progress_max)
                self.progress_bar.setValue(state.progress_value)

    @staticmethod
    def _log_category_for(line: str) -> str:
        # C# InferenceTilingRunner의 IsTifLoadLog/IsTifDetailLog 분류 규칙 그대로 포팅.
        if line.startswith("Prefetch loading ") or (line.startswith("Processing ") and " loaded_in=" in line):
            return "load"
        if ": processed " in line or line.startswith("Saved intermediate candidates:"):
            return "detail"
        return "summary"

    def _log_widget_for(self, category: str) -> QPlainTextEdit:
        return {"load": self.load_log, "detail": self.detail_log, "summary": self.summary_log}[category]

    def _on_finished_ok(self, result) -> None:
        self._route_incoming(None, result.to_display_text())
        self._get_state(None).running = False
        if self._remote_agent_id is None:
            self.start_button.setEnabled(True)

    def _on_finished_error(self, message: str) -> None:
        self._route_incoming(None, f"[오류] {message}")
        self._get_state(None).running = False
        if self._remote_agent_id is None:
            self.start_button.setEnabled(True)


class InferenceTestTab(InferenceTab):
    """6-1. 원본 추론 테스트 — 6번과 완전히 동일한 로직이되, TensorRT engine 변환(engine=1)을
    기본으로 켜서 켜보는 실험용 탭. 배포 자동화가 번거로우면 이 탭만 지워도 6번엔 영향 없음."""

    def __init__(self) -> None:
        super().__init__()
        self.options_input.setText(self.options_input.text() + ", engine=1")


class CandidateImageLabel(QLabel):
    """후보 크롭 이미지 표시 영역. 클릭해서 포커스를 줘야 A/D/Space/X 키가 먹음(§알아둘 것).
    set_box()로 받은 박스(표시된 pixmap 픽셀 좌표)를 빨간 사각형으로 겹쳐 그림 - 저장된
    후보 crop 이미지 자체엔 박스가 없고(원본 crop 그대로) 좌표만 따로 있어서 필요함."""

    def __init__(self) -> None:
        super().__init__("후보를 불러오세요.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 480)
        self.setStyleSheet("background-color: #222; color: #ccc;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._box: Optional[tuple[float, float, float, float]] = None
        self._box_color = QColor("red")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.setFocus()
        super().mousePressEvent(event)

    def set_box(self, box: Optional[tuple[float, float, float, float]],
                color: Optional[QColor] = None) -> None:
        self._box = box
        if color is not None:
            self._box_color = color
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        pixmap = self.pixmap()
        if self._box is None or pixmap is None or pixmap.isNull():
            return
        offset_x = (self.width() - pixmap.width()) / 2.0
        offset_y = (self.height() - pixmap.height()) / 2.0
        left, top, right, bottom = self._box
        rect = QRect(round(offset_x + left), round(offset_y + top),
                     round(right - left), round(bottom - top))
        painter = QPainter(self)
        pen = QPen(self._box_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(rect)


class ReviewTab(QWidget):
    """7. 후보 검수 — candidates.json을 A/D로 넘기며 Space(고래 확정)/X(고래 아님)로 분류."""

    # 의심 후보(재검토 권장) 기준값 - 실제 필터로 자동 적용되진 않고 참고용으로만 표시.
    DEFAULT_FILTER_HINT = "conf<0.2, width<10, height>100, area<30"

    def __init__(self) -> None:
        super().__init__()
        self._candidates: list[review.ReviewCandidate] = []
        self._index = 0

        self.candidate_json_input = QLineEdit()
        self.output_root_input = QLineEdit()
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText(self.DEFAULT_FILTER_HINT)
        self.filter_input.textChanged.connect(self._update_filter_status_label)
        self.filter_status_label = QLabel()
        self.filter_status_label.setStyleSheet("color: #666;")
        self.jump_input = QSpinBox()
        self.jump_input.setRange(1, 1000)
        self.jump_input.setValue(1)
        self.image_label = CandidateImageLabel()
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.valueChanged.connect(self._on_slider_changed)
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
        layout.addWidget(self.position_slider)
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
        # DEFAULT_FILTER_HINT는 참고용 의심 후보 기준일 뿐 자동 적용 안 됨 - 필터 비우면
        # 실제로는 전체 표시가 기본 동작. "지금 몇 이상만 보고 있는지"를 항상 보이게 표기.
        default_text = f"필터 기준 — 기본값(참고용, 자동 적용 안 됨): {self.DEFAULT_FILTER_HINT}"
        text = self.filter_input.text().strip()
        if not text:
            self.filter_status_label.setText(f"{default_text} / 현재 설정: 없음 (전체 표시)")
            return
        try:
            review.parse_filters(text)
        except ValueError:
            self.filter_status_label.setText(f"{default_text} / 현재 설정: {text} [형식 오류]")
            return
        self.filter_status_label.setText(f"{default_text} / 현재 설정: {text}")

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
        # 추론이 계속 돌면서 candidates.json에 후보가 계속 추가되는 중에도 검수자가
        # 다시 불러오기/필터 적용을 누를 수 있음 - 그때마다 1번으로 되돌아가면 번거로우니
        # 지금 보던 후보(candidateId)를 새 목록에서 다시 찾아 그 자리를 유지함.
        current_candidate_id = self._current().candidateId if self._current() else None
        try:
            all_candidates = review.load_candidates(candidate_json_path)
            self._candidates = review.apply_filters(all_candidates, self.filter_input.text())
        except Exception as exc:  # noqa: BLE001 - UI 레이어, 사용자에게 원인 그대로 보여줌
            self.status_label.setText(f"[오류] {exc}")
            return
        self._index = self._find_index_by_candidate_id(current_candidate_id)
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(0, max(0, len(self._candidates) - 1))
        self.position_slider.blockSignals(False)
        self.status_label.setText(f"{len(all_candidates)}개 중 {len(self._candidates)}개 표시 (필터 적용됨)")
        self.image_label.setFocus()
        self._refresh()

    def _on_slider_changed(self, value: int) -> None:
        if not self._candidates or value == self._index:
            return
        self._index = value
        self._refresh()

    def _find_index_by_candidate_id(self, candidate_id: Optional[int]) -> int:
        if candidate_id is not None:
            for i, candidate in enumerate(self._candidates):
                if candidate.candidateId == candidate_id:
                    return i
        return 0

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
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(self._index)
        self.position_slider.blockSignals(False)

        candidate = self._current()
        if candidate is None:
            self.image_label.setText("표시할 후보가 없습니다.")
            self.image_label.set_box(None)
            self.info_label.setText("-")
            return

        output_root = self._output_root()
        status = []
        if output_root and review.is_confirmed(candidate, output_root):
            status.append("CONFIRMED")
        if output_root and review.is_negative(candidate, output_root):
            status.append("NEGATIVE")
        status_text = "/".join(status) if status else "미분류"
        # 확정(Space)=초록, 고래 아님(X)=노랑, 미분류=빨강 - 검수 상태를 박스 색으로 바로 보이게 함.
        box_color = QColor("lime") if "CONFIRMED" in status else (
            QColor("yellow") if "NEGATIVE" in status else QColor("red"))

        image_path = candidate.resolved_image_path()
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText(f"이미지를 불러올 수 없음: {image_path}")
            self.image_label.set_box(None)
        else:
            scaled = pixmap.scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            crop = candidate.candidateCropBox
            if crop.width > 0 and crop.height > 0:
                box = candidate.globalBox
                scale_x = scaled.width() / crop.width
                scale_y = scaled.height() / crop.height
                self.image_label.set_box((
                    (box.left - crop.left) * scale_x, (box.top - crop.top) * scale_y,
                    (box.right - crop.left) * scale_x, (box.bottom - crop.top) * scale_y), box_color)
            else:
                self.image_label.set_box(None)

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
        self._load_worker: BackgroundCallWorker | None = None
        self._export_worker: BackgroundCallWorker | None = None
        self._pending_export_count = 0

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
        self.load_button = QPushButton("뷰어 로드")
        self.load_button.clicked.connect(self._on_load_clicked)
        top_row.addWidget(self.load_button)
        top_row.addStretch(1)

        select_row = QHBoxLayout()
        for label, handler in (("전체", lambda: self._set_checks(lambda status: True)),
                                ("해제", lambda: self._set_checks(lambda status: False)),
                                ("신규", lambda: self._set_checks(lambda status: status == "NEW")),
                                ("일치", lambda: self._set_checks(lambda status: status == "MATCH"))):
            button = QPushButton(label)
            button.clicked.connect(handler)
            select_row.addWidget(button)
        self.export_button = QPushButton("체크 내보내기")
        self.export_button.clicked.connect(self._on_export_clicked)
        select_row.addWidget(self.export_button)
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

    @staticmethod
    def _load_compare_data(left_path: str, right_path: str, threshold: float) -> dict:
        left_document = compare.load_compare_document(left_path)
        right_document = compare.load_compare_document(right_path)
        left_objects = left_document["objects"]
        right_objects = right_document["objects"]
        return {
            "left_document": left_document, "right_document": right_document,
            "left_rows": compare.build_compare_rows(left_objects, right_objects, True, threshold),
            "right_rows": compare.build_compare_rows(right_objects, left_objects, False, threshold),
        }

    def _on_load_clicked(self) -> None:
        left_path = self.left_input.text().strip()
        right_path = self.right_input.text().strip()
        if not left_path or not right_path:
            self.status_label.setText("[오류] 왼쪽/오른쪽 데이터 경로를 모두 지정하세요.")
            return

        # object 개수가 많으면 IoU 매칭이 O(n*m)이라 메인 스레드에서 그대로 돌리면 창이
        # 멈춘 것처럼 보임 - 백그라운드로 뺌.
        self.status_label.setText("불러오는 중...")
        self.load_button.setEnabled(False)
        self._load_worker = BackgroundCallWorker(
            self._load_compare_data, left_path, right_path, self.iou_input.value())
        self._load_worker.finished_ok.connect(self._on_load_finished)
        self._load_worker.finished_error.connect(self._on_load_error)
        self._load_worker.start()

    def _on_load_finished(self, data: dict) -> None:
        self.load_button.setEnabled(True)
        self._left_document = data["left_document"]
        self._right_document = data["right_document"]
        self._populate_table(self.left_table, data["left_rows"])
        self._populate_table(self.right_table, data["right_rows"])

        threshold = self.iou_input.value()
        left_objects = self._left_document["objects"]
        right_objects = self._right_document["objects"]
        matched = sum(1 for row in data["left_rows"] if row["status"] == "MATCH")
        self.status_label.setText(
            f"Base objects : {len(left_objects)}  |  Review/new : {len(right_objects)}  |  "
            f"Matched : {matched}  |  Missed : {len(left_objects) - matched}  |  "
            f"New : {sum(1 for row in data['right_rows'] if row['status'] == 'NEW')}  |  "
            f"IoU threshold : {threshold:.2f}")

    def _on_load_error(self, message: str) -> None:
        self.load_button.setEnabled(True)
        self.status_label.setText(f"[오류] {message}")

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

        # 체크한 개체가 많으면 타일 이미지/라벨 파일 복사가 오래 걸릴 수 있어서 백그라운드로 뺌.
        self.status_label.setText("내보내는 중...")
        self.export_button.setEnabled(False)
        self._pending_export_count = len(selected)
        self._export_worker = BackgroundCallWorker(
            compare.export_selected, self._right_document, selected, output_json_path)
        self._export_worker.finished_ok.connect(self._on_export_finished)
        self._export_worker.finished_error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_finished(self, exported: str) -> None:
        self.export_button.setEnabled(True)
        self.status_label.setText(
            f"[OK] checked review rows exported\nSelected objects: {self._pending_export_count}\n"
            f"Output JSON: {exported}")

    def _on_export_error(self, message: str) -> None:
        self.export_button.setEnabled(True)
        self.status_label.setText(f"[오류] {message}")


class LabelSyncTab(QWidget):
    """9. TXT 보정 반영 — 외부 라벨링 프로그램에서 고친 YOLO TXT를 기준으로 8번 Object DB의
    객체 좌표/삭제/추가를 동기화."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BackgroundCallWorker | None = None

        self.base_json_input = QLineEdit()
        self.labels_root_input = QLineEdit()
        self.output_json_input = QLineEdit()
        self.status_label = QPlainTextEdit()
        self.status_label.setReadOnly(True)

        form = QGridLayout()
        form.addWidget(QLabel("8번 기준 JSON"), 0, 0)
        form.addWidget(self.base_json_input, 0, 1)
        form.addWidget(self._browse_file_button(self.base_json_input), 0, 2)

        form.addWidget(QLabel("보정 TXT 폴더"), 1, 0)
        form.addWidget(self.labels_root_input, 1, 1)
        form.addWidget(self._browse_folder_button(self.labels_root_input), 1, 2)

        form.addWidget(QLabel("출력 JSON (비우면 <기준>_txt_synced.json)"), 2, 0)
        form.addWidget(self.output_json_input, 2, 1)
        form.addWidget(self._browse_save_button(self.output_json_input), 2, 2)

        guide = QLabel(
            "TXT 기준 규칙: 수정된 YOLO 라벨(0 cx cy width height)이 최종 값입니다. 빈 TXT는 해당 타일의 "
            "객체 삭제로 처리하며, 여러 줄은 객체 추가로 반영합니다.\n"
            "보정 TXT 폴더에는 8번 내보내기 폴더 자체 또는 그 안의 labels 폴더를 지정하세요. JSON에 등록된 "
            "TXT가 없으면 안전을 위해 기존 객체는 유지하고 경고만 남깁니다.")
        guide.setWordWrap(True)

        self.apply_button = QPushButton("TXT 보정 반영")
        self.apply_button.clicked.connect(self._on_apply_clicked)
        apply_row = QHBoxLayout()
        apply_row.addWidget(self.apply_button)
        apply_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(guide)
        layout.addLayout(apply_row)
        layout.addWidget(self.status_label, stretch=1)

    def _browse_file_button(self, target: QLineEdit) -> QPushButton:
        button = QPushButton("파일...")
        button.clicked.connect(lambda: self._pick_file(target))
        return button

    def _pick_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "8번 기준 JSON 선택", "", "JSON (*.json)")
        if path:
            target.setText(path)

    def _browse_folder_button(self, target: QLineEdit) -> QPushButton:
        button = QPushButton("폴더...")
        button.clicked.connect(lambda: self._pick_folder(target))
        return button

    def _pick_folder(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "보정 TXT 폴더 선택")
        if path:
            target.setText(path)

    def _browse_save_button(self, target: QLineEdit) -> QPushButton:
        button = QPushButton("저장...")
        button.clicked.connect(lambda: self._pick_save(target))
        return button

    def _pick_save(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "동기화 결과 JSON 저장 경로", "object_db_txt_synced.json", "JSON (*.json)")
        if path:
            target.setText(path)

    def _on_apply_clicked(self) -> None:
        base_json_path = self.base_json_input.text().strip()
        if not base_json_path:
            self.status_label.setPlainText("[오류] '8번 기준 JSON' 경로를 입력하거나 [파일...] 버튼으로 선택하세요.")
            return
        if not os.path.isfile(base_json_path):
            self.status_label.setPlainText(f"[오류] 8번 기준 JSON 파일을 찾을 수 없습니다: {base_json_path}")
            return
        labels_root = self.labels_root_input.text().strip()
        if not labels_root:
            self.status_label.setPlainText("[오류] '보정 TXT 폴더' 경로를 입력하거나 [폴더...] 버튼으로 선택하세요.")
            return
        if not os.path.isdir(labels_root):
            self.status_label.setPlainText(f"[오류] 보정 TXT 폴더를 찾을 수 없습니다: {labels_root}")
            return

        output_path = self.output_json_input.text().strip() or None
        self.status_label.setPlainText("보정 TXT와 Object DB를 동기화하는 중...\n")
        self.apply_button.setEnabled(False)
        self._worker = BackgroundCallWorker(labelsync.synchronize, base_json_path, labels_root, output_path)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.start()

    def _on_finished_ok(self, result) -> None:
        self.apply_button.setEnabled(True)
        self.output_json_input.setText(result.outputJsonPath)
        self.status_label.setPlainText(result.to_display_text())

    def _on_finished_error(self, message: str) -> None:
        self.apply_button.setEnabled(True)
        self.status_label.appendPlainText(f"[오류] {message}")


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


# 여러 RemoteBrowseDialog 인스턴스(같은 워커를 대상으로 폴더/모델/출력을 연달아 고를 때
# 등)가 각자 0부터 requestId를 매기면, 서로 다른 다이얼로그의 요청이 같은 숫자로 겹쳐서
# (컨트롤서버의 dir_result는 에이전트당 마지막 결과 하나만 들고 있음) 엉뚱한 결과가 매칭될
# 여지가 있었음 - 프로세스 전체에서 공유하는 카운터로 바꿔서 항상 유일하게 만듦.
_DIR_REQUEST_IDS = itertools.count(1)


class RemoteBrowseDialog(QDialog):
    """원격 워커 PC의 파일시스템을 탐색해서 경로를 고르는 창. 원격 대상 선택 중엔 QFileDialog가
    중앙 PC 자신의 디스크만 보여줘서 "지정한 경로를 찾을 수 없음" 에러가 났음 - 실제로 그
    워커 PC에 list_dir 명령을 보내(기존 폴링 프로토콜 재사용) 받아온 목록을 보여줌. 폴링
    주기만큼 느리지만 새 프로토콜 없이 구현 가능해서 이 방식으로 함. 주소창에 경로를 직접
    타이핑해서 이동할 수도 있음(로컬 탐색기 주소창처럼) - 목록에 안 뜨는 UNC 경로(NAS 등,
    예: \\\\nas\\share)도 워커 PC가 접근 가능하기만 하면 이렇게 바로 들어갈 수 있음."""

    def __init__(self, parent, server: "controlserver.ControlServer", agent_id: str,
                 pick_files: bool = False, file_suffix: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{agent_id}의 경로 선택")
        self.resize(560, 440)
        self._server = server
        self._agent_id = agent_id
        self._pick_files = pick_files
        self._file_suffix = file_suffix.lower()
        self._request_id = 0
        self._current_path = ""
        self._selected_path: Optional[str] = None

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(r"경로 직접 입력(예: \\nas\share\폴더) 후 Enter 또는 이동")
        self.path_input.returnPressed.connect(self._on_go_clicked)
        go_button = QPushButton("이동")
        go_button.clicked.connect(self._on_go_clicked)
        self.up_button = QPushButton("상위 폴더")
        self.up_button.clicked.connect(self._go_up)
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.status_label = QLabel("")
        self.select_button = QPushButton("파일 선택" if pick_files else "이 폴더 선택")
        self.select_button.clicked.connect(self._on_select_clicked)
        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(self.reject)

        top_row = QHBoxLayout()
        top_row.addWidget(self.up_button)
        top_row.addWidget(self.path_input, stretch=1)
        top_row.addWidget(go_button)
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.select_button)
        bottom_row.addWidget(cancel_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(self.list_widget, stretch=1)
        layout.addWidget(self.status_label)
        layout.addLayout(bottom_row)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_result)
        self._request_listing("")

    def selected_path(self) -> Optional[str]:
        return self._selected_path

    def _on_go_clicked(self) -> None:
        self._request_listing(self.path_input.text().strip())

    def _request_listing(self, path: str) -> None:
        self._current_path = path
        self._request_id = next(_DIR_REQUEST_IDS)
        self.status_label.setText("탐색 중...")
        self.list_widget.clear()
        self.path_input.setText(path)
        self._server.queue_command(self._agent_id, {
            "type": "list_dir", "requestId": self._request_id, "path": path})
        self._timer.start()

    def _poll_result(self) -> None:
        result = self._server.get_dir_result(self._agent_id)
        if result is None or result.get("requestId") != self._request_id:
            return
        self._timer.stop()
        if result.get("error"):
            self.status_label.setText(f"[오류] {result['error']}")
            return
        self.status_label.setText("")
        for entry in result.get("entries", []):
            if not entry["isDir"]:
                if not self._pick_files:
                    continue
                if self._file_suffix and not entry["name"].lower().endswith(self._file_suffix):
                    continue
            prefix = "\U0001F4C1 " if entry["isDir"] else "\U0001F4C4 "
            item = QListWidgetItem(prefix + entry["name"])
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.list_widget.addItem(item)

    def _go_up(self) -> None:
        if not self._current_path:
            return
        trimmed = self._current_path.rstrip("\\/")
        parent = os.path.dirname(trimmed)
        self._request_listing(parent if parent and parent != trimmed else "")

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        entry = item.data(Qt.ItemDataRole.UserRole)
        if entry["isDir"]:
            self._request_listing(entry["path"])
        elif self._pick_files:
            self._selected_path = entry["path"]
            self.accept()

    def _on_select_clicked(self) -> None:
        if self._pick_files:
            item = self.list_widget.currentItem()
            if item is None:
                self.status_label.setText("[오류] 파일을 선택하세요.")
                return
            entry = item.data(Qt.ItemDataRole.UserRole)
            if entry["isDir"]:
                self.status_label.setText("[오류] 파일을 선택하세요(폴더 아님).")
                return
            self._selected_path = entry["path"]
        else:
            self._selected_path = self._current_path
        self.accept()


class ControlContext(QObject):
    """전역 "지금 이 UI가 어느 PC를 조작 중인지" 상태. 왼쪽 ControlPanel에서 PC를 클릭하면
    여기가 바뀌고, InferenceTab은 이 신호를 구독해서 로컬 실행 vs 원격 명령 전송을 스스로
    전환함 - 별도 "제어 탭"이 아니라 기존 탭 자체가 선택된 PC를 조작하는 방식(사용자 요청)."""

    target_changed = Signal(object)  # agent_id: str 또는 None(로컬 = 이 PC 자신)

    def __init__(self) -> None:
        super().__init__()
        self.server: Optional[controlserver.ControlServer] = None
        self.central_output_root: str = ""
        self.selected_agent_id: Optional[str] = None

    def set_target(self, agent_id: Optional[str]) -> None:
        self.selected_agent_id = agent_id
        self.target_changed.emit(agent_id)


CONTROL_CONTEXT = ControlContext()


class _AgentThread(QThread):
    """agent.py의 run_agent()를 GUI 프로세스 안에서 돌리는 QThread. 워커 PC도 이 앱을
    그대로 켜두고 왼쪽 패널에서 중앙 PC 주소만 입력하면 접속되게 하려는 것 - 예전처럼
    별도로 --agent 커맨드라인을 띄울 필요 없음(그 방식도 여전히 됨, agent.py는 안 바뀜)."""

    status_changed = Signal(str)

    def __init__(self, server: str, token: str, agent_id: str) -> None:
        super().__init__()
        self._server = server
        self._token = token
        self._agent_id = agent_id
        self._stop_event = threading.Event()

    def run(self) -> None:
        import agent as agent_module
        try:
            agent_module.run_agent(self._server, self._token, self._agent_id,
                                    stop_event=self._stop_event, log=self.status_changed.emit)
        except Exception as exc:  # noqa: BLE001 - 접속 스레드 죽는 대신 상태 라벨에 표시
            self.status_changed.emit(f"[오류] {exc}")

    def stop(self) -> None:
        self._stop_event.set()


class ControlPanel(QWidget):
    """왼쪽 고정 패널 - 중앙 서버 시작 + agent.py로 접속한 워커 PC 목록. 목록에서 PC를
    클릭하면 CONTROL_CONTEXT가 바뀌고, 6번(원본 추론) 탭이 그 PC를 원격으로 제어하는 모드로
    전환됨(작업 배포는 6번 탭에서 그대로 함 - 폼을 여기 따로 안 둠)."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(230)
        self.setMaximumWidth(340)
        self._agent_thread: Optional[_AgentThread] = None

        self.worker_server_input = QLineEdit()
        self.worker_server_input.setPlaceholderText("중앙 PC 주소, 예: http://192.168.0.10:8765")
        self.worker_token_input = QLineEdit()
        self.worker_token_input.setPlaceholderText("중앙 PC와 같은 토큰")
        self.worker_name_input = QLineEdit()
        self.worker_name_input.setPlaceholderText("비우면 이 PC 호스트명 사용")
        self.worker_connect_button = QPushButton("중앙에 접속(이 PC를 워커로)")
        self.worker_connect_button.clicked.connect(self._on_worker_connect_clicked)
        self.worker_status_label = QLabel("연결 안 됨")
        self.worker_status_label.setWordWrap(True)

        self.port_input = QLineEdit("8765")
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("워커 PC와 공유할 비밀 문자열")
        self.server_toggle_button = QPushButton("서버 시작")
        self.server_toggle_button.clicked.connect(self._on_toggle_server)
        self.server_status_label = QLabel("서버 꺼짐")
        self.server_status_label.setWordWrap(True)

        self.central_output_input = QLineEdit()
        self.central_output_input.setPlaceholderText("(선택) 워커 결과를 이 PC 경로로도 자동 복사")
        self.central_output_input.textChanged.connect(self._on_central_output_changed)
        central_output_button = QPushButton("찾기...")
        central_output_button.clicked.connect(self._pick_central_output)

        self.pc_list = QListWidget()
        self.pc_list.itemClicked.connect(self._on_pc_clicked)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        central_output_row = QHBoxLayout()
        central_output_row.addWidget(self.central_output_input, stretch=1)
        central_output_row.addWidget(central_output_button)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>이 PC를 워커로 쓰기</b>"))
        layout.addWidget(self.worker_server_input)
        layout.addWidget(self.worker_token_input)
        layout.addWidget(self.worker_name_input)
        layout.addWidget(self.worker_connect_button)
        layout.addWidget(self.worker_status_label)
        layout.addWidget(QLabel("<b>중앙 제어</b>"))
        layout.addWidget(QLabel("포트"))
        layout.addWidget(self.port_input)
        layout.addWidget(QLabel("토큰"))
        layout.addWidget(self.token_input)
        layout.addWidget(self.server_toggle_button)
        layout.addWidget(self.server_status_label)
        layout.addWidget(QLabel("중앙 저장 경로"))
        layout.addLayout(central_output_row)
        layout.addWidget(QLabel("PC 목록 (클릭 = 제어 대상 전환)"))
        layout.addWidget(self.pc_list, stretch=1)
        layout.addWidget(QLabel("선택한 PC 로그"))
        layout.addWidget(self.log_view, stretch=1)

        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._refresh)
        self._populate_local_only()

    def _populate_local_only(self) -> None:
        self.pc_list.clear()
        item = QListWidgetItem("이 PC (중앙)")
        item.setData(Qt.ItemDataRole.UserRole, None)
        self.pc_list.addItem(item)
        self.pc_list.setCurrentRow(0)

    def _pick_central_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "중앙 저장 경로 선택")
        if path:
            self.central_output_input.setText(path)  # -> _on_central_output_changed

    def _on_central_output_changed(self, text: str) -> None:
        CONTROL_CONTEXT.central_output_root = text.strip()
        if CONTROL_CONTEXT.server is not None:
            CONTROL_CONTEXT.server.set_mirror_root(CONTROL_CONTEXT.central_output_root)

    def _on_toggle_server(self) -> None:
        if CONTROL_CONTEXT.server is not None:
            CONTROL_CONTEXT.server.stop()
            CONTROL_CONTEXT.server = None
            self._timer.stop()
            self.server_status_label.setText("서버 꺼짐")
            self.server_toggle_button.setText("서버 시작")
            self.port_input.setEnabled(True)
            self.token_input.setEnabled(True)
            self._populate_local_only()
            CONTROL_CONTEXT.set_target(None)
            return

        token = self.token_input.text().strip()
        if not token:
            self.server_status_label.setText("[오류] 토큰을 먼저 입력하세요.")
            return
        try:
            port = int(self.port_input.text().strip())
        except ValueError:
            self.server_status_label.setText("[오류] 포트는 숫자여야 합니다.")
            return

        server = controlserver.ControlServer(token)
        try:
            server.start(port)
        except OSError as exc:
            self.server_status_label.setText(f"[오류] 서버 시작 실패: {exc}")
            return
        server.set_mirror_root(self.central_output_input.text().strip())
        CONTROL_CONTEXT.server = server
        local_ip = self._local_lan_ip()
        self.server_status_label.setText(f"서버 켜짐 ({local_ip}:{port}) - 워커 PC 쪽 주소/토큰란에 그대로 넣으면 됨")
        self.server_toggle_button.setText("서버 중지")
        self.port_input.setEnabled(False)
        self.token_input.setEnabled(False)
        # 이 PC 스스로도 워커로 붙일 수 있게(예: 중앙 PC에 GPU가 있는 경우), 그리고 다른
        # 물리 PC에 알려줄 주소/토큰을 따로 찾아 적을 필요 없게 워커란에 바로 채워줌.
        self.worker_server_input.setText(f"http://{local_ip}:{port}")
        self.worker_token_input.setText(token)
        self._timer.start()

    @staticmethod
    def _local_lan_ip() -> str:
        # 실제 패킷은 안 보내고 소켓에 목적지만 지정해서 OS가 고르는 아웃바운드 인터페이스의
        # IP를 얻는 표준적인 방법 - 사내망 IP(예: 192.168.x.x)를 안내문에 그대로 쓰기 위함.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(0.5)
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except OSError:
            try:
                return socket.gethostbyname(socket.gethostname())
            except OSError:
                return "127.0.0.1"

    def _on_worker_connect_clicked(self) -> None:
        if self._agent_thread is not None:
            self._agent_thread.stop()
            self._agent_thread.wait(3000)
            self._agent_thread = None
            self.worker_status_label.setText("연결 안 됨")
            self.worker_connect_button.setText("중앙에 접속(이 PC를 워커로)")
            self.worker_server_input.setEnabled(True)
            self.worker_token_input.setEnabled(True)
            self.worker_name_input.setEnabled(True)
            return

        server_url = self.worker_server_input.text().strip()
        token = self.worker_token_input.text().strip()
        if not server_url or not token:
            self.worker_status_label.setText("[오류] 중앙 PC 주소와 토큰을 입력하세요.")
            return
        agent_id = self.worker_name_input.text().strip() or socket.gethostname()

        self._agent_thread = _AgentThread(server_url, token, agent_id)
        self._agent_thread.status_changed.connect(self.worker_status_label.setText)
        self._agent_thread.finished.connect(self._on_agent_thread_finished)
        self._agent_thread.start()
        self.worker_connect_button.setText("접속 해제")
        self.worker_server_input.setEnabled(False)
        self.worker_token_input.setEnabled(False)
        self.worker_name_input.setEnabled(False)

    def _on_agent_thread_finished(self) -> None:
        # 정상 stop()이든 예상 못한 예외든, 스레드가 끝나면 항상 UI를 리셋해서 다음 클릭이
        # 무조건 "새로 접속 시도"가 되게 함 - 안 그러면 죽은 스레드 참조가 남아서 사용자가
        # 버튼을 눌러도 그 참조 정리만 하고 실제 재접속은 한 번 더 눌러야 되는 문제가 있었음
        # (사용자 보고: "첫 등록 때 오류 뜨고 안 되다가 다시 접속하면 그때 됨").
        if self._agent_thread is not None:
            self._agent_thread = None
            self.worker_connect_button.setText("중앙에 접속(이 PC를 워커로)")
            self.worker_server_input.setEnabled(True)
            self.worker_token_input.setEnabled(True)
            self.worker_name_input.setEnabled(True)

    def _on_pc_clicked(self, item: QListWidgetItem) -> None:
        CONTROL_CONTEXT.set_target(item.data(Qt.ItemDataRole.UserRole))

    def _refresh(self) -> None:
        server = CONTROL_CONTEXT.server
        if server is None:
            return
        agents = sorted(server.snapshot()["agents"], key=lambda a: a["agentId"])
        selected = CONTROL_CONTEXT.selected_agent_id

        self.pc_list.clear()
        local_item = QListWidgetItem("이 PC (중앙)")
        local_item.setData(Qt.ItemDataRole.UserRole, None)
        self.pc_list.addItem(local_item)
        selected_row = 0
        selected_log: list[str] = []
        for row, agent in enumerate(agents, start=1):
            dot = "🟢" if agent["online"] else "🔴"
            item = QListWidgetItem(f"{dot} {agent['agentId']} - {agent['progress'] or '대기'}")
            item.setData(Qt.ItemDataRole.UserRole, agent["agentId"])
            self.pc_list.addItem(item)
            if agent["agentId"] == selected:
                selected_row = row
                selected_log = agent["logTail"]
        self.pc_list.setCurrentRow(selected_row)

        if selected is not None:
            self.log_view.setPlainText("\n".join(selected_log))
            scrollbar = self.log_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())


class UpdateBanner(QWidget):
    """업데이트 알림 배너 — 새 버전 있을 때만 나타남(없으면 높이 0, 자리 안 차지).
    의존성 변경 없는 버전이면 "빠른 업데이트" 버튼으로 소스만 받아 덮어쓸 수 있음
    (updatecheck.apply_update - 전체 zip 재다운로드 없이 재시작만 하면 됨)."""

    apply_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet("background-color: #fff3cd; color: #664d03;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel()
        self.label.setOpenExternalLinks(True)
        self.apply_button = QPushButton("빠른 업데이트 적용")
        self.apply_button.hide()
        self.apply_button.clicked.connect(self.apply_requested.emit)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.apply_button)
        self.hide()

    def show_message(self, html: str, fast: bool = False) -> None:
        self.label.setText(html)
        self.apply_button.setVisible(fast)
        self.show()


def main() -> int:
    app = QApplication(sys.argv)
    window = QMainWindow()
    version_label = appversion.BUILD_VERSION or "dev"
    window.setWindowTitle(f"Training Data Extractor (PySide6 pilot) - {version_label}")

    tabs = QTabWidget()
    tabs.addTab(LabelDbTab(), "1. 라벨 DB")
    tabs.addTab(SourceVerifyTab(), "2. 원본 검증")
    tabs.addTab(CenterTileTab(), "2.2 중앙 크롭(보정용)")
    tabs.addTab(TrainingTileTab(), "3. 학습 타일")
    tabs.addTab(YoloOrganizeTab(), "4. YOLO 정렬")
    tabs.addTab(TrainingTab(), "5. 학습")
    tabs.addTab(InferenceTab(), "6. 원본 추론")
    tabs.addTab(InferenceTestTab(), "6-1. 원본 추론 테스트(TensorRT)")
    tabs.addTab(ReviewTab(), "7. 후보 검수")
    tabs.addTab(CompareTab(), "8. 매칭/선별")
    tabs.addTab(LabelSyncTab(), "9. TXT 보정 반영")

    control_panel = ControlPanel()
    body_split = QSplitter(Qt.Orientation.Horizontal)
    body_split.addWidget(control_panel)
    body_split.addWidget(tabs)
    body_split.setStretchFactor(0, 0)
    body_split.setStretchFactor(1, 1)
    body_split.setSizes([260, 900])

    banner = UpdateBanner()
    central = QWidget()
    central_layout = QVBoxLayout(central)
    central_layout.setContentsMargins(0, 0, 0, 0)
    central_layout.addWidget(banner)
    central_layout.addWidget(body_split, stretch=1)
    window.setCentralWidget(central)
    window.resize(1080, 620)
    window.show()

    def _on_update_checked(info: Optional[dict]) -> None:
        if info:
            banner.show_message(info["message"], fast=info["fast"])

    def _on_apply_update() -> None:
        banner.apply_button.setEnabled(False)
        apply_worker = BackgroundCallWorker(updatecheck.apply_update)
        apply_worker.finished_ok.connect(
            lambda _ok: banner.show_message("업데이트 적용 완료. 앱을 재시작하세요."))
        apply_worker.finished_error.connect(
            lambda exc: banner.show_message(f"업데이트 적용 실패: {exc}"))
        window._apply_update_worker = apply_worker  # QThread가 GC되지 않게 참조 유지
        apply_worker.start()

    banner.apply_requested.connect(_on_apply_update)

    update_worker = BackgroundCallWorker(updatecheck.check_for_update)
    update_worker.finished_ok.connect(_on_update_checked)
    window._update_worker = update_worker  # QThread가 GC되지 않게 참조 유지
    update_worker.start()

    return app.exec()


if __name__ == "__main__":
    if "--agent" in sys.argv:
        # 워커 PC용 헤드리스 모드 (10번 탭에서 띄운 중앙 서버로 접속). GUI/QApplication 없음.
        sys.argv.remove("--agent")
        import agent
        agent.main()
    else:
        sys.exit(main())
