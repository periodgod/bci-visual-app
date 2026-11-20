"""
脑控小车展示页
==============
当前系统只保留 3x3 SSVEP 小车控制流程：

1. 采集训练数据：保存到 saveCarData/<subject>/train
2. 测试解码准确率：从 train 构建解码器，测试样本保存到 test
3. 在线控制小车：从 train 构建解码器，实时投票输出小车命令

刺激由 ssvep_grid_canvas.py 生成，画布频率、训练数据和解码器共享同一
sti_freqs 配置，避免刺激频率与模型参考频率不一致。
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QShortcut,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.io import loadmat, savemat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config as cfg_manager
from interface.car_interface.acquisition import preprocess_model_input
from interface.car_interface.console_design import (
    DATA_SPLIT_TEST,
    DATA_SPLIT_TRAIN,
    DEFAULT_COMMANDS as DEFAULT_CAR_COMMANDS,
    DEFAULT_STIM_FREQS,
    MODE_IDLE,
    MODE_HINTS,
    MODE_LABELS,
    MODE_ONLINE,
    MODE_TEST,
    MODE_TRAIN,
    MODE_VALUES,
    STIMULUS_PROFILE,
    default_command_labels,
)
from interface.car_interface.ssvep_grid_canvas import GridConfig, SSVEPGridCanvas
from interface.car_interface.training_framework import (
    extract_int,
    extract_target_id,
    extract_training_sample,
)
from models.FBCCA import FBCCA
from models.TDCA import TDCA


class RobotClient:
    def __init__(self, ip="10.186.179.92", port=65432):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(1.5)
        self.addr = (ip, port)
        self.connected = False

    def connect(self):
        try:
            self.sock.connect(self.addr)
            self.connected = True
            return True
        except Exception:
            return False

    def send(self, cmd):
        if not self.connected:
            return
        try:
            self.sock.send(cmd.encode("utf-8"))
        except Exception:
            self.connected = False

    def set_motor(self, v1, v2, v3, v4):
        self.send(f"{int(v1)},{int(v2)},{int(v3)},{int(v4)}")

    def move(self, cmd_idx: int):
        if cmd_idx == 4:
            self.set_motor(0, 0, 0, 0)
            return

        motions = {
            0: (30, 60, 60, 30),
            1: (50, 50, 50, 50),
            2: (60, 30, 30, 60),
            3: (-40, 40, -40, 40),
            5: (40, -40, 40, -40),
            6: (-30, -60, -60, -30),
            7: (-50, -50, -50, -50),
            8: (-60, -30, -30, -60),
        }
        if cmd_idx in motions:
            v = motions[cmd_idx]
            self.set_motor(v[0], v[1], v[2], v[3])
            time.sleep(1.2)
        self.set_motor(0, 0, 0, 0)

    def close(self):
        try:
            self.sock.close()
        finally:
            self.connected = False


class FullscreenStimulusWindow(QWidget):
    def __init__(self, frequencies, phases=None):
        super().__init__(None)
        self.setWindowTitle("SSVEP 全屏刺激")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background:#000;")
        self.canvas = SSVEPGridCanvas(target_count=9, frequencies=frequencies, phases=phases)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence("F11"), self, activated=self.close)

    def show_stimulus_fullscreen(self):
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.canvas.setFocus()
        QTimer.singleShot(0, self.canvas.update)


_STYLE = """
QWidget#carControlWindow { background: #07090C; }
QLabel { color: #E8EDF5; font-size: 12px; }
QFrame#topBar {
    background: #0D1117;
    border: 1px solid #243041;
    border-radius: 6px;
}
QGroupBox {
    border: 1px solid #263244;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    background: #0B0F14;
    color: #D8E2F0;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #111820;
    border: 1px solid #334155;
    border-radius: 5px;
    color: #E8EDF5;
    min-height: 26px;
    padding: 3px 7px;
}
QComboBox QAbstractItemView {
    background: #0F172A;
    color: #F8FAFC;
    border: 1px solid #38BDF8;
    selection-background-color: #0EA5E9;
    selection-color: #FFFFFF;
    outline: 0;
}
QPushButton {
    color: #F8FAFC;
    border: 1px solid #3B82F6;
    border-radius: 5px;
    padding: 6px 10px;
    font-weight: 600;
    background: #1D4ED8;
}
QPushButton:hover { background: #2563EB; }
QPushButton:disabled {
    color: #64748B;
    background: #1E293B;
    border-color: #334155;
}
QProgressBar {
    border: 1px solid #334155;
    border-radius: 4px;
    background: #111820;
    color: #E8EDF5;
    text-align: center;
    min-height: 18px;
}
QProgressBar::chunk { background: #22C55E; border-radius: 3px; }
QCheckBox { color: #E8EDF5; }
QMessageBox {
    background: #F8FAFC;
}
QMessageBox QLabel {
    color: #111827;
    font-size: 13px;
}
QMessageBox QPushButton {
    color: #FFFFFF;
    background: #2563EB;
    border: 1px solid #1D4ED8;
    border-radius: 5px;
    min-width: 68px;
    min-height: 28px;
    padding: 4px 10px;
}
QMessageBox QPushButton:hover { background: #1D4ED8; }
"""


class CarControlWindowV2(QWidget):
    DEFAULT_COMMANDS = DEFAULT_CAR_COMMANDS
    MODES = MODE_VALUES

    def __init__(self, objectName="carControlWindow"):
        super().__init__()
        self.setObjectName(objectName)
        self.setStyleSheet(_STYLE)

        self.subject = getattr(cfg_manager, "subjectName", "TestSubject") or "TestSubject"
        self.sample_rate = int(getattr(cfg_manager, "lsl_target_sample_rate", 250) or 250)
        self.sti_freqs = [float(x) for x in getattr(cfg_manager, "sti_lst", [])[:9]]
        if len(self.sti_freqs) < 9:
            self.sti_freqs = list(DEFAULT_STIM_FREQS)

        self.mode = MODE_IDLE
        self.phase = MODE_IDLE
        self._trial_plan: List[int] = []
        self._trial_idx = 0
        self._current_block = -1
        self._done = 0
        self._correct = 0
        self._last_scores = None

        self._decoder = None
        self._decoder_name = "FBCCA"
        self._decision_buf = deque(maxlen=3)

        self._robot = RobotClient()
        self._robot_enabled = False

        self._eeg_lock = threading.RLock()
        self._eeg_chunks = []
        self._eeg_points = 0
        self._stim_start_mono = 0.0

        self._phase_timer = QTimer(self)
        self._phase_timer.setTimerType(Qt.PreciseTimer)
        self._phase_timer.timeout.connect(self._phase_tick)
        self._phase_t0 = 0.0
        self._phase_dur = 0.0

        self.fullscreen: Optional[FullscreenStimulusWindow] = None
        self._build_ui()
        self._build_shortcuts()

        self._clock = QTimer(self)
        self._clock.timeout.connect(self._refresh_status)
        self._clock.start(500)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QFrame()
        top.setObjectName("topBar")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(10, 8, 10, 8)
        title = self._label("SSVEP 脑控小车", bold=True, size=16)
        top_lay.addWidget(title)
        self.flow_lbl = self._label("流程: 空闲", color="#93A4B8")
        top_lay.addWidget(self.flow_lbl)
        top_lay.addStretch(1)
        self.fullscreen_btn = QPushButton("全屏刺激 F11")
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        top_lay.addWidget(self.fullscreen_btn)
        root.addWidget(top)

        self.canvas = SSVEPGridCanvas(target_count=9, frequencies=self.sti_freqs)
        self.canvas.setMinimumSize(720, 420)
        root.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(self._build_flow_panel(), 2)
        controls.addWidget(self._build_decode_panel(), 2)
        controls.addWidget(self._build_car_panel(), 1)
        root.addLayout(controls)

    def _build_flow_panel(self):
        box = QGroupBox("运行流程")
        lay = QGridLayout(box)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(6)

        lay.addWidget(self._label("模式"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODE_LABELS)
        self.mode_combo.setMinimumWidth(150)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        lay.addWidget(self.mode_combo, 0, 1)

        lay.addWidget(self._label("每类次数"), 0, 2)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 80)
        self.count_spin.setValue(10)
        lay.addWidget(self.count_spin, 0, 3)

        lay.addWidget(self._label("提示秒"), 1, 0)
        self.cue_sec = QDoubleSpinBox()
        self.cue_sec.setRange(0.2, 5.0)
        self.cue_sec.setSingleStep(0.1)
        self.cue_sec.setValue(1.0)
        lay.addWidget(self.cue_sec, 1, 1)

        lay.addWidget(self._label("刺激秒"), 1, 2)
        self.win_combo = QComboBox()
        self.win_combo.addItems(["2s", "3s", "4s"])
        self.win_combo.setCurrentText("3s")
        lay.addWidget(self.win_combo, 1, 3)

        lay.addWidget(self._label("休息秒"), 2, 0)
        self.rest_sec = QDoubleSpinBox()
        self.rest_sec.setRange(0.2, 5.0)
        self.rest_sec.setSingleStep(0.1)
        self.rest_sec.setValue(0.8)
        lay.addWidget(self.rest_sec, 2, 1)

        lay.addWidget(self._label("命令标签"), 2, 2)
        self.label_edit = QLineEdit(default_command_labels())
        lay.addWidget(self.label_edit, 2, 3)

        self.mode_hint_lbl = self._label(MODE_HINTS[MODE_IDLE], color="#CBD5E1", size=12)
        self.mode_hint_lbl.setWordWrap(True)
        lay.addWidget(self.mode_hint_lbl, 3, 0, 1, 4)

        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self._start)
        lay.addWidget(self.start_btn, 4, 0, 1, 2)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background:#B91C1C;border-color:#EF4444;")
        lay.addWidget(self.stop_btn, 4, 2, 1, 2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress, 5, 0, 1, 4)

        self.status_lbl = self._label("就绪", bold=True, color="#C7D2FE", size=13)
        lay.addWidget(self.status_lbl, 6, 0, 1, 4)
        return box

    def _build_decode_panel(self):
        box = QGroupBox("算法与结果")
        lay = QGridLayout(box)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(6)

        lay.addWidget(self._label("解码算法"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["FBCCA", "TDCA"])
        lay.addWidget(self.model_combo, 0, 1)

        self.train_model_btn = QPushButton("构建解码器")
        self.train_model_btn.clicked.connect(self._manual_build_decoder)
        lay.addWidget(self.train_model_btn, 0, 2)

        self.dataset_lbl = self._label("训练: 0 | 测试: 0", color="#93A4B8")
        lay.addWidget(self.dataset_lbl, 0, 3)

        self.pred_lbl = self._label("预测: -", bold=True, color="#7DD3FC", size=16)
        lay.addWidget(self.pred_lbl, 1, 0, 1, 2)

        self.acc_lbl = self._label("测试准确率: -", color="#A7F3D0")
        lay.addWidget(self.acc_lbl, 1, 2, 1, 2)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        lay.addWidget(self.conf_bar, 2, 0, 1, 4)

        self.conf_lbl = self._label("置信度: 0.000", color="#CBD5E1")
        lay.addWidget(self.conf_lbl, 3, 0, 1, 4)

        self.freq_lbl = self._label(self._freq_summary(), color="#93A4B8")
        lay.addWidget(self.freq_lbl, 4, 0, 1, 4)
        return box

    def _build_car_panel(self):
        box = QGroupBox("小车输出")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(self._label("IP"))
        self.car_ip = QLineEdit("10.186.179.92")
        row.addWidget(self.car_ip)
        lay.addLayout(row)

        self.car_btn = QPushButton("连接")
        self.car_btn.clicked.connect(self._toggle_car)
        lay.addWidget(self.car_btn)

        self.car_cb = QCheckBox("在线解码输出到小车")
        self.car_cb.toggled.connect(lambda value: setattr(self, "_robot_enabled", bool(value)))
        lay.addWidget(self.car_cb)

        self.net_lbl = self._label("小车: 未连接", color="#FCA5A5")
        lay.addWidget(self.net_lbl)
        lay.addStretch(1)
        return box

    def _build_shortcuts(self):
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence("Esc"), self, activated=self._stop)

    def _label(self, text, bold=False, color="#E8EDF5", size=12):
        label = QLabel(text)
        label.setStyleSheet(
            f"color:{color};font-size:{size}px;{'font-weight:700;' if bold else ''}"
        )
        return label

    def _freq_summary(self):
        pairs = [f"B{i + 1}:{freq:.1f}Hz" for i, freq in enumerate(self.sti_freqs)]
        return " | ".join(pairs)

    def _on_mode_changed(self, idx):
        self.mode = self.MODES[idx]
        self._reset_runtime(clear_decoder=False)
        self.start_btn.setEnabled(self.mode != MODE_IDLE)
        self.stop_btn.setEnabled(False)
        self.mode_hint_lbl.setText(MODE_HINTS.get(self.mode, ""))
        self.status_lbl.setText("就绪" if self.mode == MODE_IDLE else f"已选择: {self.mode_combo.currentText()}")
        self.flow_lbl.setText(f"流程: {self.mode_combo.currentText()}")

    def _start(self):
        if self.mode == MODE_TRAIN:
            self._start_trials(DATA_SPLIT_TRAIN)
        elif self.mode == MODE_TEST:
            if self._ensure_decoder():
                self._start_trials(DATA_SPLIT_TEST)
        elif self.mode == MODE_ONLINE:
            self._start_online()

    def _stop(self):
        self._phase_timer.stop()
        self._set_flicker(False)
        self._cue_target(-1)
        self.phase = MODE_IDLE
        self.mode = MODE_IDLE
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.blockSignals(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.mode_hint_lbl.setText(MODE_HINTS[MODE_IDLE])
        self.status_lbl.setText("已停止")
        self.flow_lbl.setText("流程: 空闲")

    def _reset_runtime(self, clear_decoder=True):
        self._phase_timer.stop()
        self._set_flicker(False)
        self._cue_target(-1)
        self._trial_plan = []
        self._trial_idx = 0
        self._current_block = -1
        self._done = 0
        self._correct = 0
        self._decision_buf.clear()
        self.progress.setValue(0)
        self.conf_bar.setValue(0)
        if clear_decoder:
            self._decoder = None
        self._clear_eeg()

    def _start_trials(self, split):
        blocks = self._car_blocks()
        if not blocks:
            QMessageBox.warning(self, "无法开始", "至少需要 1 个命令标签。")
            return

        self.phase = "cue"
        self.mode = split
        self._trial_plan = blocks * int(self.count_spin.value())
        np.random.RandomState(int(time.time())).shuffle(self._trial_plan)
        self._trial_idx = 0
        self._done = 0
        self._correct = 0
        self.progress.setRange(0, len(self._trial_plan))
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._next_trial()

    def _next_trial(self):
        if self._trial_idx >= len(self._trial_plan):
            if self.mode == DATA_SPLIT_TEST and self._done > 0:
                self.acc_lbl.setText(f"测试准确率: {self._correct}/{self._done} = {self._correct / self._done:.1%}")
            self._stop()
            return

        block = int(self._trial_plan[self._trial_idx])
        self._current_block = block
        label = self._label_for_block(block)
        self.phase = "cue"
        self._cue_target(block)
        self._set_flicker(True)
        self._phase_t0 = time.perf_counter()
        self._phase_dur = float(self.cue_sec.value())
        self._phase_timer.start(25)
        self.status_lbl.setText(f"准备: 注视 {label}  B{block + 1:02d}  {self.sti_freqs[block]:.1f}Hz")

    def _phase_tick(self):
        elapsed = time.perf_counter() - self._phase_t0
        if self.mode in (DATA_SPLIT_TRAIN, DATA_SPLIT_TEST):
            self._trial_phase_tick(elapsed)
        elif self.mode == MODE_ONLINE and elapsed >= self._phase_dur:
            self._decode_online_window()
            self._clear_eeg()
            self._phase_t0 = time.perf_counter()

    def _trial_phase_tick(self, elapsed):
        if self.phase == "cue" and elapsed >= self._phase_dur:
            self.phase = "stim"
            self._cue_target(-1)
            self._clear_eeg()
            self._stim_start_mono = time.perf_counter()
            self._phase_t0 = self._stim_start_mono
            self._phase_dur = self._window_sec()
            self.status_lbl.setText(f"采集中: {self._label_for_block(self._current_block)}")
            return

        if self.phase == "stim" and elapsed >= self._phase_dur:
            data = self._get_eeg()
            if self.mode == DATA_SPLIT_TRAIN:
                ok = self._save_trial(DATA_SPLIT_TRAIN, self._current_block, data)
                self.status_lbl.setText("已保存训练样本" if ok else "训练样本无效: EEG 点数不足")
            else:
                pred, scores, conf = self._decode_trial_data(data)
                ok = self._save_trial(DATA_SPLIT_TEST, self._current_block, data, pred, scores, conf)
                if pred is not None:
                    self._done += 1
                    if int(pred) == int(self._current_block):
                        self._correct += 1
                    self._show_prediction(pred, scores, conf, expected=self._current_block)
                self.status_lbl.setText("已保存测试样本" if ok else "测试样本无效: EEG 点数不足")

            if self.mode == DATA_SPLIT_TRAIN:
                self._done += 1
            self._trial_idx += 1
            self.progress.setValue(self._trial_idx)
            self.phase = "rest"
            self._set_flicker(False)
            self._phase_t0 = time.perf_counter()
            self._phase_dur = float(self.rest_sec.value())
            return

        if self.phase == "rest" and elapsed >= self._phase_dur:
            self._next_trial()

    def _start_online(self):
        if not self._ensure_decoder():
            return
        self.mode = MODE_ONLINE
        self.phase = MODE_ONLINE
        self._decision_buf.clear()
        self._correct = 0
        self._done = 0
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setRange(0, 0)
        self._cue_target(-1)
        self._set_flicker(True)
        self._clear_eeg()
        self._phase_t0 = time.perf_counter()
        self._phase_dur = self._window_sec()
        self._phase_timer.start(100)
        self.status_lbl.setText(f"在线控制中: {self._decoder_name}")

    def _manual_build_decoder(self):
        if self._ensure_decoder(show_success=True):
            self._refresh_dataset_counts()

    def _ensure_decoder(self, show_success=False):
        ok, message = self._build_decoder()
        if not ok:
            QMessageBox.warning(self, "解码器不可用", message)
            return False
        if show_success:
            QMessageBox.information(self, "解码器已就绪", message)
        self.status_lbl.setText(message)
        return True

    def _build_decoder(self) -> Tuple[bool, str]:
        self._decoder_name = self.model_combo.currentText().strip().upper()
        blocks = self._car_blocks()
        freqs = [self.sti_freqs[i] for i in blocks]
        phases = [float(self.canvas.phases[i]) for i in blocks]
        win_sec = self._window_sec()

        if self._decoder_name == "FBCCA":
            self._decoder = FBCCA(
                3, win_sec, freqs, Nh=8, sample_rate=self.sample_rate, phases=phases
            )
            return True, "FBCCA 已就绪: 无需训练，直接使用频率-相位参考信号"

        decoder = TDCA(3, win_sec, freqs, Nh=8, sample_rate=self.sample_rate)
        files = self._training_files()
        X, y = self._load_training_samples(files, decoder)
        if len(X) == 0:
            return False, "TDCA 需要训练样本: 请先执行“采集训练数据”"
        missing = [self._label_for_block(i) for i in blocks if int(np.sum(np.asarray(y) == i)) == 0]
        if missing:
            return False, "TDCA 缺少类别样本: " + ", ".join(missing)
        try:
            decoder.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        except Exception as exc:
            return False, f"TDCA 训练失败: {exc}"
        self._decoder = decoder
        return True, f"TDCA 已训练: {len(X)} 条训练样本"

    def _load_training_samples(self, files, decoder):
        required_points = int(getattr(decoder, "required_points", getattr(decoder, "T", 1)))
        current_freqs = np.asarray(getattr(decoder, "targets", []), dtype=float).reshape(-1)
        blocks = self._car_blocks()
        X, y = [], []
        for fp in files:
            try:
                mat = loadmat(fp)
                if not self._freqs_compatible(mat, current_freqs):
                    continue
                bid = extract_target_id(mat)
                if bid not in blocks:
                    continue
                file_sr = extract_int(mat.get("sample_rate_hz", -1), -1)
                if file_sr > 0 and int(file_sr) != self.sample_rate:
                    continue
                sample = extract_training_sample(mat, required_points, self.sample_rate)
                sample = preprocess_model_input(sample)
                if sample is None:
                    continue
                X.append(sample)
                y.append(int(bid))
            except Exception:
                continue
        return X, y

    def _decode_trial_data(self, data):
        if self._decoder is None:
            return None, None, 0.0
        data = self._window_from_data(data, self._decoder)
        if data is None:
            return None, None, 0.0
        try:
            pred, scores, conf = self._decoder.classify_with_scores(data)
            return int(pred), np.asarray(scores, dtype=float), float(conf)
        except Exception as exc:
            self.status_lbl.setText(f"解码失败: {exc}")
            return None, None, 0.0

    def _decode_online_window(self):
        pred, scores, conf = self._decode_trial_data(self._get_eeg())
        if pred is None:
            return
        self._decision_buf.append(int(pred))
        labels = self._labels()
        winner = int(pred)
        if len(self._decision_buf) >= 3:
            unique, counts = np.unique(list(self._decision_buf), return_counts=True)
            if int(np.max(counts)) >= 2:
                winner = int(unique[int(np.argmax(counts))])
                self._decision_buf.clear()
                if self._robot_enabled and self._robot.connected:
                    threading.Thread(target=self._robot.move, args=(winner,), daemon=True).start()
        self._show_prediction(winner, scores, conf)
        self.status_lbl.setText(f"在线解码: {labels[winner] if winner < len(labels) else winner}")

    def _show_prediction(self, pred, scores, conf, expected=None):
        label = self._label_for_block(pred)
        if expected is None:
            self.pred_lbl.setText(f"预测: {label}")
        else:
            mark = "正确" if int(pred) == int(expected) else f"错误 / 应为 {self._label_for_block(expected)}"
            self.pred_lbl.setText(f"预测: {label} ({mark})")
        self.conf_lbl.setText(f"置信度: {float(conf):.4f}")
        self.conf_bar.setValue(self._confidence_percent(scores, conf))
        if self.mode == DATA_SPLIT_TEST and self._done > 0:
            self.acc_lbl.setText(f"测试准确率: {self._correct}/{self._done} = {self._correct / self._done:.1%}")

    def _confidence_percent(self, scores, conf):
        if isinstance(scores, np.ndarray) and scores.size >= 2:
            scores = np.asarray(scores, dtype=float)
            top = float(np.max(np.abs(scores)))
            return int(np.clip(100.0 * abs(float(conf)) / (top + 1e-6), 0, 100))
        return int(np.clip(abs(float(conf)) * 100.0, 0, 100))

    def _window_from_data(self, data, decoder):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            return None
        required_points = int(getattr(decoder, "required_points", getattr(decoder, "T", 1)))
        if data.shape[-1] < required_points:
            return None
        return preprocess_model_input(data[:, -required_points:])

    def _save_trial(self, split, block_idx, data, pred=None, scores=None, conf=0.0):
        if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[-1] < 2:
            return False
        split_dir = self._split_dir(split)
        os.makedirs(split_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label = self._label_for_block(block_idx).replace(" ", "")
        path = os.path.join(split_dir, f"{ts}_B{block_idx + 1:02d}_{label}.mat")
        payload = {
            "eeg_data": np.asarray(data, dtype=float),
            "data": np.asarray(data, dtype=float),
            "split": split,
            "block_idx": int(block_idx),
            "target_id": int(block_idx),
            "label_idx": int(block_idx),
            "label_text": self._label_for_block(block_idx),
            "freq_hz": float(self.sti_freqs[block_idx]),
            "phase_rad": float(self.canvas.block_phase(block_idx)),
            "sample_rate_hz": int(self.sample_rate),
            "stim_freqs_hz": np.array(self.sti_freqs, dtype=float),
            "stim_phases_rad": np.array(self.canvas.phases, dtype=float),
            "stimulus_profile": STIMULUS_PROFILE,
            "window_sec": float(self._window_sec()),
            "timestamp": ts,
        }
        if pred is not None:
            payload.update({
                "pred_idx": int(pred),
                "pred_label_text": self._label_for_block(int(pred)),
                "correct": int(int(pred) == int(block_idx)),
                "confidence": float(conf),
            })
            if isinstance(scores, np.ndarray):
                payload["scores"] = np.asarray(scores, dtype=float)
        savemat(path, payload)
        return True

    def _training_files(self):
        train_files = sorted(glob(os.path.join(self._split_dir(DATA_SPLIT_TRAIN), "*.mat")))
        legacy_files = sorted(glob(os.path.join(self._subject_dir(), "*.mat")))
        return train_files + legacy_files

    def _split_dir(self, split):
        return os.path.join(self._subject_dir(), str(split))

    def _subject_dir(self):
        return os.path.join(PROJECT_ROOT, "saveCarData", self.subject)

    def _freqs_compatible(self, mat, current_freqs):
        try:
            saved = np.asarray(mat.get("stim_freqs_hz"), dtype=float).reshape(-1)
        except Exception:
            return True
        if saved.size < len(current_freqs):
            return False
        return bool(np.allclose(saved[:len(current_freqs)], current_freqs, atol=0.03, rtol=0.0))

    def _labels(self):
        text = self.label_edit.text().strip()
        labels = [x.strip() for x in text.split(",") if x.strip()]
        if not labels:
            labels = [item.label for item in self.DEFAULT_COMMANDS]
        return labels[:GridConfig.N_BLOCKS]

    def _car_blocks(self):
        return list(range(min(len(self._labels()), GridConfig.N_BLOCKS)))

    def _label_for_block(self, block_idx):
        labels = self._labels()
        idx = int(block_idx)
        if 0 <= idx < len(labels):
            return labels[idx]
        return f"B{idx + 1:02d}"

    def _window_sec(self):
        return float(self.win_combo.currentText().replace("s", ""))

    def _clear_eeg(self):
        with self._eeg_lock:
            self._eeg_chunks.clear()
            self._eeg_points = 0

    def push_eeg_data_threadsafe(self, data):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            return
        with self._eeg_lock:
            self._eeg_chunks.append(np.asarray(data, dtype=float))
            self._eeg_points += int(data.shape[-1])

    def _get_eeg(self):
        with self._eeg_lock:
            if not self._eeg_chunks:
                return np.array([])
            if len(self._eeg_chunks) == 1:
                return self._eeg_chunks[0]
            try:
                return np.concatenate(self._eeg_chunks, axis=-1)
            except Exception:
                return np.array([])

    def _toggle_fullscreen(self):
        if self.fullscreen is not None and self.fullscreen.isVisible():
            self.fullscreen.close()
            return
        self.fullscreen = FullscreenStimulusWindow(self.sti_freqs, self.canvas.phases)
        self.fullscreen.destroyed.connect(lambda _=None: setattr(self, "fullscreen", None))
        self._sync_fullscreen_from_embedded()
        self.fullscreen.show_stimulus_fullscreen()

    def _sync_fullscreen_from_embedded(self):
        if self.fullscreen is None:
            return
        self.fullscreen.canvas.set_frequencies(self.canvas.freqs)
        self.fullscreen.canvas.set_phases(self.canvas.phases)
        if self._current_block >= 0:
            self.fullscreen.canvas.cue_target(
                self._current_block, 0, self._label_for_block(self._current_block)
            )
        else:
            self.fullscreen.canvas.clear_cue()
        if self.canvas.is_flickering:
            self.fullscreen.canvas.start_flicker()
        else:
            self.fullscreen.canvas.stop_flicker()

    def _set_flicker(self, enabled):
        canvases = [self.canvas]
        if self.fullscreen is not None and self.fullscreen.isVisible():
            canvases.append(self.fullscreen.canvas)
        for canvas in canvases:
            if enabled:
                canvas.start_flicker()
            else:
                canvas.stop_flicker()

    def _cue_target(self, block_idx):
        canvases = [self.canvas]
        if self.fullscreen is not None and self.fullscreen.isVisible():
            canvases.append(self.fullscreen.canvas)
        for canvas in canvases:
            if int(block_idx) < 0:
                canvas.clear_cue()
            else:
                canvas.cue_target(int(block_idx), 0, self._label_for_block(int(block_idx)))

    def _toggle_car(self):
        if self._robot.connected:
            self._robot.close()
            self.car_btn.setText("连接")
            self.net_lbl.setText("小车: 未连接")
            self.net_lbl.setStyleSheet("color:#FCA5A5;")
            return
        self._robot = RobotClient(self.car_ip.text().strip(), 65432)
        if self._robot.connect():
            self.car_btn.setText("断开")
            self.net_lbl.setText("小车: 已连接")
            self.net_lbl.setStyleSheet("color:#86EFAC;")
        else:
            self.net_lbl.setText("小车: 连接失败")
            self.net_lbl.setStyleSheet("color:#FCA5A5;")

    def _refresh_status(self):
        self._refresh_dataset_counts()
        with self._eeg_lock:
            pts = self._eeg_points
        if self.mode != MODE_IDLE:
            self.flow_lbl.setText(f"流程: {self.mode_combo.currentText()} | EEG {pts} 点")

    def _refresh_dataset_counts(self):
        train_n = len(glob(os.path.join(self._split_dir(DATA_SPLIT_TRAIN), "*.mat"))) + len(glob(os.path.join(self._subject_dir(), "*.mat")))
        test_n = len(glob(os.path.join(self._split_dir(DATA_SPLIT_TEST), "*.mat")))
        self.dataset_lbl.setText(f"训练: {train_n} | 测试: {test_n}")
