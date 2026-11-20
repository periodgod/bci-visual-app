"""
SSVEP 频率-相位编码刺激画布 — Qt 原生渲染
=============================================
9 个刺激块 (3×3 矩阵)，频率-相位联合编码。
每个块对应一条小车命令。

渲染方式：频谱合成亮度序列。
先按基频、谐波和相位合成目标频谱，再映射到 0~1 亮度；
不是简单在界面线程里直接翻转黑白块。
帧同步通过 QTimer(PreciseTimer) + 屏幕刷新率驱动。
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QFontMetrics,
    QPainterPath,
)
from PyQt5.QtWidgets import QWidget, QApplication


# ═══════════════════════════════════════════════════════════════════
#  配置常量
# ═══════════════════════════════════════════════════════════════════

class GridConfig:
    """SSVEP 网格刺激的全部可配置参数。"""

    # 网格 (3×3 = 9 块: 中心停止 + 八方向移动)
    N_ROWS = 3
    N_COLS = 3
    N_BLOCKS = N_ROWS * N_COLS  # 9
    N_FIXATIONS = 1              # 每块注视点数 (9 块已足够区分)
    N_TARGETS = N_BLOCKS * N_FIXATIONS  # 9

    # 频率 (Hz) — 9 个，间隔 ~1.0 Hz 确保充分区分
    FREQ_START = 8.0
    FREQ_END = 15.8
    FREQ_STEP = 0.2  # 实际频率由 config 指定，此为参考步长

    # 相位 (× π)
    PHASE_STEP = 0.35
    N_PHASES = 3  # 3×3 网格用 3 种相位即可 (棋盘格)

    # 注视点方向 (相对于块中心的归一化方向) — 仅中心
    FIXATION_DIRECTIONS = (
        (0, 0),     # 0: 中心（唯一注视点）
    )

    # 颜色 (0-255)
    BG_COLOR = (0, 0, 0)               # 纯黑背景
    BLOCK_BORDER = (60, 60, 60)        # 块边框
    FIXATION_COLOR = (220, 60, 60)     # 注视十字 (红)
    CUE_COLOR = (60, 220, 60)          # 提示高亮 (绿)
    FLICKER_HIGH = (255, 255, 255)     # 闪烁峰值 (白)

    # 频谱合成参数：基频 + 谐波。权重越高，视觉刺激越接近该频谱分量。
    SPECTRUM_HARMONICS = (1, 2, 3)
    SPECTRUM_WEIGHTS = (1.0, 0.35, 0.18)


# 在类外初始化 PHASE_VALUES（避免 Python 类体内列表推导式作用域问题）
GridConfig.PHASE_VALUES = [
    k * GridConfig.PHASE_STEP * math.pi for k in range(GridConfig.N_PHASES)
]


def _detect_screen_refresh_rate(default_hz: float = 60.0) -> float:
    """检测主屏幕刷新率。"""
    try:
        screens = QApplication.screens()
        if not screens:
            return default_hz
        # 偏好 60 Hz 附近的屏幕
        for scr in screens:
            r = float(scr.refreshRate())
            if 55.0 <= r <= 75.0 and 30.0 <= r <= 360.0:
                return r
        r = float(screens[0].refreshRate())
        if 30.0 <= r <= 360.0:
            return r
    except Exception:
        pass
    return default_hz


# ═══════════════════════════════════════════════════════════════════
#  闪烁块数据对象
# ═══════════════════════════════════════════════════════════════════

class FlickerBlock:
    """单个 SSVEP 闪烁块的状态（非 Qt widget，纯数据+绘制）。"""

    __slots__ = (
        "block_idx", "row", "col", "freq_hz", "phase_rad",
        "rect", "current_lum", "fixation_positions", "cued_fp_idx",
    )

    def __init__(self, block_idx: int, row: int, col: int,
                 freq_hz: float, phase_rad: float):
        self.block_idx = block_idx
        self.row = row
        self.col = col
        self.freq_hz = freq_hz
        self.phase_rad = phase_rad
        self.rect = QRectF()
        self.current_lum = 0.5  # 初始 50% 灰
        self.fixation_positions: List[Tuple[float, float]] = []
        self.cued_fp_idx: int = -1  # -1 表示无提示

    @property
    def target_ids(self) -> range:
        """该块对应的 5 个全局目标 ID 范围。"""
        start = self.block_idx * GridConfig.N_FIXATIONS
        return range(start, start + GridConfig.N_FIXATIONS)


# ═══════════════════════════════════════════════════════════════════
#  SSVEP 网格画布 (QWidget)
# ═══════════════════════════════════════════════════════════════════

class SSVEPGridCanvas(QWidget):
    """SSVEP 频率-相位编码刺激画布。

    Parameters
    ----------
    parent : QWidget
    target_count : int
        当前小车展示页使用 9 个目标。
    frequencies : Sequence[float] | None
        每个块的实际刺激频率。传入后画布与解码器使用同一频率表。
    """

    # 回调信号可以用 Qt Signal，这里用简单回调保持解耦
    on_stim_started = None   # callable()
    on_stim_stopped = None   # callable()

    def __init__(
        self,
        parent=None,
        target_count: int = 9,
        frequencies: Optional[Sequence[float]] = None,
        phases: Optional[Sequence[float]] = None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        # 确保有最小尺寸，防止布局时被压缩到 0
        self.setMinimumSize(200, 150)

        self._target_count = int(target_count)
        self._show_fixation_crosses = (self._target_count >= 200)

        # ── 频率 & 相位 ──
        cfg = GridConfig
        self._freqs: List[float] = self._normalize_freqs(frequencies)
        self._phases: List[float] = self._normalize_phases(phases)

        # ── 创建闪烁块 ──
        self._blocks: List[FlickerBlock] = []
        for i in range(cfg.N_BLOCKS):
            row, col = divmod(i, cfg.N_COLS)
            blk = FlickerBlock(i, row, col, self._freqs[i], self._phases[i])
            self._blocks.append(blk)

        # ── 渲染状态 ──
        self._refresh_hz = _detect_screen_refresh_rate()
        self._frame_interval_ms = max(1, int(1000.0 / self._refresh_hz))
        self._is_flickering = False
        self._cue_text = ""
        self._t_start: float = 0.0
        self._frame_index: int = 0
        self._last_render_frame: int = -1
        self._layout_done = False

        # ── 计时器 ──
        self._stim_timer = QTimer(self)
        self._stim_timer.setTimerType(Qt.PreciseTimer)
        self._stim_timer.setInterval(self._frame_interval_ms)
        self._stim_timer.timeout.connect(self._on_timer_tick)

        # ── 高 DPI ──
        self._dpr = self.devicePixelRatioF()

    # ── 属性 ────────────────────────────────────────────────────

    @property
    def target_count(self) -> int:
        return self._target_count

    @property
    def is_flickering(self) -> bool:
        return self._is_flickering

    @property
    def refresh_hz(self) -> float:
        return self._refresh_hz

    @property
    def freqs(self) -> np.ndarray:
        return np.asarray(self._freqs, dtype=float)

    @property
    def phases(self) -> np.ndarray:
        return np.asarray(self._phases, dtype=float)

    def set_frequencies(self, frequencies: Sequence[float]):
        """更新刺激频率，并同步到所有块。"""
        self._freqs = self._normalize_freqs(frequencies)
        for blk, freq in zip(self._blocks, self._freqs):
            blk.freq_hz = float(freq)
        self.update()

    def set_phases(self, phases: Sequence[float]):
        """更新相位，并同步到所有块。"""
        self._phases = self._normalize_phases(phases)
        for blk, phase in zip(self._blocks, self._phases):
            blk.phase_rad = float(phase)
        self.update()

    def block_freq(self, block_idx: int) -> float:
        return self._freqs[block_idx]

    def block_phase(self, block_idx: int) -> float:
        return self._phases[block_idx]

    def global_target_id(self, block_idx: int, fp_idx: int) -> int:
        return block_idx * GridConfig.N_FIXATIONS + fp_idx

    def _normalize_freqs(self, frequencies: Optional[Sequence[float]]) -> List[float]:
        cfg = GridConfig
        default_freqs = [
            cfg.FREQ_START + i * cfg.FREQ_STEP
            for i in range(cfg.N_BLOCKS)
        ]
        if frequencies is None:
            return [float(x) for x in default_freqs]
        values = [float(x) for x in frequencies]
        if len(values) < cfg.N_BLOCKS:
            values.extend(default_freqs[len(values):])
        return values[:cfg.N_BLOCKS]

    def _normalize_phases(self, phases: Optional[Sequence[float]]) -> List[float]:
        cfg = GridConfig
        default_phases = []
        for row in range(cfg.N_ROWS):
            for col in range(cfg.N_COLS):
                phase_idx = (row + col) % cfg.N_PHASES
                default_phases.append(cfg.PHASE_VALUES[phase_idx])
        if phases is None:
            return [float(x) for x in default_phases]
        values = [float(x) for x in phases]
        if len(values) < cfg.N_BLOCKS:
            values.extend(default_phases[len(values):])
        return values[:cfg.N_BLOCKS]

    # ── 布局计算 ───────────────────────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        if not self._layout_done:
            self._recalc_layout()
            self._layout_done = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recalc_layout()

    def _recalc_layout(self):
        """根据窗口大小重新计算所有块和注视点的像素位置。"""
        cfg = GridConfig
        w = max(1, self.width())
        h = max(1, self.height())

        if w <= 1 and h <= 1:
            return  # 尚未获得有效尺寸

        # 块大小自适应：留 2% 边距
        margin = max(4, int(min(w, h) * 0.02))
        avail_w = w - 2 * margin
        avail_h = h - 2 * margin

        # 按行列数均分，保留间隙
        gap_px = max(2, int(min(avail_w / (cfg.N_COLS * 6), avail_h / (cfg.N_ROWS * 6))))
        block_w = (avail_w - gap_px * (cfg.N_COLS - 1)) / cfg.N_COLS
        block_h = (avail_h - gap_px * (cfg.N_ROWS - 1)) / cfg.N_ROWS
        block_size = min(block_w, block_h)

        if block_size <= 0:
            return  # 布局空间不足

        total_w = cfg.N_COLS * block_size + (cfg.N_COLS - 1) * gap_px
        total_h = cfg.N_ROWS * block_size + (cfg.N_ROWS - 1) * gap_px
        start_x = (w - total_w) / 2.0
        start_y = (h - total_h) / 2.0

        for blk in self._blocks:
            x = start_x + blk.col * (block_size + gap_px)
            y = start_y + blk.row * (block_size + gap_px)
            blk.rect = QRectF(x, y, block_size, block_size)

            # 注视点位置始终计算：小车的 9 目标界面虽然不显示普通十字，
            # 但训练/测试时仍需要在被提示目标上显示醒目的引导标记。
            cx = x + block_size / 2.0
            cy = y + block_size / 2.0
            offset = block_size * 0.22  # 注视点偏移 ~22% 块半边长
            blk.fixation_positions = [
                (cx + dx * offset, cy + dy * offset)
                for dx, dy in cfg.FIXATION_DIRECTIONS
            ]

        self._layout_done = True

    # ── 闪烁控制 ───────────────────────────────────────────────

    def start_flicker(self):
        """开始频谱合成刺激。"""
        if self._is_flickering:
            return
        self._is_flickering = True
        self._t_start = time.perf_counter()
        self._frame_index = 0
        self._last_render_frame = -1
        self._stim_timer.start()
        if self.on_stim_started:
            self.on_stim_started()

    def stop_flicker(self):
        """停止刺激，恢复默认灰色（块仍可见）。"""
        self._is_flickering = False
        self._stim_timer.stop()
        for blk in self._blocks:
            blk.current_lum = 0.5  # 恢复灰色可见状态
        self.update()
        if self.on_stim_stopped:
            self.on_stim_stopped()

    def reset(self):
        """停止并清空提示。"""
        self.stop_flicker()
        for blk in self._blocks:
            blk.cued_fp_idx = -1
        self._cue_text = ""

    # ── 提示控制 ───────────────────────────────────────────────

    @property
    def cue_text(self) -> str:
        return self._cue_text

    def cue_target(self, block_idx: int, fp_idx: int = 0, text: str = ""):
        """高亮指定目标的注视点（绿色）。传 -1 取消。"""
        for blk in self._blocks:
            blk.cued_fp_idx = -1
        if 0 <= block_idx < GridConfig.N_BLOCKS:
            self._blocks[block_idx].cued_fp_idx = fp_idx
            self._cue_text = str(text).strip()
        else:
            self._cue_text = ""
        self.update()

    def clear_cue(self):
        self.cue_target(-1)

    # ── 帧更新 (计时器回调) ────────────────────────────────────

    def _on_timer_tick(self):
        if not self._is_flickering:
            return
        now = time.perf_counter()
        elapsed = now - self._t_start
        # 基于时间而非帧计数，避免累积漂移
        self._frame_index = int(elapsed * self._refresh_hz)

        if self._frame_index == self._last_render_frame:
            return
        self._last_render_frame = self._frame_index

        t_sec = self._frame_index / self._refresh_hz
        for blk in self._blocks:
            blk.current_lum = self._synthesize_luminance(
                blk.freq_hz,
                blk.phase_rad,
                t_sec,
            )
        self.update()

    def _synthesize_luminance(self, freq_hz: float, phase_rad: float, t_sec: float) -> float:
        """由目标频谱合成单帧亮度。

        基频携带主要 SSVEP 标签；谐波用于增强可识别频谱。相位按谐波阶数同步，
        保持与 FBCCA/TDCA 参考信号一致。
        """
        weights = GridConfig.SPECTRUM_WEIGHTS
        harmonics = GridConfig.SPECTRUM_HARMONICS
        total_weight = max(1e-9, float(sum(abs(w) for w in weights)))
        value = 0.0
        for harmonic, weight in zip(harmonics, weights):
            h = float(harmonic)
            value += float(weight) * math.sin(
                2.0 * math.pi * h * float(freq_hz) * t_sec + h * float(phase_rad)
            )
        value /= total_weight
        return float(np.clip(0.5 + 0.5 * value, 0.0, 1.0))

    def _current_time_sec(self) -> float:
        if self._is_flickering:
            return self._frame_index / self._refresh_hz
        return 0.0

    # ── 绘制 ───────────────────────────────────────────────────

    def paintEvent(self, event):
        # 如果布局尚未完成（首次显示前的 race condition），强制执行一次
        if not self._layout_done and self.width() > 1 and self.height() > 1:
            self._recalc_layout()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 背景
        painter.fillRect(self.rect(), QColor(*GridConfig.BG_COLOR))

        cfg = GridConfig

        # ── 左侧独立提示框 ──
        # 训练、测试时把“该看哪里”放在闪烁矩阵外，避免提示被亮度变化淹没。
        if self._cue_text:
            valid_rects = [blk.rect for blk in self._blocks if not blk.rect.isEmpty()]
            left_edge = min((rect.left() for rect in valid_rects), default=float(self.width()))
            panel_margin = max(18.0, min(self.width(), self.height()) * 0.035)
            panel_w = min(330.0, max(150.0, left_edge - panel_margin * 2))
            panel_h = min(230.0, max(140.0, self.height() * 0.28))
            panel_x = panel_margin
            panel_y = (self.height() - panel_h) / 2.0
            panel = QRectF(panel_x, panel_y, panel_w, panel_h)

            painter.setPen(QPen(QColor(62, 220, 110), 3.0))
            painter.setBrush(QBrush(QColor(8, 28, 22, 235)))
            painter.drawRoundedRect(panel, 18, 18)

            title_font = QFont("Microsoft YaHei", max(12, int(panel_w * 0.075)))
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.setPen(QColor(142, 255, 180))
            painter.drawText(
                QRectF(panel_x + 12, panel_y + 18, panel_w - 24, panel_h * 0.30),
                Qt.AlignHCenter | Qt.AlignVCenter,
                "请注视",
            )

            cue_font = QFont("Microsoft YaHei", max(16, int(panel_w * 0.13)))
            cue_font.setBold(True)
            painter.setFont(cue_font)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                QRectF(panel_x + 10, panel_y + panel_h * 0.35, panel_w - 20, panel_h * 0.48),
                Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextWordWrap,
                self._cue_text,
            )

        for blk in self._blocks:
            rect = blk.rect
            if rect.isEmpty():
                continue

            # ── 闪烁块主体 ──
            lum = blk.current_lum
            gray = int(lum * 255)
            block_color = QColor(gray, gray, gray)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(block_color))
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

            # ── 边框 ──
            border = QColor(*cfg.BLOCK_BORDER)
            pen = QPen(border, max(1, int(rect.width() * 0.008)))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

            # ── 注视十字 / 当前目标提示 ──
            # 小车界面只有 9 个目标，普通注视十字默认隐藏，但当前提示必须始终可见。
            if blk.fixation_positions:
                for fp_idx, (fx, fy) in enumerate(blk.fixation_positions):
                    is_cued = (blk.cued_fp_idx == fp_idx)
                    if not self._show_fixation_crosses and not is_cued:
                        continue

                    if is_cued:
                        cross_len = max(24.0, min(rect.width(), rect.height()) * 0.34)
                        cross_w = max(5.0, cross_len * 0.16)
                        ring_size = max(38.0, cross_len * 1.45)

                        # 黑色外描边保证提示在明暗闪烁背景上都清晰。
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(QColor(0, 0, 0), max(7.0, cross_w * 1.8)))
                        painter.drawEllipse(QRectF(
                            fx - ring_size / 2, fy - ring_size / 2,
                            ring_size, ring_size,
                        ))
                        painter.setPen(QPen(QColor(*cfg.CUE_COLOR), max(4.0, cross_w)))
                        painter.drawEllipse(QRectF(
                            fx - ring_size / 2, fy - ring_size / 2,
                            ring_size, ring_size,
                        ))
                    else:
                        cross_len = max(2.0, rect.width() * 0.06)
                        cross_w = max(1.0, cross_len * 0.22)

                    cross_color = QColor(*cfg.CUE_COLOR) if is_cued else QColor(*cfg.FIXATION_COLOR)
                    painter.setPen(QPen(QColor(0, 0, 0), max(2.0, cross_w * 1.8)))
                    painter.drawLine(QPointF(fx - cross_len / 2, fy), QPointF(fx + cross_len / 2, fy))
                    painter.drawLine(QPointF(fx, fy - cross_len / 2), QPointF(fx, fy + cross_len / 2))
                    painter.setPen(QPen(cross_color, cross_w))
                    painter.drawLine(QPointF(fx - cross_len / 2, fy), QPointF(fx + cross_len / 2, fy))
                    painter.drawLine(QPointF(fx, fy - cross_len / 2), QPointF(fx, fy + cross_len / 2))

        painter.end()

    # ── 信息查询 ───────────────────────────────────────────────

    def target_info(self, target_id: int) -> dict:
        """返回目标 ID 对应的完整信息。"""
        block_idx = target_id // GridConfig.N_FIXATIONS
        fp_idx = target_id % GridConfig.N_FIXATIONS
        if 0 <= block_idx < GridConfig.N_BLOCKS:
            blk = self._blocks[block_idx]
            return {
                "target_id": target_id,
                "block_idx": block_idx,
                "fp_idx": fp_idx,
                "freq_hz": blk.freq_hz,
                "phase_rad": blk.phase_rad,
                "row": blk.row,
                "col": blk.col,
            }
        return {}


# ═══════════════════════════════════════════════════════════════════
#  独立测试入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    win = QWidget()
    win.setWindowTitle("SSVEP Grid Canvas — Standalone Test")
    win.resize(1707, 960)
    win.setStyleSheet("background: #000;")

    from PyQt5.QtWidgets import QVBoxLayout
    layout = QVBoxLayout(win)
    layout.setContentsMargins(0, 0, 0, 0)

    canvas = SSVEPGridCanvas(target_count=9)
    layout.addWidget(canvas)

    # 按空格开始/暂停
    def on_key(e):
        if e.key() == Qt.Key_Space:
            if canvas.is_flickering:
                canvas.stop_flicker()
                print("[STOP]")
            else:
                canvas.start_flicker()
                print("[START]")
        elif e.key() == Qt.Key_C:
            # 随机提示一个目标
            import random
            bid = random.randint(0, GridConfig.N_BLOCKS - 1)
            fid = random.randint(0, 4) if canvas._show_fixation_crosses else 0
            canvas.cue_target(bid, fid)
            print(f"[CUE] Block {bid+1}, FP {fid+1}")
        elif e.key() == Qt.Key_Escape:
            win.close()

    win.keyPressEvent = on_key
    win.setFocusPolicy(Qt.StrongFocus)

    win.show()
    sys.exit(app.exec_())
