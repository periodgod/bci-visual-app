"""Console design constants for the visual BCI car workflow.

This module keeps the UI page aligned with the local
`bci-control-console-design` skill: visible workflow states, separated
train/test data splits, and explicit decoder readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


MODE_IDLE = "idle"
MODE_TRAIN = "train"
MODE_TEST = "test"
MODE_ONLINE = "online"
MODE_VALUES = [MODE_IDLE, MODE_TRAIN, MODE_TEST, MODE_ONLINE]
MODE_LABELS = ["空闲（不刺激）", "采集训练数据", "测试解码准确率", "在线控制小车"]
MODE_HINTS = {
    MODE_IDLE: "不播放刺激，不采集样本，可先检查参数和小车连接。",
    MODE_TRAIN: "按提示采集并保存 EEG 样本，用于校准 TDCA 等需要训练的解码器。",
    MODE_TEST: "使用当前解码器识别新样本，保存到测试集并统计准确率。",
    MODE_ONLINE: "连续实时解码，满足投票条件后可输出小车控制命令。",
}

DATA_SPLIT_TRAIN = "train"
DATA_SPLIT_TEST = "test"
STIMULUS_PROFILE = "spectrum_synthesis_harmonic_luminance"

DEFAULT_STIM_FREQS = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.8]


@dataclass(frozen=True)
class CarCommandSpec:
    label: str
    block: int
    description: str


DEFAULT_COMMANDS: Tuple[CarCommandSpec, ...] = (
    CarCommandSpec("↖前进左", 0, "斜向前左"),
    CarCommandSpec("↑前进", 1, "直行前进"),
    CarCommandSpec("↗前进右", 2, "斜向前右"),
    CarCommandSpec("←左转", 3, "原地左转"),
    CarCommandSpec("■停止", 4, "紧急停止"),
    CarCommandSpec("→右转", 5, "原地右转"),
    CarCommandSpec("↙后退左", 6, "斜向后左"),
    CarCommandSpec("↓后退", 7, "直行后退"),
    CarCommandSpec("↘后退右", 8, "斜向后右"),
)


def default_command_labels() -> str:
    return ",".join(item.label for item in DEFAULT_COMMANDS)
