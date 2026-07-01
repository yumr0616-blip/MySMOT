"""冻结 Tracker:Protocol 定义 + Stage-0 占位实现。

对应 §4:冻结、不 finetune。推荐的真实实现是 detector + SAM2(与
TF-SMOT 的基座保持一致,便于控制跟踪这个变量、公平对比);真实的
detector+SAM2 需要 GPU 和模型权重,超出本脚手架范围,这里先用
StubTracker 占位,只是把提前准备好的轨迹原样返回。
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from smot.types import Trajectory


class VideoHandle:
    """一个"视频引用"的最简占位对象。本脚手架里从不真正解码视频帧,
    只是存一下路径/帧数/帧率这些元信息,好让 Pipeline 的调用签名
    在未来接入真实视频解码时不需要修改。
    """

    def __init__(self, path: str, num_frames: int, fps: float = 1.0):
        self.path = path
        self.num_frames = num_frames
        self.fps = fps


@runtime_checkable
class Tracker(Protocol):
    """冻结,不 finetune。"""

    def track(self, video: VideoHandle) -> list[Trajectory]: ...


class StubTracker:
    """冻结(占位实现)。构造时传入一份写死/预先算好的轨迹列表,
    track() 调用时原样返回,不做任何真实的检测或跟踪计算。
    未来真实的 detector+SAM2 tracker 会实现同一个 Tracker Protocol。
    """

    def __init__(self, canned_trajectories: Optional[list[Trajectory]] = None):
        self._canned_trajectories = canned_trajectories or []

    def track(self, video: VideoHandle) -> list[Trajectory]:
        return list(self._canned_trajectories)
