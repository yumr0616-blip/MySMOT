"""Frozen Tracker: Protocol + Stage-0 stub.

Frozen, not finetuned, per §4. Recommended real implementation is
detector + SAM2 (matches the TF-SMOT baseline for a controlled comparison);
that requires GPU + real models and is out of scope for this scaffold.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from smot.types import Trajectory


class VideoHandle:
    """Minimal stand-in for a decoded video reference. Never decodes real
    frames in this scaffold; exists so Pipeline's call signature doesn't need
    to change once a real Tracker is wired in.
    """

    def __init__(self, path: str, num_frames: int, fps: float = 1.0):
        self.path = path
        self.num_frames = num_frames
        self.fps = fps


@runtime_checkable
class Tracker(Protocol):
    """Frozen. Not finetuned."""

    def track(self, video: VideoHandle) -> list[Trajectory]: ...


class StubTracker:
    """Frozen (stub). Returns a fixed/injected list[Trajectory] supplied at
    construction time. A real detector+SAM2 tracker plugs in later behind the
    same Protocol.
    """

    def __init__(self, canned_trajectories: Optional[list[Trajectory]] = None):
        self._canned_trajectories = canned_trajectories or []

    def track(self, video: VideoHandle) -> list[Trajectory]:
        return list(self._canned_trajectories)
