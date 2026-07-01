"""Runnable Stage-0 demo.

Builds a small synthetic two-object trajectory fixture, wires a fully
default (Stage-0) Pipeline, runs it, and pretty-prints the resulting
instance/interaction/video assertions as JSON.

Usage: python examples/run_stage0.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smot.pipeline import Pipeline
from smot.tracker import StubTracker, VideoHandle
from smot.types import FramePresence, Trajectory


def _make_demo_trajectories() -> list[Trajectory]:
    # Small inline fixture (duplicated from tests/fixtures.py by design, so
    # examples/ has zero dependency on tests/): track 1 moves right and
    # accelerates, ending adjacent to stationary track 2.
    track1 = Trajectory(
        track_id=1,
        present=(0, 4),
        per_frame=(
            FramePresence(t=0, box=(0, 0, 10, 10)),
            FramePresence(t=1, box=(5, 0, 15, 10)),
            FramePresence(t=2, box=(10, 0, 20, 10)),
            FramePresence(t=3, box=(20, 0, 30, 10)),
            FramePresence(t=4, box=(30, 0, 40, 10)),
        ),
    )
    track2 = Trajectory(
        track_id=2,
        present=(0, 4),
        per_frame=tuple(FramePresence(t=t, box=(38, 0, 48, 10)) for t in range(5)),
    )
    return [track1, track2]


def main() -> None:
    trajectories = _make_demo_trajectories()
    pipeline = Pipeline(tracker=StubTracker(trajectories))
    result = pipeline.run(VideoHandle(path="synthetic://two_object_demo", num_frames=5))
    print(json.dumps(result.to_json_dict(), indent=2))


if __name__ == "__main__":
    main()
