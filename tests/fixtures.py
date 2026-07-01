"""Synthetic trajectory fixtures for tests and examples/run_stage0.py.

Box coordinates are round numbers chosen so centroid/distance/IoU arithmetic
is exact and hand-checkable (see tests/test_motion_facts.py and
tests/test_event_filter.py for the worked-out expected values).
"""
from __future__ import annotations

from smot.types import FramePresence, Trajectory


def make_two_object_fixture() -> list[Trajectory]:
    """5 frames (t=0..4), 2 trajectories.

    track_id=1 moves steadily right at speed 5 for two steps, then speeds up
    to 10 for two steps (single, unambiguous speed-change frame at t=3), and
    ends adjacent to (overlapping) track_id=2, which is stationary. Distance
    between the two strictly decreases every frame ("approaching").
    """
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


def make_single_object_fixture() -> list[Trajectory]:
    """1 trajectory, 3 frames, zero motion: isolates presence/net_motion=0/
    speed=0, and (with no second object) confirms EventCandidateFilter
    returns no candidates.
    """
    track = Trajectory(
        track_id=1,
        present=(0, 2),
        per_frame=tuple(FramePresence(t=t, box=(0, 0, 10, 10)) for t in range(3)),
    )
    return [track]
