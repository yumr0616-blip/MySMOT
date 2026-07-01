"""Event Candidate Filter: heuristic (not learned) candidate-edge detection.

Per §4: finds candidate interaction edges and their candidate frames using
contact/overlap, sudden speed change, direction change, and occlusion
boundary heuristics. This is a fixed heuristic, not a learned component.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from smot._geometry import centroid, dist, iou, orientation
from smot.types import Trajectory


@dataclass(frozen=True)
class EventCandidate:
    edge: tuple[int, int]
    candidate_frames: tuple[int, ...]
    triggers: tuple[str, ...]


class EventCandidateFilter:
    """Heuristic, not learned. Produces candidate edges (pairs of track ids)
    plus the frames that triggered them, for downstream Pairwise KFA /
    interaction-assertion construction.
    """

    def __init__(
        self,
        contact_iou_threshold: float = 0.05,
        speed_change_ratio: float = 1.5,
        direction_change_deg: float = 45.0,
    ):
        self.contact_iou_threshold = contact_iou_threshold
        self.speed_change_ratio = speed_change_ratio
        self.direction_change_deg = direction_change_deg

    def _contact_frames(self, traj_i: Trajectory, traj_j: Trajectory) -> list[int]:
        common_ts = sorted(
            {fp.t for fp in traj_i.per_frame} & {fp.t for fp in traj_j.per_frame}
        )
        return [
            t
            for t in common_ts
            if iou(traj_i.frame_at(t).box, traj_j.frame_at(t).box)
            >= self.contact_iou_threshold
        ]

    def _speed_change_frames(self, traj: Trajectory) -> list[int]:
        frames = traj.per_frame
        if len(frames) < 3:
            return []
        speeds = []
        for a, b in zip(frames, frames[1:]):
            dt = b.t - a.t
            speeds.append(dist(centroid(a.box), centroid(b.box)) / dt if dt > 0 else 0.0)

        triggered: list[int] = []
        for k in range(1, len(speeds)):
            prev, cur = speeds[k - 1], speeds[k]
            if prev <= 0 and cur <= 0:
                continue
            ratio = cur / prev if prev > 0 else math.inf
            inv_ratio = prev / cur if cur > 0 else math.inf
            if ratio >= self.speed_change_ratio or inv_ratio >= self.speed_change_ratio:
                triggered.append(frames[k + 1].t)
        return triggered

    def _direction_change_frames(self, traj: Trajectory) -> list[int]:
        frames = traj.per_frame
        if len(frames) < 3:
            return []
        vectors = []
        for a, b in zip(frames, frames[1:]):
            ca, cb = centroid(a.box), centroid(b.box)
            if ca == cb:
                vectors.append(None)
            else:
                vectors.append(orientation(ca, cb))

        triggered: list[int] = []
        for k in range(1, len(vectors)):
            prev, cur = vectors[k - 1], vectors[k]
            if prev is None or cur is None:
                continue
            diff = abs(math.degrees(cur - prev))
            diff = min(diff, 360.0 - diff)
            if diff >= self.direction_change_deg:
                triggered.append(frames[k + 1].t)
        return triggered

    def _occlusion_boundary_frames(self, traj_i: Trajectory, traj_j: Trajectory) -> list[int]:
        common_ts = sorted(
            {fp.t for fp in traj_i.per_frame} & {fp.t for fp in traj_j.per_frame}
        )
        if len(common_ts) < 2:
            return []
        overlapping = [
            iou(traj_i.frame_at(t).box, traj_j.frame_at(t).box) >= self.contact_iou_threshold
            for t in common_ts
        ]
        triggered: list[int] = []
        for k in range(1, len(overlapping)):
            if overlapping[k] != overlapping[k - 1]:
                triggered.append(common_ts[k])
        return triggered

    def find_candidates(self, trajectories: list[Trajectory]) -> list[EventCandidate]:
        candidates: list[EventCandidate] = []
        for a in range(len(trajectories)):
            for b in range(a + 1, len(trajectories)):
                traj_i, traj_j = trajectories[a], trajectories[b]
                frames: set[int] = set()
                triggers: list[str] = []

                contact = self._contact_frames(traj_i, traj_j)
                if contact:
                    frames.update(contact)
                    triggers.append("contact")

                speed_change = self._speed_change_frames(traj_i) + self._speed_change_frames(traj_j)
                if speed_change:
                    frames.update(speed_change)
                    triggers.append("speed_change")

                direction_change = self._direction_change_frames(
                    traj_i
                ) + self._direction_change_frames(traj_j)
                if direction_change:
                    frames.update(direction_change)
                    triggers.append("direction_change")

                occlusion_boundary = self._occlusion_boundary_frames(traj_i, traj_j)
                if occlusion_boundary:
                    frames.update(occlusion_boundary)
                    triggers.append("occlusion_boundary")

                if frames:
                    candidates.append(
                        EventCandidate(
                            edge=(traj_i.track_id, traj_j.track_id),
                            candidate_frames=tuple(sorted(frames)),
                            triggers=tuple(triggers),
                        )
                    )
        return candidates
