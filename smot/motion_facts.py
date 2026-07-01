"""Motion Fact Extractor: deterministic geometry facts from box sequences.

Deterministic (not learned) per the design doc's §4 module table. Every fact
value is computed directly from tracker boxes ("trajectory-faithful, zero
hallucination"); nothing here is a model prediction.
"""
from __future__ import annotations

from smot._geometry import centroid, dist
from smot.types import Fact, FactType, Trajectory, FACT_TYPE_ORDER


def _embed(fact_type: FactType, norm_value: float, t_span: tuple[int, int]) -> tuple[float, ...]:
    type_index = float(FACT_TYPE_ORDER.index(fact_type))
    return (type_index, float(norm_value), float(t_span[0]), float(t_span[1]))


class MotionFactExtractor:
    """Deterministic. Computes presence/net_motion/speed facts per instance
    and proximity/approach facts per pair of trajectories.
    """

    def __init__(self, speed_window: int = 1, proximity_threshold: float = 50.0):
        self.speed_window = speed_window
        self.proximity_threshold = proximity_threshold

    def extract_instance_facts(self, traj: Trajectory) -> list[Fact]:
        facts: list[Fact] = []
        t_in, t_out = traj.present
        scope = f"instance:{traj.track_id}"

        facts.append(
            Fact(
                type=FactType.PRESENCE,
                scope=scope,
                value=(t_in, t_out),
                t_span=(t_in, t_out),
                embed=_embed(FactType.PRESENCE, t_out - t_in, (t_in, t_out)),
            )
        )

        if len(traj.per_frame) < 2:
            return facts

        c_first = centroid(traj.per_frame[0].box)
        c_last = centroid(traj.per_frame[-1].box)
        dx, dy = c_last[0] - c_first[0], c_last[1] - c_first[1]
        facts.append(
            Fact(
                type=FactType.NET_MOTION,
                scope=scope,
                value=(dx, dy),
                t_span=(t_in, t_out),
                embed=_embed(FactType.NET_MOTION, dist((0, 0), (dx, dy)), (t_in, t_out)),
            )
        )

        speeds: list[float] = []
        for a, b in zip(traj.per_frame, traj.per_frame[1:]):
            dt = b.t - a.t
            if dt <= 0:
                continue
            speeds.append(dist(centroid(a.box), centroid(b.box)) / dt)
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        facts.append(
            Fact(
                type=FactType.SPEED,
                scope=scope,
                value=mean_speed,
                t_span=(t_in, t_out),
                embed=_embed(FactType.SPEED, mean_speed, (t_in, t_out)),
            )
        )
        return facts

    def extract_pair_facts(self, traj_i: Trajectory, traj_j: Trajectory) -> list[Fact]:
        facts: list[Fact] = []
        common_ts = sorted(
            {fp.t for fp in traj_i.per_frame} & {fp.t for fp in traj_j.per_frame}
        )
        if not common_ts:
            return facts

        scope = f"pair:{traj_i.track_id},{traj_j.track_id}"
        t_span = (common_ts[0], common_ts[-1])
        distances = [
            dist(centroid(traj_i.frame_at(t).box), centroid(traj_j.frame_at(t).box))
            for t in common_ts
        ]

        min_dist, mean_dist = min(distances), sum(distances) / len(distances)
        facts.append(
            Fact(
                type=FactType.PROXIMITY,
                scope=scope,
                value={"min": min_dist, "mean": mean_dist},
                t_span=t_span,
                embed=_embed(FactType.PROXIMITY, min_dist, t_span),
            )
        )

        if distances[-1] < distances[0]:
            approach_value = "approaching"
        elif distances[-1] > distances[0]:
            approach_value = "receding"
        else:
            approach_value = "steady"
        facts.append(
            Fact(
                type=FactType.APPROACH,
                scope=scope,
                value=approach_value,
                t_span=t_span,
                embed=_embed(FactType.APPROACH, distances[0] - distances[-1], t_span),
            )
        )
        return facts

    def extract(self, trajectories: list[Trajectory]) -> list[Fact]:
        facts: list[Fact] = []
        for traj in trajectories:
            facts.extend(self.extract_instance_facts(traj))
        for a in range(len(trajectories)):
            for b in range(a + 1, len(trajectories)):
                facts.extend(self.extract_pair_facts(trajectories[a], trajectories[b]))
        return facts
