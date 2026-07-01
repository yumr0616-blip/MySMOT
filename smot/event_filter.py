"""Event Candidate Filter:启发式(不学习)的候选交互边检测。

对应 §4:根据接触/重叠、速度突变、方向突变、遮挡边界这几类启发式规则,
找出"可能发生了交互"的候选目标对(edge)以及触发这些规则的具体帧号。
这是固定规则,不是学习出来的组件——它只是给下游的 Pairwise KFA /
交互断言生成提供候选范围,避免对所有 O(n^2) 目标对都无脑跑一遍 MLLM。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from smot._geometry import centroid, dist, iou, orientation
from smot.types import Trajectory


@dataclass(frozen=True)
class EventCandidate:
    """一条候选交互边:哪两个 track_id、被哪些帧触发、触发原因是什么。"""

    edge: tuple[int, int]
    candidate_frames: tuple[int, ...]
    triggers: tuple[str, ...]


class EventCandidateFilter:
    """启发式,不学习。产出候选边 + 触发帧,供 Pairwise KFA /
    交互断言构建使用。
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
        """接触/重叠触发:两个目标的框在同一帧的 IoU 达到阈值,视为
        "接触"事件候选帧。只在两者都有观测的公共帧里检查。
        """
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
        """速度突变触发:对单个轨迹,比较相邻两段的速度,如果速度比值
        (或其倒数)超过 speed_change_ratio 阈值,就认为在这里发生了
        速度的突然变化,记录下变化"显现"出来的那一帧(即后一段区间的
        终点帧)。少于 3 帧(不足两段速度可比较)时直接返回空列表。
        """
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
                # 前后两段都是静止(速度都是 0),谈不上"突变"，跳过。
                continue
            # 用两个方向的比值(cur/prev 和 prev/cur)分别检查加速和减速，
            # 只要有一个方向的比值超过阈值就算触发；某一段速度为 0 时
            # 比值会趋向无穷大，用 math.inf 表示必然触发。
            ratio = cur / prev if prev > 0 else math.inf
            inv_ratio = prev / cur if cur > 0 else math.inf
            if ratio >= self.speed_change_ratio or inv_ratio >= self.speed_change_ratio:
                triggered.append(frames[k + 1].t)
        return triggered

    def _direction_change_frames(self, traj: Trajectory) -> list[int]:
        """方向突变触发:对单个轨迹,比较相邻两段位移向量的朝向角，
        如果角度差(取到 [0, 180] 范围内最短角度差)超过阈值，
        就认为方向发生了突变。原地不动(位移为零向量)的那一段方向角
        没有意义，标记为 None 并跳过比较。
        """
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
            # 角度差可能算出大于 180 度(比如 -170 度 与 170 度之间)，
            # 取 360 - diff 得到真正的最短夹角。
            diff = min(diff, 360.0 - diff)
            if diff >= self.direction_change_deg:
                triggered.append(frames[k + 1].t)
        return triggered

    def _occlusion_boundary_frames(self, traj_i: Trajectory, traj_j: Trajectory) -> list[int]:
        """遮挡边界触发:两个目标的重叠状态(是否 IoU 达到阈值)发生
        "从不重叠到重叠"或"从重叠到不重叠"的跳变,这个跳变点往往对应
        遮挡开始/结束的时刻,是值得关注的交互边界。
        """
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
        """对所有轨迹两两组合,分别跑上面四类触发规则,把命中的帧号
        合并去重、排序,同时记录命中了哪些触发规则类型。如果一对目标
        没有任何规则命中，就不会出现在返回列表里(意味着它们大概率没有
        发生交互，不值得再让 MLLM 去描述这一对)。
        """
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

                # 速度突变/方向突变是"单目标"层面的规则,但既然是在判断
                # 这一对目标的候选边,只要其中任意一方发生了突变都算数
                # (比如追逐场景里,只有追的一方会突然加速)。
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
