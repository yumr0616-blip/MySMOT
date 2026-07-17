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

    edge: tuple[int, int]  # (track_id_i, track_id_j),i < j
    candidate_frames: tuple[int, ...]  # 触发了任意规则的帧号,已排序去重
    triggers: tuple[str, ...]  # 命中的触发规则名称(如 "contact"、"speed_change")


def adaptive_proximity_gate(
    trajectories: list[Trajectory], scale: float = 1.5, floor: float = 50.0
) -> float:
    """按目标尺寸自适应的邻近门控阈值:全体观测框对角线的中位数 × scale。

    默认的 proximity_gate=50(绝对像素)隐含了小分辨率/小目标假设——
    换到 1080p 的真人框,两人隔半个身位就会被判"不近",单目标突变全被
    门控挡掉,交互召回率静默塌零(真实 BenSMOT 接入时踩到的坑)。
    "近"的语义本质上是相对目标尺度的:以框对角线(人≈身高)为单位,
    scale=1.5 即"一个半身位以内算近"。floor 兜底空轨迹/退化框。
    """
    diagonals = sorted(
        math.hypot(fp.box[2] - fp.box[0], fp.box[3] - fp.box[1])
        for traj in trajectories
        for fp in traj.per_frame
    )
    if not diagonals:
        return floor
    median = diagonals[len(diagonals) // 2]
    return max(scale * median, floor)


class EventCandidateFilter:
    """启发式,不学习。产出候选边 + 触发帧,供 Pairwise KFA /
    交互断言构建使用。
    """

    def __init__(
        self,
        contact_iou_threshold: float = 0.05,
        speed_change_ratio: float = 1.5,
        direction_change_deg: float = 45.0,
        proximity_gate: float = 50.0,
        proximity_trigger: bool = False,
    ):
        self.contact_iou_threshold = contact_iou_threshold  # IoU 达到此值算"接触"
        self.speed_change_ratio = speed_change_ratio  # 相邻两段速度比值超过此值算"突变"
        self.direction_change_deg = direction_change_deg  # 相邻两段方向夹角超过此值(度)算"突变"
        # 单目标层面的突变(速度/方向)只有在"这一对目标当时离得足够近"
        # 时才可能与这一对的交互有关。这个门控阈值(中心点距离,像素)
        # 用来避免 n 目标场景里一个目标的突变把它和其余 n-1 个目标的
        # 候选边全部点亮——每条候选边都是一次 MLLM 调用,§7 把调用次数
        # 列为一等公民成本指标,候选边爆炸会直接拉高成本基线。
        # 真实数据上建议用 adaptive_proximity_gate() 按目标尺度取值。
        self.proximity_gate = proximity_gate
        # 邻近状态跳变触发器(见 _proximity_transition_frames):覆盖
        # "无接触、无速度/方向突变"的社交类交互(交谈、并肩……)。
        # 默认关闭以保持 Stage-0 已核验的候选语义与成本地板不变;
        # 接真实数据的脚本应显式开启。
        self.proximity_trigger = proximity_trigger

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
        (或其倒数)超过 speed_change_ratio 阈值,就认为发生了速度的
        突然变化。两段速度之间的分界点是 frames[k](两段的公共端点),
        记录分界帧和它的下一帧各一帧——"突变前"和"突变后"各留一张
        证据帧,对下游做断言核验的 judge(§7)最有用。
        少于 3 帧(不足两段速度可比较)时直接返回空列表。
        """
        frames = traj.per_frame
        if len(frames) < 3:
            return []
        speeds = []
        for a, b in zip(frames, frames[1:]):
            # dt > 0 由 Trajectory 构造校验保证,这里的条件只是防御性
            # 除零守卫。
            dt = b.t - a.t
            speeds.append(dist(centroid(a.box), centroid(b.box)) / dt if dt > 0 else 0.0)

        triggered: set[int] = set()
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
                triggered.update((frames[k].t, frames[k + 1].t))
        return sorted(triggered)

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

        triggered: set[int] = set()
        for k in range(1, len(vectors)):
            prev, cur = vectors[k - 1], vectors[k]
            if prev is None or cur is None:
                continue
            diff = abs(math.degrees(cur - prev))
            # 角度差可能算出大于 180 度(比如 -170 度 与 170 度之间)，
            # 取 360 - diff 得到真正的最短夹角。
            diff = min(diff, 360.0 - diff)
            if diff >= self.direction_change_deg:
                # 同速度突变:分界帧 frames[k] 和下一帧各记一帧,
                # 给下游留下"转向前/转向后"两张证据帧。
                triggered.update((frames[k].t, frames[k + 1].t))
        return sorted(triggered)

    def _occlusion_boundary_frames(self, traj_i: Trajectory, traj_j: Trajectory) -> list[int]:
        """遮挡边界触发,两类信号:

        1) 重叠状态跳变:两个目标的框 IoU 是否达到阈值,发生"从不重叠到
           重叠"或反向的跳变,往往对应遮挡开始/结束的时刻。重叠本身就
           蕴含"离得近",所以这类信号不需要再做邻近度门控。
        2) 公共可见性断点:真实遮挡的主信号其实是"观测缺失"——目标被
           挡住时 tracker 根本给不出框。所以公共观测帧序列里出现空洞
           (相邻公共帧号不连续)时,把空洞两侧的帧记为触发帧(消失前
           最后一帧 + 重新出现的第一帧)。但观测缺失也可能只是 tracker
           在别处单纯跟丢了——如果不加限制,一个目标的一次跟丢会把它和
           场上所有目标的候选边全部点亮(与速度/方向突变同样的广播
           爆炸)。所以空洞边界帧必须通过 _pair_close_at 邻近度门控
           (边界帧是公共观测帧,双方都有框,距离可算):只有消失时刻
           这对目标本来就离得近,"被对方遮挡"才是合理怀疑。
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
        triggered: set[int] = set()
        for k in range(1, len(overlapping)):
            if overlapping[k] != overlapping[k - 1]:
                triggered.add(common_ts[k])
            if common_ts[k] - common_ts[k - 1] > 1:
                triggered.update(
                    t
                    for t in (common_ts[k - 1], common_ts[k])
                    if self._pair_close_at(traj_i, traj_j, t)
                )
        return sorted(triggered)

    def _proximity_transition_frames(
        self, traj_i: Trajectory, traj_j: Trajectory
    ) -> list[int]:
        """邻近状态跳变触发:两目标中心距离相对 proximity_gate 的"近/远"
        状态发生翻转的时刻(走到一起 / 分开),记录翻转点前后各一帧
        (与速度/方向突变同一约定);另外,公共可见的第一帧就已经"近"
        时也记为触发帧——全程贴在一起的一对(如面对面交谈)从不发生
        状态翻转,没有这一条它们永远成不了候选。

        接触/突变/遮挡四类规则都要求某种"离散事件",而 BenSMOT 这类
        数据里大量交互(交谈、对视、并肩同行)恰恰没有任何离散事件,
        本触发器是它们的补集。语义上"近"本身就是门控,不需要再过
        _pair_close_at。
        """
        common_ts = sorted(
            {fp.t for fp in traj_i.per_frame} & {fp.t for fp in traj_j.per_frame}
        )
        if not common_ts:
            return []
        near = [
            dist(centroid(traj_i.frame_at(t).box), centroid(traj_j.frame_at(t).box))
            <= self.proximity_gate
            for t in common_ts
        ]
        triggered: set[int] = set()
        if near[0]:
            triggered.add(common_ts[0])
        for k in range(1, len(near)):
            if near[k] != near[k - 1]:
                triggered.update((common_ts[k - 1], common_ts[k]))
        return sorted(triggered)

    def _pair_close_at(self, traj_i: Trajectory, traj_j: Trajectory, t: int) -> bool:
        """判断两个目标在帧 t 是否"离得足够近"(中心点距离不超过
        proximity_gate)。任意一方在该帧没有观测时返回 False——距离
        无从算起,保守起见不把这一帧算进这对目标的候选帧(观测缺失
        本身由 _occlusion_boundary_frames 的可见性断点信号负责捕捉)。
        """
        fp_i, fp_j = traj_i.frame_at(t), traj_j.frame_at(t)
        if fp_i is None or fp_j is None:
            return False
        return dist(centroid(fp_i.box), centroid(fp_j.box)) <= self.proximity_gate

    def find_candidates(self, trajectories: list[Trajectory]) -> list[EventCandidate]:
        """对所有轨迹两两组合,分别跑上面四类触发规则,把命中的帧号
        合并去重、排序,同时记录命中了哪些触发规则类型。如果一对目标
        没有任何规则命中，就不会出现在返回列表里(意味着它们大概率没有
        发生交互，不值得再让 MLLM 去描述这一对)。

        速度/方向突变是单目标层面的规则,先对每条轨迹算一次缓存起来
        (避免在 O(n^2) 的 pair 循环里对同一条轨迹重复算 n-1 次),
        并入某条 pair 边之前再用 proximity_gate 门控:只有突变发生时
        这对目标离得足够近,才认为该突变可能与这一对的交互有关——
        否则一个目标的一次急刹会把它和场上所有目标的候选边全部点亮,
        候选边数量(= MLLM 调用次数)会随目标数失控膨胀。
        """
        speed_changes = {
            traj.track_id: self._speed_change_frames(traj) for traj in trajectories
        }
        direction_changes = {
            traj.track_id: self._direction_change_frames(traj) for traj in trajectories
        }

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

                if self.proximity_trigger:
                    proximity = self._proximity_transition_frames(traj_i, traj_j)
                    if proximity:
                        frames.update(proximity)
                        triggers.append("proximity")

                # 只要这一对里任意一方发生了突变都算数(比如追逐场景里,
                # 只有追的一方会突然加速),但必须通过邻近度门控。
                speed_change = [
                    t
                    for t in speed_changes[traj_i.track_id] + speed_changes[traj_j.track_id]
                    if self._pair_close_at(traj_i, traj_j, t)
                ]
                if speed_change:
                    frames.update(speed_change)
                    triggers.append("speed_change")

                direction_change = [
                    t
                    for t in direction_changes[traj_i.track_id]
                    + direction_changes[traj_j.track_id]
                    if self._pair_close_at(traj_i, traj_j, t)
                ]
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
