"""Motion Fact Extractor:从框序列里确定性地抽取运动事实。

对应设计文档 §4 模块表里的"确定性(Deterministic,不学习)"一栏——
每一条事实的数值都是直接从 tracker 给出的框坐标算出来的,不经过任何
模型推理,因此天然满足"轨迹忠实、零幻觉"的要求。真正需要学习的部分
(选哪些事实进 prompt)在 fact_selector.py 里。
"""
from __future__ import annotations

from typing import Optional

from smot._geometry import centroid, dist
from smot.types import Fact, FactType, Trajectory, FACT_TYPE_ORDER


def _embed(
    fact_type: FactType, norm_value: float, t_span: tuple[int, int], t_scale: float
) -> tuple[float, ...]:
    """按 Fact.embed 的固定约定构造一个 4 维向量,供未来的 Fact Selector
    打分用。type_index 取该事实类型在 FACT_TYPE_ORDER 中的下标;
    t_span 两个分量除以 t_scale(视频最大帧号)归一化到 [0, 1],
    否则长视频里的原始帧号数值会淹没前两个维度的打分信号。
    norm_value 暂时仍是原始数值(跨数据集归一化需要数据集统计量,
    是 Stage-1a 训练前的显式待办)。
    """
    type_index = float(FACT_TYPE_ORDER.index(fact_type))
    scale = max(t_scale, 1.0)  # 防止 t_scale 为 0(极短/单帧视频)时除零
    return (
        type_index,  # 事实类型在 FACT_TYPE_ORDER 里的下标
        float(norm_value),  # 该事实的"强度"标量(不同类型含义不同,见各调用处)
        t_span[0] / scale,  # 起始帧号归一化到 [0, 1]
        t_span[1] / scale,  # 结束帧号归一化到 [0, 1]
    )


class MotionFactExtractor:
    """确定性(不学习)。分别计算单目标事实(presence/net_motion/speed)
    和两两 pair 事实(proximity/approach)。
    """

    def extract_instance_facts(self, traj: Trajectory, t_max: Optional[int] = None) -> list[Fact]:
        """抽取单个轨迹自身的三类事实:出现时段、净位移、平均速度。

        t_max 是整段视频的最大帧号,用于 embed 的时间归一化;单独调用
        (不经过 extract())时可以不传,此时退化为用该轨迹自身的最大
        帧号做归一化。
        """
        if t_max is None:
            t_max = traj.per_frame[-1].t if traj.per_frame else traj.present[1]
        t_scale = float(t_max)
        facts: list[Fact] = []
        t_in, t_out = traj.present
        scope = f"instance:{traj.track_id}"

        # 1) presence:直接就是轨迹的出现区间,值本身就是 (t_in, t_out)。
        facts.append(
            Fact(
                type=FactType.PRESENCE,
                scope=scope,
                value=(t_in, t_out),
                t_span=(t_in, t_out),
                embed=_embed(FactType.PRESENCE, t_out - t_in, (t_in, t_out), t_scale),
            )
        )

        # 帧数不足 2 帧算不出位移/速度,直接返回已有的 presence 事实。
        if len(traj.per_frame) < 2:
            return facts

        # 2) net_motion:首帧中心点到末帧中心点的位移向量 (dx, dy)。
        #    这是"净位移",不代表实际运动路径长度(比如来回运动会被抵消)。
        c_first = centroid(traj.per_frame[0].box)
        c_last = centroid(traj.per_frame[-1].box)
        dx, dy = c_last[0] - c_first[0], c_last[1] - c_first[1]
        facts.append(
            Fact(
                type=FactType.NET_MOTION,
                scope=scope,
                value=(dx, dy),
                t_span=(t_in, t_out),
                embed=_embed(FactType.NET_MOTION, dist((0, 0), (dx, dy)), (t_in, t_out), t_scale),
            )
        )

        # 3) speed:逐帧算相邻两帧中心点位移 / 时间差,再取平均,
        #    得到该轨迹在整个出现区间内的平均速度(单位:像素/帧)。
        speeds: list[float] = []
        for a, b in zip(traj.per_frame, traj.per_frame[1:]):
            dt = b.t - a.t
            if dt <= 0:
                # 防御性守卫:Trajectory 构造时已校验帧号严格递增,
                # 正常情况下不会触发;保留是为了避免除零。
                continue
            speeds.append(dist(centroid(a.box), centroid(b.box)) / dt)
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        facts.append(
            Fact(
                type=FactType.SPEED,
                scope=scope,
                value=mean_speed,
                t_span=(t_in, t_out),
                embed=_embed(FactType.SPEED, mean_speed, (t_in, t_out), t_scale),
            )
        )
        return facts

    def extract_pair_facts(
        self, traj_i: Trajectory, traj_j: Trajectory, t_max: Optional[int] = None
    ) -> list[Fact]:
        """抽取两个轨迹之间的 pair 事实:接近程度(proximity)和
        靠近/远离趋势(approach)。只在两者都出现的公共帧范围内计算。
        t_max 含义同 extract_instance_facts,不传时用两条轨迹的最大帧号。
        """
        if t_max is None:
            t_max = max(
                (fp.t for traj in (traj_i, traj_j) for fp in traj.per_frame),
                default=max(traj_i.present[1], traj_j.present[1]),
            )
        t_scale = float(t_max)
        facts: list[Fact] = []
        # 两个轨迹各自的观测帧号取交集,只有双方都出现的帧才能算相对距离。
        common_ts = sorted(
            {fp.t for fp in traj_i.per_frame} & {fp.t for fp in traj_j.per_frame}
        )
        if not common_ts:
            # 两者从未同时出现过(比如时间上完全错开),无法算 pair 事实。
            return facts

        # scope 键统一用排序后的 id,与调用方传入两条轨迹的先后顺序解耦
        # (pair 事实本身是对称的;方向语义不在 scope 里表达)。
        lo, hi = sorted((traj_i.track_id, traj_j.track_id))
        scope = f"pair:{lo},{hi}"
        t_span = (common_ts[0], common_ts[-1])
        distances = [
            dist(centroid(traj_i.frame_at(t).box), centroid(traj_j.frame_at(t).box))
            for t in common_ts
        ]

        # 1) proximity:公共帧范围内的最小距离和平均距离都记录下来,
        #    min 反映"最近时有多近",mean 反映整体接近程度。
        min_dist, mean_dist = min(distances), sum(distances) / len(distances)
        facts.append(
            Fact(
                type=FactType.PROXIMITY,
                scope=scope,
                value={"min": min_dist, "mean": mean_dist},
                t_span=t_span,
                embed=_embed(FactType.PROXIMITY, min_dist, t_span, t_scale),
            )
        )

        # 2) approach:只比较公共帧范围内"第一帧"和"最后一帧"的距离,
        #    用简单的首尾对比判断趋势,不做逐帧回归拟合(足够便宜、
        #    足够确定性,复杂的趋势判断留给未来可能的改进)。
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
                embed=_embed(FactType.APPROACH, distances[0] - distances[-1], t_span, t_scale),
            )
        )
        return facts

    def extract(self, trajectories: list[Trajectory]) -> list[Fact]:
        """对外的总入口:先对每个轨迹单独抽取实例事实,再对所有轨迹两两
        组合抽取 pair 事实(共 C(n,2) 对,轨迹数很大时需注意成本,当前
        脚手架规模下可以接受)。embed 的时间归一化统一用全体轨迹的最大
        帧号,保证同一段视频里所有事实的时间维度在同一个尺度上可比。
        """
        t_max = max(
            (fp.t for traj in trajectories for fp in traj.per_frame), default=0
        )
        facts: list[Fact] = []
        for traj in trajectories:
            facts.extend(self.extract_instance_facts(traj, t_max=t_max))
        # 双重循环 b 从 a+1 开始:只枚举无序对 (a, b),既不重复也不算 (i, i)。
        for a in range(len(trajectories)):
            for b in range(a + 1, len(trajectories)):
                facts.extend(
                    self.extract_pair_facts(trajectories[a], trajectories[b], t_max=t_max)
                )
        return facts
