"""确定性构造候选边的逐帧 PairFeature(§4 Pairwise KFA 的输入)。

Pairwise KFA 之所以能学出交互的方向性,靠的是逐帧的相对几何信号
(相对位置/距离/相对速度/朝向/重叠度);如果只把两个目标各自的
unary 特征拼在一起喂给 slot,方向信息就丢了。这里的构造完全是
确定性几何计算,和 Motion Fact 一样"轨迹忠实、零幻觉"。

Stage-0 没有视觉塔,vis_i / vis_j 暂时是空 tuple;Stage-1b 接入真实
视觉特征时只需要在 Pipeline 侧把这两个分量填上,本函数与 KFA 的
Protocol 签名都不需要再改。
"""
from __future__ import annotations

from typing import Iterable

from smot._geometry import centroid, dist, iou, orientation
from smot.types import PairFeature, RelGeom, Trajectory


def build_pair_features(
    traj_i: Trajectory, traj_j: Trajectory, ts: Iterable[int]
) -> tuple[PairFeature, ...]:
    """对给定帧号序列构造逐帧 PairFeature,只保留双方都有观测的帧
    (缺观测的帧算不出相对几何,直接跳过)。

    rel_vel(j 相对 i 的速度)用"与前一个双方共同观测帧的差分"估计:
    第一个共同观测帧没有历史可差分,记 (0, 0)。
    """
    edge = (traj_i.track_id, traj_j.track_id)
    features: list[PairFeature] = []
    prev: tuple[int, tuple[float, float], tuple[float, float]] | None = None  # (t, ci, cj)
    for t in sorted(set(ts)):
        fp_i, fp_j = traj_i.frame_at(t), traj_j.frame_at(t)
        if fp_i is None or fp_j is None:
            continue
        ci, cj = centroid(fp_i.box), centroid(fp_j.box)
        if prev is None:
            rel_vel = (0.0, 0.0)
        else:
            t_prev, ci_prev, cj_prev = prev
            dt = float(t - t_prev)
            vel_i = ((ci[0] - ci_prev[0]) / dt, (ci[1] - ci_prev[1]) / dt)
            vel_j = ((cj[0] - cj_prev[0]) / dt, (cj[1] - cj_prev[1]) / dt)
            rel_vel = (vel_j[0] - vel_i[0], vel_j[1] - vel_i[1])
        features.append(
            PairFeature(
                edge=edge,
                t=t,
                vis_i=(),  # Stage-0 无视觉塔;Stage-1b 填真实特征
                vis_j=(),
                rel_geom=RelGeom(
                    rel_pos=(cj[0] - ci[0], cj[1] - ci[1]),
                    dist=dist(ci, cj),
                    rel_vel=rel_vel,
                    orient=orientation(ci, cj),
                    overlap=iou(fp_i.box, fp_j.box),
                ),
            )
        )
        prev = (t, ci, cj)
    return tuple(features)
