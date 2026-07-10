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

import math
from typing import Iterable, Optional, Sequence

from smot._geometry import centroid, dist, iou, orientation
from smot.types import PairFeature, RelGeom, Trajectory

# pair_feature_vectors 输出向量的维度,可学习 Pairwise KFA 构造时用它定 in_dim。
PAIR_FEATURE_DIM = 8


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
    for t in sorted(set(ts)):  # sorted+set:去重且保证按时间顺序差分
        fp_i, fp_j = traj_i.frame_at(t), traj_j.frame_at(t)
        if fp_i is None or fp_j is None:
            continue  # 双方缺一,这一帧的"相对"几何没有意义,跳过而非补零
        ci, cj = centroid(fp_i.box), centroid(fp_j.box)
        if prev is None:
            rel_vel = (0.0, 0.0)  # 第一个双方都有观测的帧,没有历史帧可差分
        else:
            t_prev, ci_prev, cj_prev = prev
            dt = float(t - t_prev)  # 注意:是与"上一个共同观测帧"的间隔,可能 >1
            vel_i = ((ci[0] - ci_prev[0]) / dt, (ci[1] - ci_prev[1]) / dt)
            vel_j = ((cj[0] - cj_prev[0]) / dt, (cj[1] - cj_prev[1]) / dt)
            rel_vel = (vel_j[0] - vel_i[0], vel_j[1] - vel_i[1])  # j 相对 i 的速度
        features.append(
            PairFeature(
                edge=edge,
                t=t,
                vis_i=(),  # Stage-0 无视觉塔;Stage-1b 填真实特征
                vis_j=(),
                rel_geom=RelGeom(
                    rel_pos=(cj[0] - ci[0], cj[1] - ci[1]),  # j 相对 i 的位移向量
                    dist=dist(ci, cj),
                    rel_vel=rel_vel,
                    orient=orientation(ci, cj),  # 从 i 指向 j 的方向角
                    overlap=iou(fp_i.box, fp_j.box),
                ),
            )
        )
        prev = (t, ci, cj)  # 更新差分基准,供下一个共同观测帧使用
    return tuple(features)


def pair_feature_vectors(
    pair_features: Sequence[PairFeature],
    t_max: Optional[int] = None,
    scale: float = 1000.0,
) -> tuple[tuple[float, ...], ...]:
    """把逐帧 PairFeature 序列压成定长 float 向量序列(与输入逐帧对齐),
    作为可学习 Pairwise KFA 的打分输入——与 frame_features.
    geometric_frame_features 完全同一种角色/同一套归一化约定。

    每帧分量(共 PAIR_FEATURE_DIM=8 个,全部为 float):
        0-1  rel_pos x, y     j 相对 i 的位移(除以 scale)
        2    dist             中心点距离(除以 scale)
        3-4  rel_vel x, y     j 相对 i 的速度(除以 scale)
        5    orient           i 指向 j 的方向角(除以 π,归一到 [-1, 1])
        6    overlap          IoU,本身在 [0, 1] 无需再缩放
        7    t_norm           帧号/t_max,归一化到 [0, 1]

    vis_i / vis_j 视觉分量目前为空(Stage-0/1 无视觉塔),接入后在此
    追加分量并同步调大 PAIR_FEATURE_DIM 即可,KFA 侧只看到 in_dim 变化。
    t_max 不传时退化用序列内最大帧号(与 frame_features 的约定一致);
    整段视频统一调用时应传全局最大帧号,保证跨候选边的时间维度同尺度。
    """
    if not pair_features:
        return ()
    if t_max is None:
        t_max = max(pf.t for pf in pair_features)
    t_scale = float(max(t_max, 1))
    s = float(max(scale, 1e-6))
    return tuple(
        (
            pf.rel_geom.rel_pos[0] / s,  # [0] 相对位移 x(归一化)
            pf.rel_geom.rel_pos[1] / s,  # [1] 相对位移 y(归一化)
            pf.rel_geom.dist / s,  # [2] 中心点距离(归一化)
            pf.rel_geom.rel_vel[0] / s,  # [3] 相对速度 x(归一化)
            pf.rel_geom.rel_vel[1] / s,  # [4] 相对速度 y(归一化)
            pf.rel_geom.orient / math.pi,  # [5] 方向角,[-π, π] -> [-1, 1]
            float(pf.rel_geom.overlap),  # [6] IoU,已在 [0, 1]
            pf.t / t_scale,  # [7] 帧号归一化到 [0, 1]
        )
        for pf in pair_features
    )
