"""逐帧几何/运动特征(纯 stdlib):Stage-1a 可学习 Unary KFA 的打分输入。

对应 §4 里 Unary KFA 的输入契约:"与 per_frame 逐帧对齐的实例特征"。
设计文档预期的完整体是视觉塔特征,但视觉塔前向很贵;Stage-1a 第一版
先用确定性可得的几何/运动量验证训练闭环(能不能学、梯度通不通),
视觉特征作为后续增强拼接在这些分量之后即可,KFA 侧只看到 in_dim 变大。

所有长度量统一除以 scale(默认 1000 像素),时间除以 t_max——保证
不同分辨率/时长的视频喂进同一个打分 MLP 时数值尺度大致可比。
"""
from __future__ import annotations

from typing import Optional

from smot._geometry import centroid, dist
from smot.types import Trajectory

# geometric_frame_features 输出向量的维度,可学习 KFA 构造时用它定 in_dim。
FRAME_FEATURE_DIM = 9


def geometric_frame_features(
    traj: Trajectory,
    t_max: Optional[int] = None,
    scale: float = 1000.0,
) -> tuple[tuple[float, ...], ...]:
    """对一条轨迹的每个观测帧计算 9 维几何/运动特征,与 per_frame 对齐。

    每帧分量(全部为 float):
        0-3  cx, cy, w, h        中心点坐标与框宽高(除以 scale)
        4    speed               与上一观测帧的中心点距离/帧差(首帧为 0)
        5-6  dx, dy              与上一观测帧的中心点位移/帧差(首帧为 0)
        7    conf                tracker 置信度(BenSMOT 里是 visibility)
        8    t_norm              帧号/t_max,归一化到 [0, 1]

    t_max 不传时用该轨迹自身最后一帧的帧号(与 MotionFactExtractor 的
    退化约定一致);整段视频统一调用时应传全局最大帧号,保证跨轨迹的
    时间维度同尺度。
    """
    if not traj.per_frame:
        return ()
    if t_max is None:
        t_max = traj.per_frame[-1].t
    t_scale = float(max(t_max, 1))
    s = float(max(scale, 1e-6))

    features: list[tuple[float, ...]] = []
    prev = None  # 上一个观测帧,用来算帧间差分(速度/位移);首帧没有上一帧
    for fp in traj.per_frame:
        x1, y1, x2, y2 = fp.box
        cx, cy = centroid(fp.box)
        if prev is None:
            speed = dx = dy = 0.0  # 首帧没有"上一帧"可比,运动量约定为 0
        else:
            dt = fp.t - prev.t  # Trajectory 构造已保证严格递增,dt >= 1
            pcx, pcy = centroid(prev.box)
            dx = (cx - pcx) / dt  # x 方向位移速率(per 帧)
            dy = (cy - pcy) / dt  # y 方向位移速率(per 帧)
            speed = dist((pcx, pcy), (cx, cy)) / dt  # 标量速度(位移距离/dt)
        features.append(
            (
                cx / s,  # [0] 中心点 x(归一化)
                cy / s,  # [1] 中心点 y(归一化)
                (x2 - x1) / s,  # [2] 框宽(归一化)
                (y2 - y1) / s,  # [3] 框高(归一化)
                speed / s,  # [4] 速度(归一化)
                dx / s,  # [5] x 方向位移速率(归一化)
                dy / s,  # [6] y 方向位移速率(归一化)
                float(fp.conf),  # [7] 置信度/可见度,已在 [0,1] 无需再缩放
                fp.t / t_scale,  # [8] 帧号归一化到 [0, 1]
            )
        )
        prev = fp
    return tuple(features)
