"""跟框(box)序列相关的几何基础函数。

这些都是确定性计算(不涉及任何学习),被 motion_facts.py 和
event_filter.py 两个"真正实现"的模块共用,避免同样的几何公式在两处
各写一份、容易出现不一致。
"""
from __future__ import annotations

import math

from smot.types import Box


def centroid(box: Box) -> tuple[float, float]:
    """计算一个框的中心点坐标(对角线两端点坐标各自取平均)。"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """两点间的欧氏距离。math.hypot 等价于 sqrt(dx**2 + dy**2),
    但对极端值更数值稳定,且不用手写平方和开根号。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iou(a: Box, b: Box) -> float:
    """两个框的交并比(Intersection over Union)。

    先算交集矩形的宽高(不重叠时用 max(0, ...) 截断为 0),再用
    面积公式 union = area_a + area_b - inter 算并集,union 为 0
    (两个框都退化成零面积)时直接返回 0,避免除零。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # 交集矩形 = 两个框各自边界取"内侧"的那一个:左上角取 max,右下角取 min。
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    # 两框不重叠时 ix2-ix1 或 iy2-iy1 会是负数,用 max(0, ...) 截断为 0,
    # 表示"没有交集"而不是负面积。
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0  # 两个框都退化成零面积(如 x1==x2),约定返回 0 而不是抛除零异常
    return inter / union


def orientation(a: tuple[float, float], b: tuple[float, float]) -> float:
    """从点 a 指向点 b 的方向角(弧度),用于检测运动方向是否发生突变。

    使用 atan2(dy, dx) 而不是 atan(dy/dx),是为了保留完整的
    [-pi, pi] 象限信息(atan 会把方向相反的两个向量算成同一个角度)。
    """
    return math.atan2(b[1] - a[1], b[0] - a[0])
