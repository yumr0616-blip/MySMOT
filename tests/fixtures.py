"""供测试和 examples/run_stage0.py 使用的合成轨迹 fixture。

框坐标都特意选用整数,让中心点/距离/IoU 的计算结果是精确值,方便在
tests/test_motion_facts.py 和 tests/test_event_filter.py 里手算出
期望值直接做 assertEqual 断言(而不必用近似比较)。
"""
from __future__ import annotations

from smot.types import FramePresence, Trajectory


def make_two_object_fixture() -> list[Trajectory]:
    """5 帧(t=0..4),2 条轨迹。

    track_id=1 先以速度 5 匀速向右移动两步,然后加速到速度 10 再移动
    两步——这样只有唯一一个、不会产生歧义的"速度突变帧"(在 t=3);
    同时它的终点和静止不动的 track_id=2 发生了框重叠(t=4)。
    两条轨迹中心点之间的距离逐帧严格递减,因此 approach 事实的值
    应该是 "approaching"。
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
    """只有 1 条轨迹、3 帧、完全静止不动:用来验证
    presence/net_motion=0/speed=0 这几个"零值"分支,以及在没有第二个
    目标的情况下 EventCandidateFilter 应该返回空列表(没有 pair 可组)。
    """
    track = Trajectory(
        track_id=1,
        present=(0, 2),
        per_frame=tuple(FramePresence(t=t, box=(0, 0, 10, 10)) for t in range(3)),
    )
    return [track]
