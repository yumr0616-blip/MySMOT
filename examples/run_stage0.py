"""可运行的 Stage-0 演示脚本。

构造一个小的合成双目标轨迹 fixture,用全默认(Stage-0)配置组装一个
Pipeline,跑一遍,把得到的 instance/interaction/video 断言以 JSON
形式打印出来。

用法: python examples/run_stage0.py
"""
from __future__ import annotations

import json
import os
import sys

# 让脚本无论从哪个工作目录运行,都能找到仓库根目录下的 smot 包。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smot.pipeline import Pipeline
from smot.tracker import StubTracker, VideoHandle
from smot.types import FramePresence, Trajectory


def _make_demo_trajectories() -> list[Trajectory]:
    # 这里特意从 tests/fixtures.py 里内联复制了一份同样的 fixture
    # (而不是 import tests 模块),是为了让 examples/ 目录不依赖 tests/
    # 目录——两者各自独立,互不影响。
    # track1 向右加速移动,最终与静止不动的 track2 发生框重叠。
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


def main() -> None:
    """构造 -> 跑 Pipeline -> 打印结果,三步演示全流程最小闭环。"""
    trajectories = _make_demo_trajectories()
    # StubTracker 直接把提前造好的轨迹原样返回,相当于"假装"这是
    # 冻结 tracker 的输出结果。
    # 构造 Pipeline 时不传其余任何组件参数——全部使用各自的 Stage-0
    # 默认实现(NoOp KFA/Projector、DeterministicFactSelector、MockMLLMAdapter)。
    pipeline = Pipeline(tracker=StubTracker(trajectories))
    result = pipeline.run(VideoHandle(path="synthetic://two_object_demo", num_frames=5))
    print(json.dumps(result.to_json_dict(), indent=2))


if __name__ == "__main__":
    main()
