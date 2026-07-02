"""fact_norm 归一化器 + Pipeline fact_transform 接缝的测试。"""
from __future__ import annotations

import unittest

from smot.fact_norm import make_fact_embed_normalizer
from smot.mllm import MockMLLMAdapter
from smot.pipeline import Pipeline
from smot.tracker import StubTracker, VideoHandle
from smot.types import Fact, FactType, FramePresence, Trajectory


def _speed_fact(value: float) -> Fact:
    return Fact(
        type=FactType.SPEED,
        scope="instance:1",
        value=value,
        t_span=(0, 4),
        embed=(3.0, value, 0.0, 1.0),
    )


class FactNormalizerTest(unittest.TestCase):
    def test_zscore_applied_to_norm_value_only(self):
        normalize = make_fact_embed_normalizer(
            {"speed": {"n": 10, "mean": 5.0, "std": 2.0}}
        )
        (fact,) = normalize([_speed_fact(9.0)])
        # embed[1] 被 z-score:(9-5)/2 = 2;其余分量与 value 本身不动。
        self.assertEqual(fact.embed, (3.0, 2.0, 0.0, 1.0))
        self.assertEqual(fact.value, 9.0)

    def test_unknown_type_and_zero_std_pass_through(self):
        normalize = make_fact_embed_normalizer(
            {"presence": {"n": 5, "mean": 2.0, "std": 0.0}}  # std=0 -> 不变换
        )
        (fact,) = normalize([_speed_fact(9.0)])  # speed 不在表里 -> 不变换
        self.assertEqual(fact.embed, (3.0, 9.0, 0.0, 1.0))


class _RecordingProjector:
    def __init__(self):
        self.inputs = []

    def project(self, pooled_vector):
        self.inputs.append(tuple(pooled_vector))
        return ()


class PipelineFactTransformSeamTest(unittest.TestCase):
    def test_transform_applies_before_selection_and_pooling(self):
        traj = Trajectory(
            track_id=1,
            present=(0, 1),
            per_frame=(
                FramePresence(t=0, box=(0, 0, 10, 10)),
                FramePresence(t=1, box=(5, 0, 15, 10)),
            ),
        )
        marker = lambda facts: [  # noqa: E731 - 测试用变换:embed 全打成常量
            type(f)(f.type, f.scope, f.value, f.t_span, (0.0, 42.0, 0.0, 0.0))
            for f in facts
        ]
        projector = _RecordingProjector()
        pipeline = Pipeline(
            tracker=StubTracker([traj]),
            mllm_adapter=MockMLLMAdapter(),
            projector=projector,
            fact_transform=marker,
        )
        pipeline.run(VideoHandle(path="synthetic://x", num_frames=2))
        # 所有事实的 embed 都被变换 -> 逐维平均后 pooled[1] 必为 42。
        self.assertTrue(projector.inputs)
        for pooled in projector.inputs:
            if pooled:
                self.assertEqual(pooled[1], 42.0)


if __name__ == "__main__":
    unittest.main()
