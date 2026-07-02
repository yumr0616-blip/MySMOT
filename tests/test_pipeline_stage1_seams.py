"""Pipeline 的 Stage-1 新接缝测试:逐帧特征注入、KFA soft 向量拼入
projector 输入、video_path / frame_boxes 上下文传递。

这些接缝在 Stage-0 默认组件下必须完全无感(NoOp KFA 返回 None ->
不拼接;Mock 不消费新字段),接入真实组件时又必须真实生效——两头
都要锁住。
"""
from __future__ import annotations

import unittest

from smot.kfa import KeyFrameSelection
from smot.mllm import MockMLLMAdapter
from smot.pipeline import Pipeline
from smot.tracker import StubTracker, VideoHandle
from smot.types import FramePresence, Trajectory


def _fixture_trajectories() -> list[Trajectory]:
    """track1 向右加速接近静止的 track2(会触发一条交互候选边)。"""
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


class _RecordingMLLM:
    """转发给 Mock,同时记录收到的所有请求(检查新字段是否被填上)。"""

    def __init__(self):
        self.requests = []
        self._mock = MockMLLMAdapter()

    def generate(self, request):
        self.requests.append(request)
        return self._mock.generate(request)


class _RecordingUnaryKFA:
    """记录收到的 features,并返回带 soft_token 的选择结果。"""

    def __init__(self, soft_token=(9.0, 8.0)):
        self.calls = []
        self.soft_token = soft_token

    def select(self, track_id, frames, top_k, features=None):
        self.calls.append((track_id, features))
        return KeyFrameSelection(
            key_frames=tuple(fp.t for fp in frames[:top_k]),
            soft_token=self.soft_token,
        )


class _RecordingProjector:
    def __init__(self):
        self.inputs = []

    def project(self, pooled_vector):
        self.inputs.append(tuple(pooled_vector))
        return ()


class PipelineStage1SeamsTest(unittest.TestCase):
    def _run(self, **pipeline_kwargs):
        trajectories = _fixture_trajectories()
        mllm = _RecordingMLLM()
        pipeline = Pipeline(
            tracker=StubTracker(trajectories),
            mllm_adapter=mllm,
            **pipeline_kwargs,
        )
        result = pipeline.run(
            VideoHandle(path="synthetic://imgs", num_frames=5)
        )
        return result, mllm

    # ------------------------------------------------------------------
    # frame_feature_fn 注入
    # ------------------------------------------------------------------

    def test_frame_feature_fn_reaches_unary_kfa(self):
        kfa = _RecordingUnaryKFA()
        marker = lambda traj: ((float(traj.track_id), 0.5),)  # noqa: E731
        self._run(unary_kfa=kfa, frame_feature_fn=marker)
        self.assertEqual(
            [(tid, feats) for tid, feats in kfa.calls],
            [(1, ((1.0, 0.5),)), (2, ((2.0, 0.5),))],
        )

    def test_default_features_stay_none(self):
        kfa = _RecordingUnaryKFA()
        self._run(unary_kfa=kfa)
        self.assertTrue(all(feats is None for _tid, feats in kfa.calls))

    # ------------------------------------------------------------------
    # KFA soft 向量拼入 projector 输入
    # ------------------------------------------------------------------

    def test_kfa_soft_token_concatenated_into_projector_input(self):
        kfa = _RecordingUnaryKFA(soft_token=(9.0, 8.0))
        projector = _RecordingProjector()
        self._run(unary_kfa=kfa, projector=projector)
        # 每个 instance 调用的 projector 输入 = 4 维事实池化 + 2 维 soft。
        instance_inputs = [p for p in projector.inputs if len(p) == 6]
        self.assertEqual(len(instance_inputs), 2)  # 两条轨迹
        for pooled in instance_inputs:
            self.assertEqual(pooled[-2:], (9.0, 8.0))

    def test_noop_kfa_keeps_projector_input_untouched(self):
        projector = _RecordingProjector()
        self._run(projector=projector)
        # NoOp KFA 的 soft_token 是 None:所有调用点都只有 4 维事实池化。
        self.assertTrue(all(len(p) == 4 for p in projector.inputs if p))

    # ------------------------------------------------------------------
    # video_path / frame_boxes 上下文
    # ------------------------------------------------------------------

    def test_video_path_on_every_request(self):
        _result, mllm = self._run()
        self.assertTrue(len(mllm.requests) >= 3)
        self.assertTrue(
            all(r.video_path == "synthetic://imgs" for r in mllm.requests)
        )

    def test_instance_frame_boxes_match_trajectory(self):
        _result, mllm = self._run()
        request = next(r for r in mllm.requests if r.prompt_type == "instance")
        self.assertEqual(len(request.frame_boxes), len(request.frame_refs))
        for (t, entries), ref in zip(request.frame_boxes, request.frame_refs):
            self.assertEqual(t, ref)
            self.assertEqual(len(entries), 1)  # instance 任务只有一个 track
            track_id, box = entries[0]
            self.assertEqual(len(box), 4)

    def test_interaction_frame_boxes_contain_both_tracks(self):
        _result, mllm = self._run()
        request = next(r for r in mllm.requests if r.prompt_type == "interaction")
        self.assertTrue(request.frame_boxes)
        for _t, entries in request.frame_boxes:
            self.assertEqual(sorted(tid for tid, _box in entries), [1, 2])

    def test_video_request_has_no_frames(self):
        _result, mllm = self._run()
        request = next(r for r in mllm.requests if r.prompt_type == "video")
        self.assertEqual(request.frame_refs, ())
        self.assertEqual(request.frame_boxes, ())


if __name__ == "__main__":
    unittest.main()
