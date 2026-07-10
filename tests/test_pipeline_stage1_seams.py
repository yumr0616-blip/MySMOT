"""Pipeline 的 Stage-1 新接缝测试:逐帧特征注入、KFA soft 向量拼入
projector 输入、video_path / frame_boxes 上下文传递,以及 Stage-1b 的
pair_feature_fn 注入、[fact | unary | pairwise] 槽位布局、fact selector
soft 读出取代均值池化。

这些接缝在 Stage-0 默认组件下必须完全无感(NoOp KFA 返回 None ->
不拼接;Mock 不消费新字段),接入真实组件时又必须真实生效——两头
都要锁住。
"""
from __future__ import annotations

import unittest

from smot.fact_selector import DeterministicFactSelector, FactSelection
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


class _SoftUnaryKFA(_RecordingUnaryKFA):
    """模拟可学习 unary KFA:除了返回 soft_token,还像真实实现一样
    暴露 out_dim——_compose_pooled 靠它决定 unary 槽位的补零宽度。"""

    def __init__(self, soft_token=(9.0, 8.0)):
        super().__init__(soft_token)
        self.out_dim = len(soft_token)


class _SoftPairwiseKFA:
    """模拟可学习 pairwise KFA:记录收到的向量化 features,返回带
    soft_token 的选择结果。"""

    def __init__(self, soft_token=(7.0, 6.0, 5.0)):
        self.soft_token = soft_token
        self.received_features = []
        self.out_dim = len(soft_token)

    def select(self, edge, event_candidate, top_k, pair_features=(), features=None):
        self.received_features.append(features)
        return KeyFrameSelection(
            key_frames=event_candidate.candidate_frames[:top_k],
            soft_token=self.soft_token,
        )


class _SoftFactSelector:
    """模拟可学习 fact selector:hard 选择沿用确定性实现,soft_token
    固定——检验 fact 槽位取 soft 读出而不是 embed 均值池化。"""

    def __init__(self, soft_token=(4.0, 3.0)):
        self.soft_token = soft_token
        self._det = DeterministicFactSelector()

    def select(self, facts, query_context):
        selection = self._det.select(facts, query_context)
        return FactSelection(
            selected_facts=selection.selected_facts,
            soft_token=self.soft_token,
            text=selection.text,
        )


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
    # Stage-1b 接缝:pair_feature_fn 注入 + 槽位布局 + fact soft 读出
    # ------------------------------------------------------------------

    def test_pair_feature_fn_reaches_pairwise_kfa(self):
        kfa = _SoftPairwiseKFA()
        marker = lambda pfs: tuple((float(pf.t), 1.0) for pf in pfs)  # noqa: E731
        self._run(pairwise_kfa=kfa, pair_feature_fn=marker)
        self.assertEqual(len(kfa.received_features), 1)  # fixture 只有一条候选边
        received = kfa.received_features[0]
        # 向量化特征与 pair_features 逐帧对齐((t, 1.0) 标记可辨识)。
        self.assertTrue(all(vec[1] == 1.0 for vec in received))

    def test_default_pair_features_stay_none(self):
        kfa = _SoftPairwiseKFA()
        self._run(pairwise_kfa=kfa)
        self.assertEqual(kfa.received_features, [None])

    def test_interaction_layout_pads_unary_slot(self):
        """interaction 调用点:pairwise soft 必须落在自己的槽位——
        unary KFA 可学习(暴露 out_dim)时,中间的 unary 槽位补零。"""
        projector = _RecordingProjector()
        self._run(
            unary_kfa=_SoftUnaryKFA(soft_token=(9.0, 8.0)),
            pairwise_kfa=_SoftPairwiseKFA(soft_token=(7.0, 6.0, 5.0)),
            projector=projector,
        )
        interaction_inputs = [p for p in projector.inputs if p[-3:] == (7.0, 6.0, 5.0)]
        self.assertEqual(len(interaction_inputs), 1)
        pooled = interaction_inputs[0]
        # [fact(4) | unary 槽位补零(2) | pairwise(3)]
        self.assertEqual(len(pooled), 4 + 2 + 3)
        self.assertEqual(pooled[4:6], (0.0, 0.0))

    def test_interaction_layout_without_learnable_unary(self):
        """unary 仍是 NoOp(无 out_dim)时布局里没有 unary 槽位:
        pairwise soft 直接接在 fact 分量之后。"""
        projector = _RecordingProjector()
        self._run(
            pairwise_kfa=_SoftPairwiseKFA(soft_token=(7.0, 6.0, 5.0)),
            projector=projector,
        )
        interaction_inputs = [p for p in projector.inputs if p[-3:] == (7.0, 6.0, 5.0)]
        self.assertEqual(len(interaction_inputs), 1)
        self.assertEqual(len(interaction_inputs[0]), 4 + 3)

    def test_fact_selector_soft_token_replaces_mean_pooling(self):
        """fact selector 可学习时,projector 输入的 fact 分量是它的 soft
        读出(2 维标记值),而不是 4 维 embed 均值池化。"""
        projector = _RecordingProjector()
        self._run(fact_selector=_SoftFactSelector(soft_token=(4.0, 3.0)), projector=projector)
        # 三个调用点(instance x2 / interaction / video)的 fact 分量全部
        # 是 soft 读出:没有 KFA 分量时池化向量恰好就是它。
        self.assertTrue(projector.inputs)
        for pooled in projector.inputs:
            self.assertEqual(pooled[:2], (4.0, 3.0))
            self.assertEqual(len(pooled), 2)

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
