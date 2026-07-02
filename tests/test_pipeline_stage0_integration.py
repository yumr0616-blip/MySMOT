from __future__ import annotations

import json
import unittest

from smot.kfa import NoOpPairwiseKFA
from smot.mllm import MockMLLMAdapter
from smot.pipeline import Pipeline
from smot.tracker import StubTracker, VideoHandle
from tests.fixtures import make_two_object_fixture


class _RecordingMLLM:
    """记录收到的所有 MLLMRequest,再委托给 MockMLLMAdapter 生成回复,
    用来验证 Pipeline 布线(尤其是 soft_tokens 是否真的到达 MLLM)。
    """

    def __init__(self):
        self.requests = []
        self._inner = MockMLLMAdapter()

    def generate(self, request):
        self.requests.append(request)
        return self._inner.generate(request)


class _FakeProjector:
    """恒返回一个非空 soft token 的假 projector:如果 Pipeline 把
    projector 的输出丢弃了,这个 token 永远到不了 MLLM。
    """

    def project(self, pooled_vector):
        return ((1.0, 2.0, 3.0),)


class _RecordingPairwiseKFA(NoOpPairwiseKFA):
    """记录收到的 pair_features,验证相对几何信号真的流进了 Pairwise KFA。"""

    def __init__(self):
        self.received_pair_features = []

    def select(self, edge, event_candidate, top_k, pair_features=()):
        self.received_pair_features.append(tuple(pair_features))
        return super().select(edge, event_candidate, top_k, pair_features)


class TestPipelineStage0Integration(unittest.TestCase):
    def setUp(self):
        self.trajectories = make_two_object_fixture()
        self.pipeline = Pipeline(tracker=StubTracker(self.trajectories))
        self.video = VideoHandle(path="synthetic://two_object", num_frames=5)

    def test_runs_with_fully_default_stage0_wiring(self):
        result = self.pipeline.run(self.video)
        self.assertEqual(len(result.instances), 2)
        self.assertGreaterEqual(len(result.interactions), 1)
        self.assertEqual(set(result.video.involved_ids), {1, 2})

    def test_all_evidence_frames_are_valid_indices(self):
        result = self.pipeline.run(self.video)
        ts_by_id = {
            traj.track_id: {fp.t for fp in traj.per_frame}
            for traj in self.trajectories
        }
        for assertion in result.instances:
            # 每条 instance 断言的证据帧必须来自它自己那条轨迹的观测帧,
            # 不能"借用"别的轨迹的帧号。
            self.assertTrue(
                set(assertion.evidence_frames).issubset(ts_by_id[assertion.track_id])
            )
        for assertion in result.interactions:
            # 交互证据帧必须是双方都有观测的帧。
            allowed = ts_by_id[assertion.subject_id] & ts_by_id[assertion.object_id]
            self.assertTrue(set(assertion.evidence_frames).issubset(allowed))

    def test_result_round_trips_through_json(self):
        result = self.pipeline.run(self.video)
        payload = json.loads(json.dumps(result.to_json_dict()))
        self.assertIn("instances", payload)
        self.assertIn("interactions", payload)
        self.assertIn("video", payload)
        self.assertIn("cost", payload)

    def test_interaction_predicate_is_canonicalized(self):
        result = self.pipeline.run(self.video)
        interaction = result.interactions[0]
        self.assertEqual(interaction.canonical_label, "approach")

    def test_projector_output_reaches_mllm_as_soft_tokens(self):
        # 注入一个返回非空 token 的 projector + 一个记录请求的 MLLM:
        # 每一个请求(instance/interaction/video)都必须带上这个 token,
        # 证明 soft-token 通路是接通的,而不是被 Pipeline 丢弃。
        recording = _RecordingMLLM()
        pipeline = Pipeline(
            tracker=StubTracker(self.trajectories),
            projector=_FakeProjector(),
            mllm_adapter=recording,
        )
        pipeline.run(self.video)
        self.assertGreaterEqual(len(recording.requests), 4)
        for request in recording.requests:
            self.assertEqual(request.soft_tokens, ((1.0, 2.0, 3.0),))

    def test_pairwise_kfa_receives_pair_features_with_rel_geom(self):
        recording_kfa = _RecordingPairwiseKFA()
        pipeline = Pipeline(
            tracker=StubTracker(self.trajectories), pairwise_kfa=recording_kfa
        )
        pipeline.run(self.video)
        self.assertGreaterEqual(len(recording_kfa.received_pair_features), 1)
        for pair_features in recording_kfa.received_pair_features:
            self.assertGreater(len(pair_features), 0)
            for pf in pair_features:
                # 相对几何是真实计算值(fixture 里两目标始终不同心),
                # 不是占位的零。
                self.assertGreater(pf.rel_geom.dist, 0.0)

    def test_interaction_direction_follows_mllm_statement(self):
        # MLLM(canned)明确声明相反方向时,组装出的断言应交换
        # subject/object,而不是沿用候选边的下标顺序。
        mllm = MockMLLMAdapter(
            canned_responses={"interaction": "subject_id=2 approaches object_id=1."}
        )
        pipeline = Pipeline(tracker=StubTracker(self.trajectories), mllm_adapter=mllm)
        result = pipeline.run(self.video)
        interaction = result.interactions[0]
        self.assertEqual(interaction.subject_id, 2)
        self.assertEqual(interaction.object_id, 1)

    def test_cost_report_counts_vlm_calls(self):
        result = self.pipeline.run(self.video)
        expected_calls = len(result.instances) + len(result.interactions) + 1
        self.assertEqual(result.cost.n_vlm_calls, expected_calls)
        self.assertGreater(result.cost.n_key_frames, 0)
        self.assertGreater(result.cost.n_facts_selected, 0)
        # Stage-0 的 NoOpProjector 不产生 soft token。
        self.assertEqual(result.cost.n_soft_tokens, 0)


if __name__ == "__main__":
    unittest.main()
