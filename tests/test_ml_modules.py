"""smot/ml 可学习模块的单元测试(需要 torch;stdlib-only 环境自动跳过)。

覆盖:LearnableUnaryKFA / LearnablePairwiseKFA / LearnableFactSelector
的 select 契约与梯度通路、MLPProjector 的零初始化/补零/维度校验、
[fact | unary | pairwise] 槽位布局在训练侧(torch.cat)与推理侧
(Pipeline._compose_pooled + projector 尾部补零)的一致性、soft token
注入机制(占位追加 + embedding hook)在无真实 LM 的情况下的正确性。
真实 Qwen3.5 上的端到端梯度验证由 python -m smot.ml.gradient_check
门禁负责,不在单元测试范围内。
"""
from __future__ import annotations

import importlib.util
import unittest

from smot.event_filter import EventCandidate
from smot.fact_selector import SelectionContext, render_fact
from smot.pair_features import PAIR_FEATURE_DIM, pair_feature_vectors
from smot.types import Fact, FactType, FramePresence, PairFeature, RelGeom

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    from torch import nn

    from smot.frame_features import FRAME_FEATURE_DIM
    from smot.ml.fact_selector import (
        FACT_SCORE_DIM,
        LearnableFactSelector,
        fact_scoring_features,
    )
    from smot.ml.pairwise_kfa import LearnablePairwiseKFA
    from smot.ml.projector import MLPProjector
    from smot.ml.qwen_adapter import append_placeholder_tokens, soft_token_injection
    from smot.ml.unary_kfa import LearnableUnaryKFA


def _frames(n: int) -> list:
    return [FramePresence(t=t, box=(t * 5.0, 0.0, t * 5.0 + 10.0, 10.0)) for t in range(n)]


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class LearnableUnaryKFATest(unittest.TestCase):
    def test_select_returns_sorted_topk_and_soft_token(self):
        kfa = LearnableUnaryKFA()
        frames = _frames(10)
        features = [tuple(float(i) for _ in range(FRAME_FEATURE_DIM)) for i in range(10)]
        selection = kfa.select(1, frames, top_k=4, features=features)
        self.assertEqual(len(selection.key_frames), 4)
        self.assertEqual(list(selection.key_frames), sorted(selection.key_frames))
        self.assertEqual(len(selection.soft_token), kfa.out_dim)

    def test_select_without_features_raises(self):
        with self.assertRaises(ValueError):
            LearnableUnaryKFA().select(1, _frames(5), top_k=2, features=None)

    def test_select_feature_length_mismatch_raises(self):
        features = [(0.0,) * FRAME_FEATURE_DIM] * 3
        with self.assertRaises(ValueError):
            LearnableUnaryKFA().select(1, _frames(5), top_k=2, features=features)

    def test_top_k_larger_than_frames_returns_all(self):
        features = [(0.0,) * FRAME_FEATURE_DIM] * 3
        selection = LearnableUnaryKFA().select(1, _frames(3), top_k=8, features=features)
        self.assertEqual(len(selection.key_frames), 3)

    def test_gradient_flows_through_soft_readout(self):
        """hard top-k 无自有梯度,训练信号必须经 soft 读出到达 scorer。"""
        kfa = LearnableUnaryKFA()
        features = torch.rand(6, FRAME_FEATURE_DIM)
        _hard, soft = kfa(features, top_k=3)
        soft.sum().backward()
        scorer_norm = sum(
            float(p.grad.norm()) for p in kfa.scorer.parameters() if p.grad is not None
        )
        value_norm = sum(
            float(p.grad.norm()) for p in kfa.value.parameters() if p.grad is not None
        )
        self.assertGreater(scorer_norm, 0.0)
        self.assertGreater(value_norm, 0.0)


def _pair_fixture(n: int):
    """n 帧的候选边 fixture:EventCandidate + 逐帧 PairFeature(几何值
    随 t 变化,保证打分有区分度)+ 对齐的向量化特征。"""
    candidate = EventCandidate(
        edge=(1, 2), candidate_frames=tuple(range(n)), triggers=("proximity",)
    )
    pfs = tuple(
        PairFeature(
            edge=(1, 2),
            t=t,
            vis_i=(),
            vis_j=(),
            rel_geom=RelGeom(
                rel_pos=(float(t), 0.0),
                dist=float(t),
                rel_vel=(1.0, 0.0),
                orient=0.0,
                overlap=0.0,
            ),
        )
        for t in range(n)
    )
    return candidate, pfs, pair_feature_vectors(pfs, t_max=n)


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class LearnablePairwiseKFATest(unittest.TestCase):
    def test_select_returns_sorted_topk_and_soft_token(self):
        kfa = LearnablePairwiseKFA()
        candidate, pfs, features = _pair_fixture(10)
        selection = kfa.select((1, 2), candidate, top_k=4, pair_features=pfs, features=features)
        self.assertEqual(len(selection.key_frames), 4)
        self.assertEqual(list(selection.key_frames), sorted(selection.key_frames))
        self.assertEqual(len(selection.soft_token), kfa.out_dim)

    def test_select_without_features_raises(self):
        candidate, pfs, _features = _pair_fixture(5)
        with self.assertRaises(ValueError):
            LearnablePairwiseKFA().select((1, 2), candidate, top_k=2, pair_features=pfs)

    def test_select_feature_length_mismatch_raises(self):
        candidate, pfs, features = _pair_fixture(5)
        with self.assertRaises(ValueError):
            LearnablePairwiseKFA().select(
                (1, 2), candidate, top_k=2, pair_features=pfs, features=features[:3]
            )

    def test_empty_pair_features_falls_back_with_zero_soft(self):
        """双方无共同观测帧:hard 退回候选帧等间隔抽稀(仍给 MLLM 看图),
        soft 为全零(信息源缺席),槽位布局不塌缩。"""
        kfa = LearnablePairwiseKFA()
        candidate = EventCandidate(
            edge=(1, 2), candidate_frames=(0, 3, 6, 9), triggers=("proximity",)
        )
        selection = kfa.select((1, 2), candidate, top_k=2, pair_features=())
        self.assertEqual(selection.key_frames, (0, 9))
        self.assertEqual(selection.soft_token, (0.0,) * kfa.out_dim)

    def test_top_k_larger_than_frames_returns_all(self):
        candidate, pfs, features = _pair_fixture(3)
        selection = LearnablePairwiseKFA().select(
            (1, 2), candidate, top_k=8, pair_features=pfs, features=features
        )
        self.assertEqual(len(selection.key_frames), 3)

    def test_gradient_flows_through_soft_readout(self):
        """hard top-k 无自有梯度,训练信号必须经 soft 读出到达 scorer。"""
        kfa = LearnablePairwiseKFA()
        features = torch.rand(6, PAIR_FEATURE_DIM)
        _hard, soft = kfa(features, top_k=3)
        soft.sum().backward()
        scorer_norm = sum(
            float(p.grad.norm()) for p in kfa.scorer.parameters() if p.grad is not None
        )
        value_norm = sum(
            float(p.grad.norm()) for p in kfa.value.parameters() if p.grad is not None
        )
        self.assertGreater(scorer_norm, 0.0)
        self.assertGreater(value_norm, 0.0)


def _facts_fixture() -> list:
    """一组跨类型/跨 scope 的事实,embed 值有区分度。"""
    def fact(ftype, scope, value, span, norm_value):
        return Fact(
            type=ftype,
            scope=scope,
            value=value,
            t_span=span,
            embed=(0.0, norm_value, span[0] / 10.0, span[1] / 10.0),
        )

    return [
        fact(FactType.SPEED, "instance:1", 5.0, (0, 4), 0.5),
        fact(FactType.NET_MOTION, "instance:1", (30.0, 0.0), (0, 4), 1.2),
        fact(FactType.PRESENCE, "instance:1", "0..4", (0, 4), 0.0),
        fact(FactType.PROXIMITY, "pair:1,2", 8.0, (2, 4), -0.3),
        fact(FactType.APPROACH, "pair:1,2", "closer", (2, 4), 0.7),
    ]


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class LearnableFactSelectorTest(unittest.TestCase):
    def test_select_scopes_and_renders_like_deterministic(self):
        """scope 过滤 + 渲染格式与确定性实现一致(优先级顺序、render_fact)。"""
        selector = LearnableFactSelector()
        facts = _facts_fixture()
        selection = selector.select(facts, SelectionContext(scope="pair:1,2", top_k=8))
        # pair scope 只有 2 条事实,全部选中;渲染顺序按 FACT_TYPE_ORDER
        # (PROXIMITY 在 APPROACH 之前)。
        self.assertEqual(len(selection.selected_facts), 2)
        self.assertEqual(
            [f.type for f in selection.selected_facts],
            [FactType.PROXIMITY, FactType.APPROACH],
        )
        self.assertEqual(
            selection.text,
            "; ".join(render_fact(f) for f in selection.selected_facts),
        )
        self.assertEqual(len(selection.soft_token), selector.out_dim)

    def test_top_k_limits_selection(self):
        selector = LearnableFactSelector()
        selection = selector.select(
            _facts_fixture(), SelectionContext(scope="instance:1", top_k=2)
        )
        self.assertEqual(len(selection.selected_facts), 2)

    def test_video_scope_sees_all_facts(self):
        selector = LearnableFactSelector()
        selection = selector.select(
            _facts_fixture(), SelectionContext(scope="video", top_k=8)
        )
        self.assertEqual(len(selection.selected_facts), 5)

    def test_empty_scope_returns_zero_soft(self):
        """scope 内没有事实:soft 为全零而不是 None——槽位布局不塌缩。"""
        selector = LearnableFactSelector()
        selection = selector.select(
            _facts_fixture(), SelectionContext(scope="instance:99", top_k=4)
        )
        self.assertEqual(selection.selected_facts, ())
        self.assertEqual(selection.text, "")
        self.assertEqual(selection.soft_token, (0.0,) * selector.out_dim)

    def test_scoring_features_dim(self):
        feats = fact_scoring_features(_facts_fixture())
        for row in feats:
            self.assertEqual(len(row), FACT_SCORE_DIM)

    def test_gradient_flows_through_soft_readout(self):
        selector = LearnableFactSelector()
        feats = torch.rand(5, FACT_SCORE_DIM)
        _hard, soft = selector(feats, top_k=3)
        soft.sum().backward()
        scorer_norm = sum(
            float(p.grad.norm())
            for p in selector.scorer.parameters()
            if p.grad is not None
        )
        self.assertGreater(scorer_norm, 0.0)


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class SlotLayoutContractTest(unittest.TestCase):
    """[fact | unary | pairwise] 槽位布局:训练侧 torch.cat 组装与推理侧
    Pipeline._compose_pooled(+ projector.project 尾部补零)必须逐槽一致
    ——两侧各自实现,靠本测试锁死同一契约。"""

    def setUp(self):
        from smot.fact_selector import FactSelection
        from smot.pipeline import Pipeline
        from smot.tracker import StubTracker

        self.unary = LearnableUnaryKFA()
        self.pairwise = LearnablePairwiseKFA()
        self.selector = LearnableFactSelector()
        self.in_dim = (
            self.selector.out_dim + self.unary.out_dim + self.pairwise.out_dim
        )
        self.pipeline = Pipeline(
            tracker=StubTracker([]),
            unary_kfa=self.unary,
            pairwise_kfa=self.pairwise,
            fact_selector=self.selector,
        )
        # 三个槽位用可辨识的标记值,错位立刻暴露。
        self.soft_f = (1.0,) * self.selector.out_dim
        self.soft_u = (2.0,) * self.unary.out_dim
        self.soft_p = (3.0,) * self.pairwise.out_dim
        self.fact_selection = FactSelection(
            selected_facts=(), soft_token=self.soft_f, text=""
        )

    def _padded(self, pooled: tuple) -> tuple:
        """projector.project 的尾部补零语义(project() 内部逻辑的镜像)。"""
        return tuple(pooled) + (0.0,) * (self.in_dim - len(pooled))

    def test_instance_layout_matches_training(self):
        pipeline_pooled = self._padded(
            self.pipeline._compose_pooled(self.fact_selection, unary_soft=self.soft_u)
        )
        training_pooled = torch.cat(
            [
                torch.tensor(self.soft_f),
                torch.tensor(self.soft_u),
                torch.zeros(self.pairwise.out_dim),
            ]
        )
        self.assertEqual(pipeline_pooled, tuple(training_pooled.tolist()))

    def test_interaction_layout_matches_training(self):
        pipeline_pooled = self._padded(
            self.pipeline._compose_pooled(
                self.fact_selection, pairwise_soft=self.soft_p
            )
        )
        training_pooled = torch.cat(
            [
                torch.tensor(self.soft_f),
                torch.zeros(self.unary.out_dim),
                torch.tensor(self.soft_p),
            ]
        )
        self.assertEqual(pipeline_pooled, tuple(training_pooled.tolist()))

    def test_video_layout_matches_training(self):
        pipeline_pooled = self._padded(
            self.pipeline._compose_pooled(self.fact_selection)
        )
        training_pooled = torch.cat(
            [
                torch.tensor(self.soft_f),
                torch.zeros(self.unary.out_dim),
                torch.zeros(self.pairwise.out_dim),
            ]
        )
        self.assertEqual(pipeline_pooled, tuple(training_pooled.tolist()))


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class StageCheckpointRoundTripTest(unittest.TestCase):
    """1b checkpoint 存/取往返 + 1a 历史格式经通用加载器的判别。"""

    def test_stage1b_round_trip(self):
        import tempfile
        from pathlib import Path

        from smot.ml.checkpoint import load_checkpoint, save_stage1b_checkpoint

        unary = LearnableUnaryKFA()
        pairwise = LearnablePairwiseKFA()
        selector = LearnableFactSelector()
        projector = MLPProjector(
            in_dim=selector.out_dim + unary.out_dim + pairwise.out_dim,
            d_llm=32,
            n_tokens=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stage1b.pt"
            save_stage1b_checkpoint(
                path, unary, pairwise, selector, projector, extra={"steps": 7}
            )
            loaded = load_checkpoint(path)
        self.assertEqual(loaded.stage, "1b")
        self.assertEqual(loaded.extra["steps"], 7)
        self.assertEqual(loaded.projector.in_dim, projector.in_dim)
        # 权重逐张量一致(state_dict 往返无损)。
        for orig, back in (
            (unary, loaded.unary_kfa),
            (pairwise, loaded.pairwise_kfa),
            (selector, loaded.fact_selector),
            (projector, loaded.projector),
        ):
            for key, value in orig.state_dict().items():
                self.assertTrue(torch.equal(value, back.state_dict()[key]))

    def test_stage1a_payload_detected(self):
        import tempfile
        from pathlib import Path

        from smot.ml.checkpoint import load_checkpoint, save_stage1a_checkpoint

        kfa = LearnableUnaryKFA()
        projector = MLPProjector(in_dim=4 + kfa.out_dim, d_llm=32, n_tokens=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stage1a.pt"
            save_stage1a_checkpoint(path, kfa, projector, extra={"epochs": 1})
            loaded = load_checkpoint(path)
        self.assertEqual(loaded.stage, "1a")
        self.assertIsNone(loaded.pairwise_kfa)
        self.assertIsNone(loaded.fact_selector)
        self.assertEqual(loaded.extra["epochs"], 1)


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class MLPProjectorTest(unittest.TestCase):
    def test_output_gain_sets_token_scale(self):
        """LayerNorm 归一化到单位方差后按 output_gain 缩放:token 的
        每维标准差应约等于 output_gain(对齐词嵌入 RMS 的机制)。"""
        torch.manual_seed(0)
        gain = 0.05
        projector = MLPProjector(in_dim=6, d_llm=64, n_tokens=2, output_gain=gain)
        tokens = projector.project((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertEqual(len(tokens), 2)
        self.assertEqual(len(tokens[0]), 64)
        for row in tokens:
            tensor = torch.tensor(row)
            self.assertLess(abs(float(tensor.mean())), gain * 0.5)
            self.assertAlmostEqual(float(tensor.std(correction=0)), gain, delta=gain * 0.2)

    def test_first_step_gradients_reach_all_layers(self):
        """梯度门禁抓出的回归:输出层不许零初始化(会堵死上游梯度)。

        loss 用平方和而不是 sum()——LayerNorm 输出的逐维求和恒等于 0
        (均值中心化),sum() 对它的梯度处处为零,会假阳性地"复现"
        梯度堵塞;平方和没有这个退化。"""
        projector = MLPProjector(in_dim=6, d_llm=16, n_tokens=2)
        projector(torch.rand(1, 6)).pow(2).sum().backward()
        grad_norm = float(projector.input_proj.weight.grad.norm())
        self.assertGreater(grad_norm, 0.0)

    def test_short_input_zero_padded(self):
        """interaction/video 调用点(Stage-1b 前)只有 4 维事实池化,
        projector 按缺失分量为零补齐,不报错。"""
        projector = MLPProjector(in_dim=36, d_llm=16, n_tokens=2)
        tokens = projector.project((1.0, 2.0, 3.0, 4.0))
        self.assertEqual(len(tokens), 2)

    def test_overlong_input_raises(self):
        projector = MLPProjector(in_dim=4, d_llm=16)
        with self.assertRaises(ValueError):
            projector.project((1.0,) * 5)

    def test_empty_input_returns_empty(self):
        projector = MLPProjector(in_dim=4, d_llm=16)
        self.assertEqual(projector.project(()), ())

    def test_forward_batch_shape(self):
        projector = MLPProjector(in_dim=8, d_llm=16, n_tokens=3)
        out = projector(torch.rand(5, 8))
        self.assertEqual(tuple(out.shape), (5, 3, 16))


if HAS_TORCH:

    class _TinyEmbedModel(nn.Module):
        """只有 embedding 层的假模型,用来单测注入机制本身。"""

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 8)

        def get_input_embeddings(self):
            return self.embedding


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class SoftTokenInjectionTest(unittest.TestCase):
    def test_placeholder_append_extends_aligned_fields(self):
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, 3, dtype=torch.long),
        }
        start = append_placeholder_tokens(inputs, 2, pad_id=0)
        self.assertEqual(start, 3)
        self.assertEqual(tuple(inputs["input_ids"].shape), (1, 5))
        self.assertEqual(tuple(inputs["attention_mask"].shape), (1, 5))
        self.assertEqual(int(inputs["attention_mask"][0, -1]), 1)
        self.assertEqual(tuple(inputs["mm_token_type_ids"].shape), (1, 5))
        self.assertEqual(int(inputs["mm_token_type_ids"][0, -1]), 0)

    def test_placeholder_insert_at_position(self):
        """position 指定时在序列中间插入(soft token 放在用户回合末尾、
        assistant 生成头之前的布局依赖这一行为)。"""
        inputs = {
            "input_ids": torch.tensor([[10, 11, 12, 13]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        start = append_placeholder_tokens(inputs, 2, pad_id=0, position=3)
        self.assertEqual(start, 3)
        self.assertEqual(inputs["input_ids"][0].tolist(), [10, 11, 12, 0, 0, 13])
        self.assertEqual(tuple(inputs["attention_mask"].shape), (1, 6))

    def test_hook_replaces_positions_and_carries_gradient(self):
        model = _TinyEmbedModel()
        ids = torch.tensor([[1, 2, 3, 0, 0]])  # 后两个是占位 token
        soft = torch.randn(2, 8, requires_grad=True)
        with soft_token_injection(model, soft, start_pos=3):
            out = model.embedding(ids)
        self.assertTrue(torch.equal(out[0, 3:5], soft))
        # 前缀位置不受影响。
        self.assertTrue(torch.equal(out[0, :3], model.embedding.weight[ids[0, :3]]))
        out.sum().backward()
        self.assertIsNotNone(soft.grad)
        self.assertTrue(torch.all(soft.grad == 1.0))

    def test_hook_skips_incremental_steps(self):
        """kv-cache 增量步(序列长度不覆盖注入区间)必须原样放行。"""
        model = _TinyEmbedModel()
        soft = torch.randn(2, 8)
        short_ids = torch.tensor([[7]])
        with soft_token_injection(model, soft, start_pos=3):
            out = model.embedding(short_ids)
        self.assertTrue(torch.equal(out[0, 0], model.embedding.weight[7]))

    def test_none_soft_is_noop(self):
        model = _TinyEmbedModel()
        ids = torch.tensor([[1, 2]])
        with soft_token_injection(model, None, start_pos=0):
            out = model.embedding(ids)
        self.assertTrue(torch.equal(out[0], model.embedding.weight[ids[0]]))


if __name__ == "__main__":
    unittest.main()
