"""smot/ml 可学习模块的单元测试(需要 torch;stdlib-only 环境自动跳过)。

覆盖:LearnableUnaryKFA 的 select 契约与梯度通路、MLPProjector 的
零初始化/补零/维度校验、soft token 注入机制(占位追加 + embedding
hook)在无真实 LM 的情况下的正确性。真实 Qwen3.5 上的端到端梯度验证
由 python -m smot.ml.gradient_check 门禁负责,不在单元测试范围内。
"""
from __future__ import annotations

import importlib.util
import unittest

from smot.types import FramePresence

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    from torch import nn

    from smot.frame_features import FRAME_FEATURE_DIM
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
