"""smot.ml.feature_cache 的单元测试(需要 torch;stdlib-only 环境自动跳过)。

覆盖不依赖真实视觉塔前向的部分:NearestFillCache 的最近邻查找、
augmented_frame_features/augmented_pair_feature_vectors 的维度与拼接
正确性、save_cache/load_cache 的 npz 往返、以及两个 KFA 在放大后的
in_dim(几何+视觉)下仍能正确构造/前向——锁住 P2-2 接线的两侧契约。
真实视觉塔前向(batched_visual_features)由 P2-1 的离线缓存构建脚本
覆盖,不在这里重复(需要真实模型,不适合单元测试)。
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from smot.pair_features import build_pair_features
from smot.types import FramePresence, Trajectory

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import numpy as np
    import torch

    from smot.frame_features import FRAME_FEATURE_DIM
    from smot.ml.feature_cache import (
        AUGMENTED_FRAME_FEATURE_DIM,
        AUGMENTED_PAIR_FEATURE_DIM,
        VISUAL_FEATURE_DIM,
        NearestFillCache,
        augmented_frame_features,
        augmented_pair_feature_vectors,
        cache_path,
        load_cache,
        save_cache,
    )
    from smot.ml.pairwise_kfa import LearnablePairwiseKFA
    from smot.ml.unary_kfa import LearnableUnaryKFA
    from smot.pair_features import PAIR_FEATURE_DIM


def _traj(track_id: int, ts: list[int]) -> Trajectory:
    per_frame = tuple(
        FramePresence(t=t, box=(t * 5.0, 0.0, t * 5.0 + 10.0, 10.0)) for t in ts
    )
    return Trajectory(track_id=track_id, present=(ts[0], ts[-1]), per_frame=per_frame)


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class NearestFillCacheTest(unittest.TestCase):
    def test_exact_hit(self):
        vec = np.arange(VISUAL_FEATURE_DIM, dtype=np.float32)
        cache = NearestFillCache({"1_5": vec})
        found = cache.lookup(1, 5)
        self.assertIsNotNone(found)
        np.testing.assert_array_equal(found, vec)

    def test_nearest_neighbor_fallback(self):
        v0 = np.zeros(VISUAL_FEATURE_DIM, dtype=np.float32)
        v10 = np.ones(VISUAL_FEATURE_DIM, dtype=np.float32)
        cache = NearestFillCache({"1_0": v0, "1_10": v10})
        # t=3 离 t=0 更近 -> 应该拿到 v0
        np.testing.assert_array_equal(cache.lookup(1, 3), v0)
        # t=8 离 t=10 更近 -> 应该拿到 v10
        np.testing.assert_array_equal(cache.lookup(1, 8), v10)

    def test_missing_track_returns_none(self):
        cache = NearestFillCache({"1_0": np.zeros(VISUAL_FEATURE_DIM, dtype=np.float32)})
        self.assertIsNone(cache.lookup(2, 0))

    def test_empty_cache_is_falsy(self):
        self.assertFalse(NearestFillCache({}))


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class SaveLoadCacheTest(unittest.TestCase):
    def test_round_trip(self):
        features = {
            "1_0": np.arange(VISUAL_FEATURE_DIM, dtype=np.float32),
            "2_3": np.zeros(VISUAL_FEATURE_DIM, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.npz"
            save_cache(path, features)
            loaded = load_cache(path)
        self.assertEqual(set(loaded), set(features))
        for k in features:
            np.testing.assert_array_equal(loaded[k], features[k])

    def test_load_missing_file_returns_empty(self):
        self.assertEqual(load_cache(Path("/nonexistent/path.npz")), {})

    def test_cache_path_splits_activity_from_stem(self):
        p = cache_path("out/feat_cache", "test", "playing_basketball/abc123_0")
        self.assertEqual(
            p, Path("out/feat_cache/test/playing_basketball/abc123_0.npz")
        )


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class AugmentedFeaturesTest(unittest.TestCase):
    def test_augmented_frame_features_dim_and_content(self):
        traj = _traj(1, [0, 5, 10])
        vec5 = np.full(VISUAL_FEATURE_DIM, 7.0, dtype=np.float32)
        cache = NearestFillCache({"1_5": vec5})
        feats = augmented_frame_features(traj, cache, t_max=10)
        self.assertEqual(len(feats), 3)
        for row in feats:
            self.assertEqual(len(row), AUGMENTED_FRAME_FEATURE_DIM)
        # t=0 和 t=10 都离 t=5 最近(唯一条目)-> 视觉分量应回填成同一份
        self.assertEqual(feats[0][FRAME_FEATURE_DIM:], feats[2][FRAME_FEATURE_DIM:])
        self.assertAlmostEqual(feats[0][FRAME_FEATURE_DIM], 7.0)

    def test_augmented_frame_features_missing_track_zero_fills(self):
        traj = _traj(1, [0, 1])
        cache = NearestFillCache({})  # 缓存里完全没有这条轨迹
        feats = augmented_frame_features(traj, cache, t_max=1)
        for row in feats:
            self.assertEqual(len(row), AUGMENTED_FRAME_FEATURE_DIM)
            self.assertEqual(row[FRAME_FEATURE_DIM:], (0.0,) * VISUAL_FEATURE_DIM)

    def test_augmented_pair_feature_vectors_dim_and_sides(self):
        traj_i, traj_j = _traj(1, [0, 1, 2]), _traj(2, [0, 1, 2])
        pfs = build_pair_features(traj_i, traj_j, [0, 1, 2])
        cache = NearestFillCache(
            {
                "1_1": np.full(VISUAL_FEATURE_DIM, 1.0, dtype=np.float32),
                "2_1": np.full(VISUAL_FEATURE_DIM, 2.0, dtype=np.float32),
            }
        )
        feats = augmented_pair_feature_vectors(pfs, cache, t_max=2)
        self.assertEqual(len(feats), 3)
        for row in feats:
            self.assertEqual(len(row), AUGMENTED_PAIR_FEATURE_DIM)
        mid = feats[1]  # t=1,精确命中两侧缓存
        vis_i = mid[PAIR_FEATURE_DIM : PAIR_FEATURE_DIM + VISUAL_FEATURE_DIM]
        vis_j = mid[PAIR_FEATURE_DIM + VISUAL_FEATURE_DIM :]
        self.assertEqual(vis_i, (1.0,) * VISUAL_FEATURE_DIM)
        self.assertEqual(vis_j, (2.0,) * VISUAL_FEATURE_DIM)


@unittest.skipUnless(HAS_TORCH, "torch 未安装(stdlib-only 环境跳过 ml 测试)")
class KFAWithAugmentedDimTest(unittest.TestCase):
    """P2-2 的核心断言:两个 KFA 只需换个更大的 in_dim 构造,forward/select
    契约不用改一行——in_dim 只影响内部 Linear 层的输入宽度。"""

    def test_unary_kfa_accepts_augmented_in_dim(self):
        kfa = LearnableUnaryKFA(in_dim=AUGMENTED_FRAME_FEATURE_DIM)
        features = torch.rand(5, AUGMENTED_FRAME_FEATURE_DIM)
        hard, soft = kfa(features, top_k=3)
        self.assertEqual(hard.shape[0], 3)
        self.assertEqual(soft.shape[0], kfa.out_dim)
        soft.sum().backward()  # 梯度必须能穿过放大后的输入层

    def test_pairwise_kfa_accepts_augmented_in_dim(self):
        kfa = LearnablePairwiseKFA(in_dim=AUGMENTED_PAIR_FEATURE_DIM)
        features = torch.rand(4, AUGMENTED_PAIR_FEATURE_DIM)
        hard, soft = kfa(features, top_k=2)
        self.assertEqual(hard.shape[0], 2)
        self.assertEqual(soft.shape[0], kfa.out_dim)
        soft.sum().backward()


if __name__ == "__main__":
    unittest.main()
