from __future__ import annotations

import unittest

from smot.pair_features import (
    PAIR_FEATURE_DIM,
    build_pair_features,
    pair_feature_vectors,
)
from tests.fixtures import make_two_object_fixture


class TestBuildPairFeatures(unittest.TestCase):
    def setUp(self):
        self.track1, self.track2 = make_two_object_fixture()

    def test_builds_one_feature_per_common_frame(self):
        features = build_pair_features(self.track1, self.track2, (2, 3, 4))
        self.assertEqual([pf.t for pf in features], [2, 3, 4])
        for pf in features:
            self.assertEqual(pf.edge, (1, 2))

    def test_rel_geom_values_on_fixture(self):
        # t=4: track1 中心 (35, 5),track2 中心 (43, 5)。
        features = build_pair_features(self.track1, self.track2, (3, 4))
        last = features[-1]
        self.assertEqual(last.rel_geom.rel_pos, (8.0, 0.0))
        self.assertAlmostEqual(last.rel_geom.dist, 8.0)
        # t=3->4:track1 中心 x 从 25 到 35(速度 10),track2 不动,
        # 所以 j 相对 i 的速度是 (-10, 0)——正在被追上。
        self.assertEqual(last.rel_geom.rel_vel, (-10.0, 0.0))
        # t=4 两框 (30..40) 与 (38..48) 有重叠,IoU > 0。
        self.assertGreater(last.rel_geom.overlap, 0.0)

    def test_first_common_frame_has_zero_rel_vel(self):
        features = build_pair_features(self.track1, self.track2, (2, 3, 4))
        self.assertEqual(features[0].rel_geom.rel_vel, (0.0, 0.0))

    def test_frames_missing_observation_are_skipped(self):
        # track1 只有 t=0..4 的观测,传入越界帧号时应跳过而不是报错。
        features = build_pair_features(self.track1, self.track2, (3, 4, 7))
        self.assertEqual([pf.t for pf in features], [3, 4])


class TestPairFeatureVectors(unittest.TestCase):
    """pair_feature_vectors:可学习 Pairwise KFA 的打分输入向量化。"""

    def setUp(self):
        track1, track2 = make_two_object_fixture()
        self.pfs = build_pair_features(track1, track2, (2, 3, 4))

    def test_vector_dim_and_alignment(self):
        vectors = pair_feature_vectors(self.pfs, t_max=4)
        self.assertEqual(len(vectors), len(self.pfs))
        for vec in vectors:
            self.assertEqual(len(vec), PAIR_FEATURE_DIM)

    def test_time_component_normalized_by_t_max(self):
        vectors = pair_feature_vectors(self.pfs, t_max=8)
        # 最后一个分量是 t/t_max:帧号 (2, 3, 4) 除以 8。
        self.assertEqual([v[-1] for v in vectors], [0.25, 0.375, 0.5])

    def test_length_components_normalized_by_scale(self):
        vectors = pair_feature_vectors(self.pfs, t_max=4, scale=10.0)
        last = vectors[-1]
        # t=4:rel_pos=(8,0)、dist=8(见上方 fixture 注释),除以 scale=10。
        self.assertAlmostEqual(last[0], 0.8)
        self.assertAlmostEqual(last[1], 0.0)
        self.assertAlmostEqual(last[2], 0.8)

    def test_overlap_kept_raw(self):
        vectors = pair_feature_vectors(self.pfs, t_max=4)
        # IoU 分量本身在 [0,1],不缩放,应与 RelGeom 原值一致。
        for vec, pf in zip(vectors, self.pfs):
            self.assertEqual(vec[6], pf.rel_geom.overlap)

    def test_empty_input_returns_empty(self):
        self.assertEqual(pair_feature_vectors(()), ())

    def test_t_max_defaults_to_sequence_max(self):
        vectors = pair_feature_vectors(self.pfs)  # 不传 t_max -> 用序列内最大帧号 4
        self.assertEqual([v[-1] for v in vectors], [0.5, 0.75, 1.0])


if __name__ == "__main__":
    unittest.main()
