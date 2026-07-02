from __future__ import annotations

import unittest

from smot.pair_features import build_pair_features
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


if __name__ == "__main__":
    unittest.main()
