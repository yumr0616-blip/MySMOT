"""geometric_frame_features 的单元测试:维度、首帧约定、缺帧间隔、归一化。"""
from __future__ import annotations

import unittest

from smot.frame_features import FRAME_FEATURE_DIM, geometric_frame_features
from smot.types import FramePresence, Trajectory


def _walking_traj() -> Trajectory:
    """每帧右移 5 像素的 10x10 框(与 tests/fixtures 的 track1 前三帧一致)。"""
    return Trajectory(
        track_id=1,
        present=(1, 3),
        per_frame=(
            FramePresence(t=1, box=(0, 0, 10, 10)),
            FramePresence(t=2, box=(5, 0, 15, 10)),
            FramePresence(t=3, box=(10, 0, 20, 10)),
        ),
    )


class FrameFeaturesTest(unittest.TestCase):
    def test_shape_and_alignment(self):
        features = geometric_frame_features(_walking_traj())
        self.assertEqual(len(features), 3)  # 与 per_frame 逐帧对齐
        self.assertTrue(all(len(f) == FRAME_FEATURE_DIM for f in features))

    def test_first_frame_motion_components_are_zero(self):
        first = geometric_frame_features(_walking_traj())[0]
        # 分量布局: cx, cy, w, h, speed, dx, dy, conf, t_norm
        self.assertAlmostEqual(first[0], 5 / 1000)  # cx
        self.assertAlmostEqual(first[1], 5 / 1000)  # cy
        self.assertAlmostEqual(first[2], 10 / 1000)  # w
        self.assertAlmostEqual(first[3], 10 / 1000)  # h
        self.assertEqual((first[4], first[5], first[6]), (0.0, 0.0, 0.0))
        self.assertEqual(first[7], 1.0)  # conf 默认 1.0
        self.assertAlmostEqual(first[8], 1 / 3)  # t_norm = t / t_max(自身末帧)

    def test_speed_and_displacement(self):
        second = geometric_frame_features(_walking_traj())[1]
        self.assertAlmostEqual(second[4], 5 / 1000)  # speed
        self.assertAlmostEqual(second[5], 5 / 1000)  # dx
        self.assertAlmostEqual(second[6], 0.0)  # dy

    def test_observation_gap_divides_by_dt(self):
        """帧 1 与帧 3 之间缺一帧观测:位移/速度都按 dt=2 折算。"""
        traj = Trajectory(
            track_id=1,
            present=(1, 3),
            per_frame=(
                FramePresence(t=1, box=(0, 0, 10, 10)),
                FramePresence(t=3, box=(20, 0, 30, 10)),
            ),
        )
        second = geometric_frame_features(traj)[1]
        self.assertAlmostEqual(second[4], 10 / 1000)  # 20px / 2帧 / scale
        self.assertAlmostEqual(second[5], 10 / 1000)

    def test_t_max_and_scale_overrides(self):
        features = geometric_frame_features(_walking_traj(), t_max=10, scale=10.0)
        self.assertAlmostEqual(features[0][8], 0.1)  # t_norm 用全局 t_max
        self.assertAlmostEqual(features[0][0], 0.5)  # cx=5 / scale=10

    def test_empty_trajectory(self):
        traj = Trajectory(track_id=1, present=(0, 0), per_frame=())
        self.assertEqual(geometric_frame_features(traj), ())


if __name__ == "__main__":
    unittest.main()
