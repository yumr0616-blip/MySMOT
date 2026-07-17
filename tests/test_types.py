from __future__ import annotations

import unittest

from smot.types import FramePresence, Trajectory


class TestTrajectoryValidation(unittest.TestCase):
    """Trajectory 在构造时做数据契约校验:下游的净位移/等间隔抽帧/
    approach 首尾对比全都隐式依赖 per_frame 有序,乱序数据必须在入口
    fail-fast,而不是让运动事实静默算错。
    """

    def test_valid_trajectory_constructs(self):
        traj = Trajectory(
            track_id=1,
            present=(0, 2),
            per_frame=(
                FramePresence(t=0, box=(0, 0, 10, 10)),
                FramePresence(t=1, box=(5, 0, 15, 10)),
                FramePresence(t=2, box=(10, 0, 20, 10)),
            ),
        )
        self.assertEqual(traj.track_id, 1)

    def test_unsorted_per_frame_raises(self):
        with self.assertRaises(ValueError):
            Trajectory(
                track_id=1,
                present=(0, 2),
                per_frame=(
                    FramePresence(t=2, box=(0, 0, 10, 10)),
                    FramePresence(t=0, box=(5, 0, 15, 10)),
                ),
            )

    def test_duplicate_frame_number_raises(self):
        with self.assertRaises(ValueError):
            Trajectory(
                track_id=1,
                present=(0, 2),
                per_frame=(
                    FramePresence(t=1, box=(0, 0, 10, 10)),
                    FramePresence(t=1, box=(5, 0, 15, 10)),
                ),
            )

    def test_per_frame_outside_present_span_raises(self):
        with self.assertRaises(ValueError):
            Trajectory(
                track_id=1,
                present=(0, 2),
                per_frame=(
                    FramePresence(t=0, box=(0, 0, 10, 10)),
                    FramePresence(t=5, box=(5, 0, 15, 10)),
                ),
            )

    def test_inverted_present_span_raises(self):
        with self.assertRaises(ValueError):
            Trajectory(track_id=1, present=(4, 0), per_frame=())


if __name__ == "__main__":
    unittest.main()
