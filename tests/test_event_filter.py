from __future__ import annotations

import unittest

from smot.event_filter import EventCandidateFilter, adaptive_proximity_gate
from smot.types import FramePresence, Trajectory
from tests.fixtures import make_single_object_fixture, make_two_object_fixture


class TestEventCandidateFilterTwoObject(unittest.TestCase):
    def setUp(self):
        self.trajectories = make_two_object_fixture()
        self.filt = EventCandidateFilter()

    def test_finds_single_edge(self):
        candidates = self.filt.find_candidates(self.trajectories)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].edge, (1, 2))

    def test_candidate_frames_and_triggers(self):
        candidate = self.filt.find_candidates(self.trajectories)[0]
        # 速度突变贡献 (2, 3)(分界帧 + 后一帧),contact/occlusion 贡献 4。
        self.assertEqual(candidate.candidate_frames, (2, 3, 4))
        self.assertEqual(
            set(candidate.triggers), {"contact", "speed_change", "occlusion_boundary"}
        )
        self.assertNotIn("direction_change", candidate.triggers)

    def test_contact_frames_private_helper(self):
        track1, track2 = self.trajectories
        self.assertEqual(self.filt._contact_frames(track1, track2), [4])

    def test_speed_change_frames_private_helper(self):
        track1, track2 = self.trajectories
        # 速度在第 2、3 段之间(分界帧 t=2)从 5 跳到 10:记录分界帧
        # 和它的下一帧,"突变前/突变后"各一张证据帧。
        self.assertEqual(self.filt._speed_change_frames(track1), [2, 3])
        self.assertEqual(self.filt._speed_change_frames(track2), [])

    def test_direction_change_frames_private_helper(self):
        track1, track2 = self.trajectories
        self.assertEqual(self.filt._direction_change_frames(track1), [])
        self.assertEqual(self.filt._direction_change_frames(track2), [])

    def test_occlusion_boundary_frames_private_helper(self):
        track1, track2 = self.trajectories
        self.assertEqual(self.filt._occlusion_boundary_frames(track1, track2), [4])


class TestProximityGate(unittest.TestCase):
    """单目标速度/方向突变只应点亮"突变发生时离得足够近"的目标对,
    不应无差别广播到场上所有目标对(候选边数 = MLLM 调用数,§7 的
    一等公民成本指标)。
    """

    def test_far_away_pair_not_triggered_by_unary_speed_change(self):
        trajectories = make_two_object_fixture()
        # track3 停在很远的地方(距 track1 始终 > 150 像素),它不该
        # 因为 track1 的加速而和 track1 组成候选边。
        track3 = Trajectory(
            track_id=3,
            present=(0, 4),
            per_frame=tuple(
                FramePresence(t=t, box=(200, 0, 210, 10)) for t in range(5)
            ),
        )
        candidates = EventCandidateFilter().find_candidates(trajectories + [track3])
        self.assertEqual([c.edge for c in candidates], [(1, 2)])


class TestOcclusionVisibilityGap(unittest.TestCase):
    def test_nearby_visibility_gap_marks_both_boundary_frames(self):
        """一方观测缺失(真实遮挡的主信号)造成公共可见帧序列出现空洞,
        且消失时刻两目标离得足够近(可能是被对方遮挡)时,空洞两侧的帧
        (消失前最后一帧、重新出现第一帧)都应被记为遮挡边界触发帧。
        """
        track_a = Trajectory(
            track_id=1,
            present=(0, 5),
            per_frame=tuple(FramePresence(t=t, box=(0, 0, 10, 10)) for t in range(6)),
        )
        track_b = Trajectory(
            track_id=2,
            present=(0, 5),
            per_frame=tuple(
                FramePresence(t=t, box=(20, 0, 30, 10)) for t in (0, 1, 4, 5)
            ),
        )
        filt = EventCandidateFilter()
        self.assertEqual(filt._occlusion_boundary_frames(track_a, track_b), [1, 4])

    def test_far_away_visibility_gap_is_gated_out(self):
        """观测缺失也可能只是 tracker 在别处单纯跟丢:消失时刻两目标
        相距很远("被对方遮挡"不成立)时,空洞边界帧应被邻近度门控
        排除,不产生触发帧。
        """
        track_a = Trajectory(
            track_id=1,
            present=(0, 5),
            per_frame=tuple(FramePresence(t=t, box=(0, 0, 10, 10)) for t in range(6)),
        )
        track_b = Trajectory(
            track_id=2,
            present=(0, 5),
            per_frame=tuple(
                FramePresence(t=t, box=(100, 0, 110, 10)) for t in (0, 1, 4, 5)
            ),
        )
        filt = EventCandidateFilter()
        self.assertEqual(filt._occlusion_boundary_frames(track_a, track_b), [])

    def test_tracker_blink_does_not_broadcast_candidate_edges(self):
        """一个目标被跟丢一次,不应让它和场上所有(距离很远的)目标
        都组成候选边——候选边数量就是 MLLM 调用次数。
        """
        track1 = Trajectory(
            track_id=1,
            present=(0, 5),
            per_frame=tuple(FramePresence(t=t, box=(0, 0, 10, 10)) for t in range(6)),
        )
        track2 = Trajectory(
            track_id=2,
            present=(0, 5),
            per_frame=tuple(
                FramePresence(t=t, box=(100, 0, 110, 10)) for t in (0, 1, 4, 5)
            ),
        )
        track3 = Trajectory(
            track_id=3,
            present=(0, 5),
            per_frame=tuple(
                FramePresence(t=t, box=(300, 0, 310, 10)) for t in range(6)
            ),
        )
        candidates = EventCandidateFilter().find_candidates([track1, track2, track3])
        self.assertEqual(candidates, [])


class TestEventCandidateFilterSingleObject(unittest.TestCase):
    def test_no_candidates_without_a_pair(self):
        trajectories = make_single_object_fixture()
        candidates = EventCandidateFilter().find_candidates(trajectories)
        self.assertEqual(candidates, [])


def _static_traj(track_id: int, x: float, n: int = 5, size: float = 10.0) -> Trajectory:
    return Trajectory(
        track_id=track_id,
        present=(0, n - 1),
        per_frame=tuple(
            FramePresence(t=t, box=(x, 0.0, x + size, size)) for t in range(n)
        ),
    )


class TestProximityTrigger(unittest.TestCase):
    """邻近状态跳变触发器(默认关闭,显式开启后覆盖交谈类无事件交互)。"""

    def test_disabled_by_default_keeps_stage0_semantics(self):
        # 两个静止目标始终相距 20(< 默认 gate 50):没有接触/突变/遮挡,
        # 默认配置下不产生任何候选——Stage-0 行为保持不变。
        trajectories = [_static_traj(1, 0.0), _static_traj(2, 20.0)]
        self.assertEqual(EventCandidateFilter().find_candidates(trajectories), [])

    def test_always_near_pair_triggers_on_first_common_frame(self):
        # 开启后,全程贴在一起的一对(面对面交谈的抽象)在公共可见的
        # 第一帧被点亮。
        trajectories = [_static_traj(1, 0.0), _static_traj(2, 20.0)]
        candidates = EventCandidateFilter(proximity_trigger=True).find_candidates(
            trajectories
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].triggers, ("proximity",))
        self.assertEqual(candidates[0].candidate_frames, (0,))

    def test_transition_records_frames_on_both_sides(self):
        # track1 逐帧靠近静止的 track2:距离 60, 45, 30(gate=50,
        # 状态在第 0->1 帧之间翻转)-> 记录翻转前后两帧 (0, 1)。
        track1 = Trajectory(
            track_id=1,
            present=(0, 2),
            per_frame=(
                FramePresence(t=0, box=(0.0, 0.0, 10.0, 10.0)),
                FramePresence(t=1, box=(15.0, 0.0, 25.0, 10.0)),
                FramePresence(t=2, box=(30.0, 0.0, 40.0, 10.0)),
            ),
        )
        track2 = _static_traj(2, 60.0, n=3)
        filt = EventCandidateFilter(proximity_trigger=True)
        self.assertEqual(filt._proximity_transition_frames(track1, track2), [0, 1])

    def test_far_pair_stays_silent(self):
        trajectories = [_static_traj(1, 0.0), _static_traj(2, 500.0)]
        candidates = EventCandidateFilter(proximity_trigger=True).find_candidates(
            trajectories
        )
        self.assertEqual(candidates, [])


class TestAdaptiveProximityGate(unittest.TestCase):
    def test_scales_with_median_box_diagonal(self):
        # 100x220 的"真人"框:对角线 ≈ 241.66,gate = 1.5 倍 ≈ 362.49。
        # 混入的小框只有 1 个观测,4 个大框观测占多数 -> 中位数取大框。
        trajectories = [
            _static_traj(1, 0.0, n=1, size=1.0),  # 故意混入一个小框
            Trajectory(
                track_id=2,
                present=(0, 1),
                per_frame=tuple(
                    FramePresence(t=t, box=(0.0, 0.0, 100.0, 220.0)) for t in range(2)
                ),
            ),
            Trajectory(
                track_id=3,
                present=(0, 1),
                per_frame=tuple(
                    FramePresence(t=t, box=(0.0, 0.0, 100.0, 220.0)) for t in range(2)
                ),
            ),
        ]
        gate = adaptive_proximity_gate(trajectories)
        self.assertAlmostEqual(gate, 1.5 * (100.0**2 + 220.0**2) ** 0.5, places=3)

    def test_floor_applies_to_tiny_boxes_and_empty_input(self):
        self.assertEqual(adaptive_proximity_gate([]), 50.0)
        tiny = [_static_traj(1, 0.0, size=1.0)]
        self.assertEqual(adaptive_proximity_gate(tiny), 50.0)


if __name__ == "__main__":
    unittest.main()
