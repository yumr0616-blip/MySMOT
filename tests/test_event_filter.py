from __future__ import annotations

import unittest

from smot.event_filter import EventCandidateFilter
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
        self.assertEqual(candidate.candidate_frames, (3, 4))
        self.assertEqual(
            set(candidate.triggers), {"contact", "speed_change", "occlusion_boundary"}
        )
        self.assertNotIn("direction_change", candidate.triggers)

    def test_contact_frames_private_helper(self):
        track1, track2 = self.trajectories
        self.assertEqual(self.filt._contact_frames(track1, track2), [4])

    def test_speed_change_frames_private_helper(self):
        track1, track2 = self.trajectories
        self.assertEqual(self.filt._speed_change_frames(track1), [3])
        self.assertEqual(self.filt._speed_change_frames(track2), [])

    def test_direction_change_frames_private_helper(self):
        track1, track2 = self.trajectories
        self.assertEqual(self.filt._direction_change_frames(track1), [])
        self.assertEqual(self.filt._direction_change_frames(track2), [])

    def test_occlusion_boundary_frames_private_helper(self):
        track1, track2 = self.trajectories
        self.assertEqual(self.filt._occlusion_boundary_frames(track1, track2), [4])


class TestEventCandidateFilterSingleObject(unittest.TestCase):
    def test_no_candidates_without_a_pair(self):
        trajectories = make_single_object_fixture()
        candidates = EventCandidateFilter().find_candidates(trajectories)
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
