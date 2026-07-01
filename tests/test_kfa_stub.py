from __future__ import annotations

import unittest

from smot.event_filter import EventCandidate
from smot.kfa import NoOpPairwiseKFA, NoOpUnaryKFA
from tests.fixtures import make_two_object_fixture


class TestNoOpUnaryKFA(unittest.TestCase):
    def setUp(self):
        self.kfa = NoOpUnaryKFA()
        self.track1, _ = make_two_object_fixture()

    def test_respects_top_k(self):
        selection = self.kfa.select(self.track1.track_id, list(self.track1.per_frame), top_k=2)
        self.assertLessEqual(len(selection.key_frames), 2)

    def test_frames_are_in_bounds(self):
        valid_ts = {fp.t for fp in self.track1.per_frame}
        selection = self.kfa.select(self.track1.track_id, list(self.track1.per_frame), top_k=3)
        self.assertTrue(set(selection.key_frames).issubset(valid_ts))

    def test_returns_all_frames_when_fewer_than_top_k(self):
        selection = self.kfa.select(self.track1.track_id, list(self.track1.per_frame), top_k=100)
        self.assertEqual(len(selection.key_frames), len(self.track1.per_frame))

    def test_soft_token_is_none(self):
        selection = self.kfa.select(self.track1.track_id, list(self.track1.per_frame), top_k=2)
        self.assertIsNone(selection.soft_token)


class TestNoOpPairwiseKFA(unittest.TestCase):
    def setUp(self):
        self.kfa = NoOpPairwiseKFA()
        self.candidate = EventCandidate(
            edge=(1, 2), candidate_frames=(3, 4), triggers=("contact",)
        )

    def test_returns_candidate_frames(self):
        selection = self.kfa.select(self.candidate.edge, self.candidate, top_k=8)
        self.assertEqual(selection.key_frames, (3, 4))

    def test_respects_top_k(self):
        selection = self.kfa.select(self.candidate.edge, self.candidate, top_k=1)
        self.assertEqual(selection.key_frames, (3,))

    def test_soft_token_is_none(self):
        selection = self.kfa.select(self.candidate.edge, self.candidate, top_k=8)
        self.assertIsNone(selection.soft_token)


if __name__ == "__main__":
    unittest.main()
