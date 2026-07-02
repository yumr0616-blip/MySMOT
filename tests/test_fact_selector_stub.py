from __future__ import annotations

import unittest

from smot.fact_selector import DeterministicFactSelector, SelectionContext
from smot.motion_facts import MotionFactExtractor
from tests.fixtures import make_two_object_fixture


class TestDeterministicFactSelector(unittest.TestCase):
    def setUp(self):
        self.trajectories = make_two_object_fixture()
        self.facts = MotionFactExtractor().extract(self.trajectories)
        self.selector = DeterministicFactSelector()

    def test_selects_only_facts_in_scope(self):
        selection = self.selector.select(
            self.facts, SelectionContext(scope="instance:1", top_k=8)
        )
        self.assertTrue(all(f.scope == "instance:1" for f in selection.selected_facts))

    def test_respects_top_k(self):
        selection = self.selector.select(
            self.facts, SelectionContext(scope="instance:1", top_k=1)
        )
        self.assertEqual(len(selection.selected_facts), 1)

    def test_priority_order_pair_scope(self):
        selection = self.selector.select(
            self.facts, SelectionContext(scope="pair:1,2", top_k=8)
        )
        types = [f.type.value for f in selection.selected_facts]
        # proximity before approach, per FACT_TYPE_ORDER
        self.assertEqual(types, ["proximity", "approach"])

    def test_soft_token_is_none_in_stage0(self):
        selection = self.selector.select(
            self.facts, SelectionContext(scope="instance:1", top_k=8)
        )
        self.assertIsNone(selection.soft_token)

    def test_video_scope_selects_across_all_facts(self):
        selection = self.selector.select(
            self.facts, SelectionContext(scope="video", top_k=100)
        )
        self.assertEqual(len(selection.selected_facts), len(self.facts))

    def test_text_is_rendered(self):
        selection = self.selector.select(
            self.facts, SelectionContext(scope="instance:1", top_k=8)
        )
        self.assertIsInstance(selection.text, str)
        self.assertGreater(len(selection.text), 0)


if __name__ == "__main__":
    unittest.main()
