from __future__ import annotations

import unittest

from smot.motion_facts import MotionFactExtractor
from smot.types import FactType
from tests.fixtures import make_single_object_fixture, make_two_object_fixture


def _facts_by_type(facts, fact_type):
    return [f for f in facts if f.type == fact_type]


class TestMotionFactExtractorTwoObject(unittest.TestCase):
    def setUp(self):
        self.trajectories = make_two_object_fixture()
        self.extractor = MotionFactExtractor()
        self.track1, self.track2 = self.trajectories

    def test_presence(self):
        facts = self.extractor.extract_instance_facts(self.track1)
        presence = _facts_by_type(facts, FactType.PRESENCE)[0]
        self.assertEqual(presence.value, (0, 4))
        self.assertEqual(presence.t_span, (0, 4))

    def test_net_motion_moving_object(self):
        facts = self.extractor.extract_instance_facts(self.track1)
        net_motion = _facts_by_type(facts, FactType.NET_MOTION)[0]
        self.assertEqual(net_motion.value, (30.0, 0.0))

    def test_net_motion_stationary_object(self):
        facts = self.extractor.extract_instance_facts(self.track2)
        net_motion = _facts_by_type(facts, FactType.NET_MOTION)[0]
        self.assertEqual(net_motion.value, (0.0, 0.0))

    def test_speed_moving_object(self):
        facts = self.extractor.extract_instance_facts(self.track1)
        speed = _facts_by_type(facts, FactType.SPEED)[0]
        self.assertAlmostEqual(speed.value, 7.5)

    def test_speed_stationary_object(self):
        facts = self.extractor.extract_instance_facts(self.track2)
        speed = _facts_by_type(facts, FactType.SPEED)[0]
        self.assertAlmostEqual(speed.value, 0.0)

    def test_proximity(self):
        facts = self.extractor.extract_pair_facts(self.track1, self.track2)
        proximity = _facts_by_type(facts, FactType.PROXIMITY)[0]
        self.assertAlmostEqual(proximity.value["min"], 8.0)
        self.assertAlmostEqual(proximity.value["mean"], 25.0)
        self.assertEqual(proximity.t_span, (0, 4))

    def test_approach(self):
        facts = self.extractor.extract_pair_facts(self.track1, self.track2)
        approach = _facts_by_type(facts, FactType.APPROACH)[0]
        self.assertEqual(approach.value, "approaching")

    def test_extract_combines_instance_and_pair_facts(self):
        all_facts = self.extractor.extract(self.trajectories)
        # 3 instance facts each (presence/net_motion/speed) x2 tracks + 2 pair facts
        self.assertEqual(len(all_facts), 3 * 2 + 2)

    def test_pair_scope_is_order_independent(self):
        # scope 键用排序后的 id,与轨迹列表顺序解耦——评测器/外部调用
        # 按 id 构造 scope 键时才不会失配。
        fwd = {f.scope for f in self.extractor.extract(self.trajectories)}
        rev = {f.scope for f in self.extractor.extract(list(reversed(self.trajectories)))}
        self.assertEqual(fwd, rev)
        self.assertIn("pair:1,2", fwd)

    def test_embed_temporal_components_are_normalized(self):
        # embed = (type_index, norm_value, t_start_norm, t_end_norm),
        # 时间分量按视频最大帧号(fixture 里为 4)归一化到 [0, 1]。
        all_facts = self.extractor.extract(self.trajectories)
        presence = [
            f
            for f in _facts_by_type(all_facts, FactType.PRESENCE)
            if f.scope == "instance:1"
        ][0]
        self.assertEqual(presence.embed, (4.0, 4.0, 0.0, 1.0))
        for fact in all_facts:
            self.assertGreaterEqual(fact.embed[2], 0.0)
            self.assertLessEqual(fact.embed[3], 1.0)


class TestMotionFactExtractorSingleObject(unittest.TestCase):
    def setUp(self):
        self.trajectories = make_single_object_fixture()
        self.extractor = MotionFactExtractor()

    def test_zero_motion(self):
        facts = self.extractor.extract_instance_facts(self.trajectories[0])
        net_motion = _facts_by_type(facts, FactType.NET_MOTION)[0]
        speed = _facts_by_type(facts, FactType.SPEED)[0]
        self.assertEqual(net_motion.value, (0.0, 0.0))
        self.assertAlmostEqual(speed.value, 0.0)

    def test_no_pair_facts_with_single_trajectory(self):
        all_facts = self.extractor.extract(self.trajectories)
        self.assertEqual(len(_facts_by_type(all_facts, FactType.PROXIMITY)), 0)


if __name__ == "__main__":
    unittest.main()
