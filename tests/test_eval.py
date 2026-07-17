from __future__ import annotations

import unittest

from smot.eval import (
    COARSE_MAP,
    aggregate_cost,
    direction_accuracy,
    evaluate,
    instance_coverage,
    match_interactions,
)


def _interaction(subject_id, object_id, label, time_span=(0, 4)):
    """按 §5 schema 的 JSON 形状手搓一条交互断言(评测只关心这些键)。"""
    return {
        "subject_id": subject_id,
        "object_id": object_id,
        "predicate": label,
        "canonical_label": label,
        "time_span": list(time_span),
        "evidence_frames": [0],
        "direction": "subj->obj",
        "confidence": 1.0,
        "type": "interaction",
    }


def _instance(track_id, time_span):
    return {
        "track_id": track_id,
        "caption": f"track {track_id}",
        "time_span": list(time_span),
        "evidence_frames": [time_span[0]],
        "type": "instance",
    }


class TestMatchInteractions(unittest.TestCase):
    def test_identical_is_perfect(self):
        gold = [_interaction(1, 2, "approach")]
        prf = match_interactions(gold, gold)
        self.assertEqual((prf.precision, prf.recall, prf.f1), (1.0, 1.0, 1.0))
        self.assertEqual(prf.n_matched, 1)

    def test_flipped_direction_misses_under_strict(self):
        pred = [_interaction(2, 1, "approach")]
        gold = [_interaction(1, 2, "approach")]
        prf = match_interactions(pred, gold, require_direction=True)
        self.assertEqual(prf.n_matched, 0)
        self.assertEqual(prf.f1, 0.0)
        # 无序匹配下同一条断言可以配上。
        prf_undirected = match_interactions(pred, gold, require_direction=False)
        self.assertEqual(prf_undirected.n_matched, 1)

    def test_synonym_map_merges_labels(self):
        pred = [_interaction(1, 2, "approach")]
        gold = [_interaction(1, 2, "near")]
        self.assertEqual(match_interactions(pred, gold).n_matched, 0)
        merged = match_interactions(pred, gold, label_map={"near": "approach"})
        self.assertEqual(merged.n_matched, 1)

    def test_coarse_map_merges_further(self):
        pred = [_interaction(1, 2, "approach")]
        gold = [_interaction(1, 2, "follow")]
        self.assertEqual(match_interactions(pred, gold).n_matched, 0)
        coarse = match_interactions(pred, gold, label_map=COARSE_MAP)
        self.assertEqual(coarse.n_matched, 1)

    def test_duplicate_assertions_matched_as_multiset(self):
        # 两条相同的 pred 只能"吃掉"一条 gold,不能重复计分。
        pred = [_interaction(1, 2, "approach")] * 2
        gold = [_interaction(1, 2, "approach")]
        prf = match_interactions(pred, gold)
        self.assertEqual(prf.n_matched, 1)
        self.assertEqual(prf.precision, 0.5)
        self.assertEqual(prf.recall, 1.0)


class TestDirectionAccuracy(unittest.TestCase):
    def test_identical_is_one(self):
        gold = [_interaction(1, 2, "approach")]
        self.assertEqual(direction_accuracy(gold, gold), 1.0)

    def test_flipped_direction_is_zero(self):
        pred = [_interaction(2, 1, "approach")]
        gold = [_interaction(1, 2, "approach")]
        self.assertEqual(direction_accuracy(pred, gold), 0.0)

    def test_no_undirected_match_returns_none(self):
        pred = [_interaction(1, 2, "approach")]
        gold = [_interaction(3, 4, "approach")]
        self.assertIsNone(direction_accuracy(pred, gold))


class TestInstanceCoverage(unittest.TestCase):
    def test_partial_coverage_and_time_iou(self):
        pred = [_instance(1, (0, 4))]
        gold = [_instance(1, (2, 6)), _instance(2, (0, 4))]
        cov = instance_coverage(pred, gold)
        self.assertEqual(cov["track_coverage"], 0.5)
        # [0,4] 与 [2,6]:交集帧 2..4 共 3 帧,并集帧 0..6 共 7 帧。
        self.assertAlmostEqual(cov["mean_time_iou"], 3 / 7)


class TestAggregateCost(unittest.TestCase):
    def test_total_and_mean(self):
        payloads = [
            {"cost": {"n_vlm_calls": 4, "n_key_frames": 11}},
            {"cost": {"n_vlm_calls": 6, "n_key_frames": 9}},
        ]
        agg = aggregate_cost(payloads)
        self.assertEqual(agg["n_videos"], 2)
        self.assertEqual(agg["total"], {"n_key_frames": 20, "n_vlm_calls": 10})
        self.assertEqual(agg["mean"], {"n_key_frames": 10.0, "n_vlm_calls": 5.0})


class TestEvaluateEndToEnd(unittest.TestCase):
    def _payload(self, interactions, instances):
        return {
            "instances": instances,
            "interactions": interactions,
            "video": {"summary": "s", "involved_ids": [1, 2], "type": "video"},
            "cost": {"n_vlm_calls": 4},
        }

    def test_self_evaluation_is_perfect(self):
        payload = self._payload(
            [_interaction(1, 2, "approach")], [_instance(1, (0, 4)), _instance(2, (0, 4))]
        )
        report = evaluate(payload, payload)
        for tier in ("strict", "synonym_merged", "coarse"):
            self.assertEqual(report["interaction_f1"][tier]["f1"], 1.0)
        self.assertEqual(report["direction_accuracy"], 1.0)
        self.assertEqual(report["instance"]["track_coverage"], 1.0)
        self.assertEqual(report["instance"]["mean_time_iou"], 1.0)
        self.assertEqual(report["cost"]["total"]["n_vlm_calls"], 4)

    def test_multi_video_micro_average(self):
        # 视频 1 完全命中,视频 2 方向颠倒:strict 层 micro F1 = 0.5,
        # 方向准确率 = 1/2(两条无序匹配里一条方向对)。
        good = self._payload([_interaction(1, 2, "approach")], [_instance(1, (0, 4))])
        flipped_pred = self._payload(
            [_interaction(2, 1, "approach")], [_instance(1, (0, 4))]
        )
        gold2 = self._payload([_interaction(1, 2, "approach")], [_instance(1, (0, 4))])
        report = evaluate([good, flipped_pred], [good, gold2])
        self.assertEqual(report["interaction_f1"]["strict"]["f1"], 0.5)
        self.assertEqual(report["direction_accuracy"], 0.5)
        # coarse 层不要求方向,两条都命中。
        self.assertEqual(report["interaction_f1"]["coarse"]["f1"], 1.0)

    def test_mismatched_video_counts_raise(self):
        payload = self._payload([], [])
        with self.assertRaises(ValueError):
            evaluate([payload], [payload, payload])


if __name__ == "__main__":
    unittest.main()
