"""smot.track_matching 的单元测试(纯 stdlib,几何构造 + 贪心匹配)。"""
from __future__ import annotations

import unittest

from smot.track_matching import match_tracks, remap_assertion_ids
from smot.types import FramePresence, Trajectory


def _traj(track_id: int, boxes: dict[int, tuple[float, float, float, float]]) -> Trajectory:
    ts = sorted(boxes)
    return Trajectory(
        track_id=track_id,
        present=(ts[0], ts[-1]),
        per_frame=tuple(FramePresence(t=t, box=boxes[t]) for t in ts),
    )


class MatchTracksTest(unittest.TestCase):
    def test_identity_match_when_pred_equals_gold(self):
        """StubTracker 场景的退化情形:预测轨迹就是 gold 轨迹,id 也
        相同——匹配必须是恒等映射,IoU 恰好 1.0(这是迁移到真实跟踪器
        前的现状,必须先在这个基线上验证不回归)。"""
        gold = [
            _traj(1, {1: (0, 0, 10, 10), 2: (1, 1, 11, 11)}),
            _traj(2, {1: (50, 50, 60, 60), 2: (51, 51, 61, 61)}),
        ]
        result = match_tracks(gold, gold)
        self.assertEqual(result.id_map, {1: 1, 2: 2})
        self.assertEqual(result.unmatched_pred, ())
        self.assertEqual(result.unmatched_gold, ())
        for iou_val in result.matched_ious.values():
            self.assertAlmostEqual(iou_val, 1.0)

    def test_matches_across_different_ids(self):
        """真实跟踪器场景:预测 track_id 与 gold 完全不是同一套编号,
        但空间轨迹对应——必须靠 IoU 而不是 id 找到正确配对。"""
        gold = [
            _traj(1, {1: (0, 0, 10, 10), 2: (1, 1, 11, 11)}),
            _traj(2, {1: (50, 50, 60, 60), 2: (51, 51, 61, 61)}),
        ]
        # 预测用完全不同的 id(7, 9),且顺序也反过来
        pred = [
            _traj(9, {1: (50, 50, 60, 60), 2: (51, 51, 61, 61)}),  # 对应 gold 2
            _traj(7, {1: (0, 0, 10, 10), 2: (1, 1, 11, 11)}),  # 对应 gold 1
        ]
        result = match_tracks(pred, gold)
        self.assertEqual(result.id_map, {9: 2, 7: 1})
        self.assertEqual(result.unmatched_pred, ())
        self.assertEqual(result.unmatched_gold, ())

    def test_hallucinated_pred_track_is_unmatched_fp(self):
        """预测多出一条 gold 没有的轨迹(跟踪器误检)——必须落进
        unmatched_pred,不能被强行凑给某个不相关的 gold 轨迹。"""
        gold = [_traj(1, {1: (0, 0, 10, 10)})]
        pred = [
            _traj(1, {1: (0, 0, 10, 10)}),
            _traj(2, {1: (500, 500, 510, 510)}),  # 远处一条不相关的幻觉轨迹
        ]
        result = match_tracks(pred, gold)
        self.assertEqual(result.id_map, {1: 1})
        self.assertEqual(result.unmatched_pred, (2,))
        self.assertEqual(result.unmatched_gold, ())

    def test_missed_gold_track_is_unmatched_fn(self):
        """gold 里有一条轨迹跟踪器完全没跟到——必须落进
        unmatched_gold,不能被强行凑给某个不相关的预测轨迹。"""
        gold = [
            _traj(1, {1: (0, 0, 10, 10)}),
            _traj(2, {1: (500, 500, 510, 510)}),
        ]
        pred = [_traj(1, {1: (0, 0, 10, 10)})]
        result = match_tracks(pred, gold)
        self.assertEqual(result.id_map, {1: 1})
        self.assertEqual(result.unmatched_gold, (2,))

    def test_below_threshold_overlap_not_matched(self):
        """轨迹级 IoU 达不到阈值:哪怕是当前最高分也不接受,两边都算
        没匹配上,而不是牵强凑一对。"""
        gold = [_traj(1, {1: (0, 0, 10, 10), 2: (0, 0, 10, 10), 3: (0, 0, 10, 10)})]
        # 只在 3 帧里的 1 帧有观测,且那一帧刚好完美重叠——并集分母把
        # 这个"运气好的单帧"拉到阈值以下(1/3 ≈ 0.33 < 0.5 默认阈值)。
        pred = [_traj(9, {1: (0, 0, 10, 10)})]
        result = match_tracks(pred, gold, iou_threshold=0.5)
        self.assertEqual(result.id_map, {})
        self.assertEqual(result.unmatched_pred, (9,))
        self.assertEqual(result.unmatched_gold, (1,))

    def test_greedy_prefers_higher_score_pair_over_alternative(self):
        """贪心的核心行为:A 和 B 都能跟 X 配上,但 A-X 分数更高——
        A 应该拿下 X,把 B 挤成未匹配,而不是任意分配。"""
        gold_x = _traj(100, {1: (0, 0, 10, 10)})
        pred_a = _traj(1, {1: (0, 0, 10, 10)})  # 与 X 完美重叠,IoU=1.0
        pred_b = _traj(2, {1: (0, 0, 8, 8)})  # 与 X 部分重叠,IoU 更低但仍≥阈值
        result = match_tracks([pred_a, pred_b], [gold_x], iou_threshold=0.3)
        self.assertEqual(result.id_map, {1: 100})
        self.assertEqual(result.unmatched_pred, (2,))

    def test_empty_inputs(self):
        result = match_tracks([], [])
        self.assertEqual(result.id_map, {})
        self.assertEqual(result.unmatched_pred, ())
        self.assertEqual(result.unmatched_gold, ())


class TrackMatchResultStatsTest(unittest.TestCase):
    def test_stats_precision_recall_f1(self):
        gold = [_traj(i, {1: (0, 0, 10, 10)}) for i in range(1, 4)]  # 3 条 gold
        pred = [
            _traj(1, {1: (0, 0, 10, 10)}),
            _traj(2, {1: (0, 0, 10, 10)}),
        ]
        # 手工构造:1 条 gold 匹配失败(id=3),另两条各自被一条预测匹配上
        gold_matchable = [gold[0], gold[1]]
        result = match_tracks(pred, gold_matchable + [gold[2]])
        stats = result.stats()
        self.assertEqual(stats["n_matched"], 2)
        self.assertEqual(stats["n_pred"], 2)
        self.assertEqual(stats["n_gold"], 3)
        self.assertAlmostEqual(stats["precision"], 1.0)
        self.assertAlmostEqual(stats["recall"], 2 / 3)
        self.assertAlmostEqual(stats["mean_matched_iou"], 1.0)

    def test_stats_all_zero_on_empty(self):
        result = match_tracks([], [])
        stats = result.stats()
        self.assertEqual(stats["precision"], 0.0)
        self.assertEqual(stats["recall"], 0.0)
        self.assertEqual(stats["f1"], 0.0)
        self.assertEqual(stats["mean_matched_iou"], 0.0)


class RemapAssertionIdsTest(unittest.TestCase):
    def test_remaps_all_id_fields(self):
        assertions = [
            {"track_id": 9, "caption": "x"},
            {"subject_id": 9, "object_id": 7, "predicate": "talk"},
        ]
        remapped = remap_assertion_ids(assertions, {9: 1, 7: 2})
        self.assertEqual(remapped[0]["track_id"], 1)
        self.assertEqual(remapped[1]["subject_id"], 1)
        self.assertEqual(remapped[1]["object_id"], 2)

    def test_unmapped_id_left_unchanged(self):
        """不在 id_map 里的 id(unmatched_pred,已判定为 FP)原样保留,
        不静默丢弃——它自然无法匹配任何 gold id,交给下游断言匹配。"""
        assertions = [{"track_id": 99, "caption": "hallucinated"}]
        remapped = remap_assertion_ids(assertions, {1: 1})
        self.assertEqual(remapped[0]["track_id"], 99)

    def test_does_not_mutate_input(self):
        original = [{"track_id": 9}]
        remap_assertion_ids(original, {9: 1})
        self.assertEqual(original[0]["track_id"], 9)


if __name__ == "__main__":
    unittest.main()
