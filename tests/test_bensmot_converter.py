"""BenSMOT 转换器的微型 fixture 测试。

在临时目录里伪造一个最小但格式完整的 BenSMOT 序列(MOT gt.txt、
instance_captions.txt、video_caption.txt、interactions.graphml),
锁住转换器的全部格式假设——真实数据到手后如果 probe 发现格式偏差,
改动转换器时这些测试保证已确认的行为不被顺手破坏。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smot.canonical_labels import map_predicate
from smot.datasets.bensmot import (
    build_gold_payloads,
    compute_fact_statistics,
    describe_sequence,
    graphml_to_interactions,
    iter_sequences,
    load_sequence,
    load_split,
    map_names_to_track_ids,
    parse_gt_txt,
    sequence_to_video_handle,
)
from smot.types import FramePresence, Trajectory

# 轨迹 1(man0)向右移动;轨迹 2(woman0)静止;conf 取自 visibility 列。
# 特意包含两类脏数据:重复的 (id=1, frame=1) 行(应保留首行)和
# consider 标志为 0 的 id=3 行(应整行跳过)。
_GT_LINES = """\
1,1,0,0,10,10,1,1,1.0
1,1,99,99,1,1,1,1,1.0
2,1,5,0,10,10,1,1,1.0
3,1,10,0,10,10,1,1,1.0
1,2,38,0,10,10,1,1,0.5
2,2,38,0,10,10,1,1,0.5
3,2,38,0,10,10,1,1,0.5
2,3,100,100,5,5,0,1,1.0
"""

_CAPTIONS = """\
man0: A man walks to the right.
woman0: A woman stands still.
"""

_VIDEO_CAPTION = "A man walks toward a standing woman.\n"

# 最简 GraphML:节点 id 直接就是实例名,边上只有一个 relation 属性。
_GRAPHML = """\
<?xml version="1.0" encoding="utf-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="edge" attr.name="relation" attr.type="string"/>
  <graph edgedefault="directed">
    <node id="man0"/>
    <node id="woman0"/>
    <edge source="man0" target="woman0">
      <data key="d0">walks towards</data>
    </edge>
  </graph>
</graphml>
"""

# 另一种真实可能出现的 GraphML 形态:节点 id 是 n0/n1、实例名放在 name
# 属性里;边同时带数值属性(score)和起止帧属性——用于验证谓词挑选
# 会跳过数值、起止帧会被解析为 time_span。
_GRAPHML_RICH = """\
<?xml version="1.0" encoding="utf-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="k0" for="node" attr.name="name" attr.type="string"/>
  <key id="k1" for="edge" attr.name="relation" attr.type="string"/>
  <key id="k2" for="edge" attr.name="score" attr.type="double"/>
  <key id="k3" for="edge" attr.name="start_frame" attr.type="int"/>
  <key id="k4" for="edge" attr.name="end_frame" attr.type="int"/>
  <graph edgedefault="directed">
    <node id="n0"><data key="k0">man0</data></node>
    <node id="n1"><data key="k0">woman0</data></node>
    <edge source="n0" target="n1">
      <data key="k2">0.93</data>
      <data key="k3">2</data>
      <data key="k4">3</data>
      <data key="k1">pushes</data>
    </edge>
  </graph>
</graphml>
"""


def _write_sequence(
    seq_dir: Path,
    gt: str = _GT_LINES,
    captions: str = _CAPTIONS,
    video_caption: str = _VIDEO_CAPTION,
    graphml: str = _GRAPHML,
) -> Path:
    """在 seq_dir 下伪造一个 BenSMOT 序列目录。"""
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "gt" / "gt.txt").write_text(gt, encoding="utf-8")
    (seq_dir / "instance_captions.txt").write_text(captions, encoding="utf-8")
    (seq_dir / "video_caption.txt").write_text(video_caption, encoding="utf-8")
    (seq_dir / "interactions.graphml").write_text(graphml, encoding="utf-8")
    return seq_dir


class BenSMOTConverterTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        # 模拟真实的 train/<活动类别>/<序列名> 两级嵌套。
        self.seq_dir = _write_sequence(self.root / "train" / "walking" / "seq0001")

    # ------------------------------------------------------------------
    # gt.txt
    # ------------------------------------------------------------------

    def test_parse_gt_boxes_spans_conf(self):
        """xywh->xyxy 转换、present 区间、visibility->conf、脏数据容忍。"""
        trajectories = parse_gt_txt(self.seq_dir / "gt" / "gt.txt")
        self.assertEqual([t.track_id for t in trajectories], [1, 2])  # id=3 被 consider=0 过滤

        traj1, traj2 = trajectories
        self.assertEqual(traj1.present, (1, 3))
        # 重复的 (id=1, frame=1) 行保留首行:框是 (0,0,10,10) 不是 (99,99,...)。
        self.assertEqual(traj1.frame_at(1).box, (0.0, 0.0, 10.0, 10.0))
        # xywh (5,0,10,10) -> xyxy (5,0,15,10)。
        self.assertEqual(traj1.frame_at(2).box, (5.0, 0.0, 15.0, 10.0))
        self.assertEqual(traj1.frame_at(1).conf, 1.0)
        self.assertEqual(traj2.frame_at(1).conf, 0.5)

    def test_parse_gt_bad_line_raises(self):
        bad = self.root / "bad_gt.txt"
        bad.write_text("1,1,not_a_number,0,10,10\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            parse_gt_txt(bad)

    # ------------------------------------------------------------------
    # 实例名映射
    # ------------------------------------------------------------------

    def test_caption_mapping_line_order(self):
        """格式假设 #1:caption 行序 <-> track_id 升序。"""
        mapping = map_names_to_track_ids(["man0", "woman0"], [2, 1])
        self.assertEqual(mapping, {"man0": 1, "woman0": 2})

    def test_caption_count_mismatch_raises(self):
        (self.seq_dir / "instance_captions.txt").write_text(
            "man0: only one line.\n", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            load_sequence(self.seq_dir)

    # ------------------------------------------------------------------
    # interactions.graphml
    # ------------------------------------------------------------------

    def test_graphml_direction_predicate_default_span(self):
        """边方向 = subject->object;无起止帧属性时 span 取 present 交集。"""
        seq = load_sequence(self.seq_dir)
        self.assertEqual(len(seq.interactions), 1)
        assertion = seq.interactions[0]
        self.assertEqual(assertion.subject_id, 1)  # man0 -> track_id 1
        self.assertEqual(assertion.object_id, 2)
        self.assertEqual(assertion.predicate, "walks towards")
        self.assertEqual(assertion.canonical_label, map_predicate("walks towards"))
        self.assertEqual(assertion.time_span, (1, 3))

    def test_graphml_node_data_names_and_explicit_span(self):
        """节点名在 name 属性里、谓词挑选跳过数值属性、起止帧属性生效。"""
        path = self.root / "rich.graphml"
        path.write_text(_GRAPHML_RICH, encoding="utf-8")
        traj = lambda tid: Trajectory(  # noqa: E731 - 测试内的最小构造捷径
            track_id=tid, present=(1, 3), per_frame=(FramePresence(t=1, box=(0, 0, 1, 1)),)
        )
        assertions = graphml_to_interactions(
            path, {"man0": 1, "woman0": 2}, {1: traj(1), 2: traj(2)}
        )
        self.assertEqual(len(assertions), 1)
        self.assertEqual(assertions[0].predicate, "pushes")
        self.assertEqual(assertions[0].time_span, (2, 3))
        self.assertEqual((assertions[0].subject_id, assertions[0].object_id), (1, 2))

    def test_graphml_unknown_node_raises(self):
        graphml = _GRAPHML.replace('source="man0"', 'source="ghost0"')
        (self.seq_dir / "interactions.graphml").write_text(graphml, encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_sequence(self.seq_dir)
        self.assertIn("ghost0", str(ctx.exception))

    # ------------------------------------------------------------------
    # gold payload / 统计 / VideoHandle
    # ------------------------------------------------------------------

    def test_gold_payload_shape_json_ready(self):
        """payload 与 PipelineResult.to_json_dict() 同形且可直接 json.dumps。"""
        seq = load_sequence(self.seq_dir)
        payloads = build_gold_payloads([seq])
        self.assertEqual(len(payloads), 1)
        payload = payloads[0]

        self.assertEqual(payload["sequence"], "walking/seq0001")
        self.assertEqual(len(payload["instances"]), 2)
        first = payload["instances"][0]
        self.assertEqual(first["track_id"], 1)
        self.assertEqual(first["caption"], "A man walks to the right.")
        self.assertEqual(first["time_span"], [1, 3])  # tuple 已拍平为 list
        self.assertEqual(first["type"], "instance")
        self.assertEqual(payload["interactions"][0]["direction"], "subj->obj")
        self.assertEqual(
            payload["video"]["summary"], "A man walks toward a standing woman."
        )
        self.assertEqual(payload["video"]["involved_ids"], [1, 2])
        json.dumps(payload)  # 不抛异常即为可序列化

    def test_fact_statistics(self):
        """速度事实:轨迹 1 恒为 5 像素/帧,轨迹 2 恒为 0 -> mean 2.5 / std 2.5;
        presence 事实:两条轨迹时长都是 2 帧 -> mean 2 / std 0。
        """
        seq = load_sequence(self.seq_dir)
        stats = compute_fact_statistics([seq])
        self.assertEqual(stats["speed"]["n"], 2)
        self.assertAlmostEqual(stats["speed"]["mean"], 2.5)
        self.assertAlmostEqual(stats["speed"]["std"], 2.5)
        self.assertAlmostEqual(stats["presence"]["mean"], 2.0)
        self.assertAlmostEqual(stats["presence"]["std"], 0.0)

    def test_video_handle(self):
        seq = load_sequence(self.seq_dir)
        handle = sequence_to_video_handle(seq)
        self.assertTrue(handle.path.endswith("imgs"))
        # fixture 没有 imgs 目录,num_frames 退化为最大帧号。
        self.assertEqual(handle.num_frames, 3)

    # ------------------------------------------------------------------
    # 目录遍历 / 批量加载
    # ------------------------------------------------------------------

    def test_iter_sequences_nested_and_direct(self):
        found = [p.name for p in iter_sequences(self.root)]
        self.assertEqual(found, ["seq0001"])
        # root 本身就是序列目录时直接返回它。
        direct = list(iter_sequences(self.seq_dir))
        self.assertEqual(direct, [self.seq_dir])

    def test_load_split_skip_errors(self):
        # 再造一个 gt 损坏的序列,验证 skip 模式收集错误、raise 模式中止。
        broken = _write_sequence(
            self.root / "train" / "walking" / "seq0002",
            gt="1,1,broken,0,10,10\n",
        )
        sequences, errors = load_split(self.root, on_error="skip")
        self.assertEqual([s.name for s in sequences], ["walking/seq0001"])
        self.assertEqual(len(errors), 1)
        self.assertIn(str(broken), errors[0][0])
        with self.assertRaises(ValueError):
            load_split(self.root, on_error="raise")

    def test_load_split_limit(self):
        _write_sequence(self.root / "train" / "walking" / "seq0002")
        sequences, _ = load_split(self.root, limit=1)
        self.assertEqual(len(sequences), 1)

    # ------------------------------------------------------------------
    # 格式探查
    # ------------------------------------------------------------------

    def test_describe_sequence_reports_content(self):
        report = describe_sequence(self.seq_dir)
        self.assertIn("walks towards", report)
        self.assertIn("man0", report)
        self.assertIn("track_id", report)

    def test_describe_sequence_never_raises_on_broken_data(self):
        (self.seq_dir / "interactions.graphml").write_text(
            "<not-valid-xml", encoding="utf-8"
        )
        report = describe_sequence(self.seq_dir)  # 不应抛异常
        self.assertIn("解析失败", report)


if __name__ == "__main__":
    unittest.main()
