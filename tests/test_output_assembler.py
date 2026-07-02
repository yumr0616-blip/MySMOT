from __future__ import annotations

import json
import unittest

from smot.output_assembler import OutputAssembler


class TestOutputAssembler(unittest.TestCase):
    def setUp(self):
        self.assembler = OutputAssembler()

    def test_assemble_instance(self):
        assertion = self.assembler.assemble_instance(
            track_id=1,
            mllm_text="track_id=1 is present and moving.",
            time_span=(0, 4),
            evidence_frames=(0, 2, 4),
        )
        self.assertEqual(assertion.track_id, 1)
        self.assertEqual(assertion.caption, "track_id=1 is present and moving.")
        self.assertEqual(assertion.time_span, (0, 4))
        self.assertEqual(assertion.evidence_frames, (0, 2, 4))
        self.assertEqual(assertion.type, "instance")

    def test_assemble_interaction_maps_known_predicate(self):
        assertion = self.assembler.assemble_interaction(
            subject_id=1,
            object_id=2,
            mllm_text="subject_id=1 approaches object_id=2.",
            time_span=(3, 4),
            evidence_frames=(3, 4),
        )
        self.assertEqual(assertion.predicate, "approaches")
        self.assertEqual(assertion.canonical_label, "approach")
        self.assertEqual(assertion.subject_id, 1)
        self.assertEqual(assertion.object_id, 2)
        self.assertEqual(assertion.direction, "subj->obj")

    def test_assemble_interaction_unmapped_predicate_falls_back_to_raw(self):
        assertion = self.assembler.assemble_interaction(
            subject_id=1,
            object_id=2,
            mllm_text="the two objects juggle nearby",
            time_span=(0, 1),
            evidence_frames=(0, 1),
        )
        self.assertEqual(assertion.predicate, "the two objects juggle nearby")
        self.assertEqual(assertion.canonical_label, "the two objects juggle nearby")

    def test_assemble_interaction_verified_direction_keeps_confidence(self):
        assertion = self.assembler.assemble_interaction(
            subject_id=1,
            object_id=2,
            mllm_text="subject_id=1 approaches object_id=2.",
            time_span=(3, 4),
            evidence_frames=(3, 4),
        )
        self.assertEqual(assertion.confidence, 1.0)

    def test_assemble_interaction_swaps_when_mllm_states_reverse_direction(self):
        # MLLM 明确说方向相反时,以模型判断为准交换 subject/object,
        # 而不是沿用上游候选边的下标顺序。
        assertion = self.assembler.assemble_interaction(
            subject_id=1,
            object_id=2,
            mllm_text="subject_id=2 approaches object_id=1.",
            time_span=(3, 4),
            evidence_frames=(3, 4),
        )
        self.assertEqual(assertion.subject_id, 2)
        self.assertEqual(assertion.object_id, 1)
        self.assertEqual(assertion.confidence, 1.0)

    def test_assemble_interaction_unverified_direction_lowers_confidence(self):
        # 文本里解析不出 subject_id/object_id 时,方向只是下标启发式,
        # 置信度应被压到标记值以便下游区分。
        assertion = self.assembler.assemble_interaction(
            subject_id=1,
            object_id=2,
            mllm_text="the two objects juggle nearby",
            time_span=(0, 1),
            evidence_frames=(0, 1),
        )
        self.assertEqual(assertion.confidence, 0.5)

    def test_injected_canonical_map_is_used(self):
        # §7 分层 F1 依赖注入不同粒度的映射表重跑;注入的表必须真正
        # 生效于谓词提取和规范化两个环节。
        assembler = OutputAssembler(canonical_map={"juggles": "juggle"})
        assertion = assembler.assemble_interaction(
            subject_id=1,
            object_id=2,
            mllm_text="subject_id=1 juggles object_id=2.",
            time_span=(0, 1),
            evidence_frames=(0, 1),
        )
        self.assertEqual(assertion.predicate, "juggles")
        self.assertEqual(assertion.canonical_label, "juggle")

    def test_negated_predicate_is_not_extracted_as_affirmative(self):
        assertion = self.assembler.assemble_interaction(
            subject_id=1,
            object_id=2,
            mllm_text="subject_id=1 never approaches object_id=2.",
            time_span=(0, 1),
            evidence_frames=(0, 1),
        )
        # "never approaches" 不能被提取成肯定的 "approaches",
        # 应走整句 fallback。
        self.assertEqual(
            assertion.predicate, "subject_id=1 never approaches object_id=2."
        )

    def test_assemble_video(self):
        assertion = self.assembler.assemble_video(
            mllm_text="Two tracked objects approach each other during the video.",
            involved_ids=(1, 2),
        )
        self.assertEqual(assertion.involved_ids, (1, 2))
        self.assertEqual(assertion.type, "video")

    def test_to_json_dict_round_trips_and_matches_schema_keys(self):
        instance = self.assembler.assemble_instance(1, "text", (0, 1), (0,))
        interaction = self.assembler.assemble_interaction(1, 2, "approaches", (0, 1), (0,))
        video = self.assembler.assemble_video("summary", (1, 2))

        instance_dict = json.loads(json.dumps(instance.to_json_dict()))
        interaction_dict = json.loads(json.dumps(interaction.to_json_dict()))
        video_dict = json.loads(json.dumps(video.to_json_dict()))

        self.assertEqual(
            set(instance_dict),
            {"track_id", "caption", "time_span", "evidence_frames", "type"},
        )
        self.assertEqual(
            set(interaction_dict),
            {
                "subject_id",
                "object_id",
                "predicate",
                "canonical_label",
                "time_span",
                "evidence_frames",
                "direction",
                "confidence",
                "type",
            },
        )
        self.assertEqual(set(video_dict), {"summary", "involved_ids", "type"})


if __name__ == "__main__":
    unittest.main()
