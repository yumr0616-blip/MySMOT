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
