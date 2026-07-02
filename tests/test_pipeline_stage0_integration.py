from __future__ import annotations

import json
import unittest

from smot.pipeline import Pipeline
from smot.tracker import StubTracker, VideoHandle
from tests.fixtures import make_two_object_fixture


class TestPipelineStage0Integration(unittest.TestCase):
    def setUp(self):
        self.trajectories = make_two_object_fixture()
        self.pipeline = Pipeline(tracker=StubTracker(self.trajectories))
        self.video = VideoHandle(path="synthetic://two_object", num_frames=5)

    def test_runs_with_fully_default_stage0_wiring(self):
        result = self.pipeline.run(self.video)
        self.assertEqual(len(result.instances), 2)
        self.assertGreaterEqual(len(result.interactions), 1)
        self.assertEqual(set(result.video.involved_ids), {1, 2})

    def test_all_evidence_frames_are_valid_indices(self):
        result = self.pipeline.run(self.video)
        valid_ts = {fp.t for traj in self.trajectories for fp in traj.per_frame}
        for assertion in result.instances:
            self.assertTrue(set(assertion.evidence_frames).issubset(valid_ts))
        for assertion in result.interactions:
            self.assertTrue(set(assertion.evidence_frames).issubset(valid_ts))

    def test_result_round_trips_through_json(self):
        result = self.pipeline.run(self.video)
        payload = json.loads(json.dumps(result.to_json_dict()))
        self.assertIn("instances", payload)
        self.assertIn("interactions", payload)
        self.assertIn("video", payload)

    def test_interaction_predicate_is_canonicalized(self):
        result = self.pipeline.run(self.video)
        interaction = result.interactions[0]
        self.assertEqual(interaction.canonical_label, "approach")


if __name__ == "__main__":
    unittest.main()
