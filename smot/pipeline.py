"""Pipeline orchestrator: wires all §4 modules for a single run.

Every learnable/model-backed component is an optional constructor argument
defaulting to its Stage-0 implementation (deterministic/no-op). Passing real
Stage-1a/1b components (learnable KFA, Fact Selector, Projector, or a real
Tracker/MLLM) later requires no change to this class's signature - that's
the "non-breaking upgrade" contract from §6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from smot.event_filter import EventCandidateFilter
from smot.fact_selector import DeterministicFactSelector, FactSelector, SelectionContext
from smot.kfa import NoOpPairwiseKFA, NoOpUnaryKFA, PairwiseKFA, UnaryKFA
from smot.mllm import MLLMAdapter, MLLMRequest, MockMLLMAdapter
from smot.motion_facts import MotionFactExtractor
from smot.output_assembler import OutputAssembler
from smot.projector import NoOpProjector, Projector
from smot.prompts import build_instance_prompt, build_interaction_prompt, build_video_prompt
from smot.tracker import Tracker, VideoHandle
from smot.types import InstanceAssertion, InteractionAssertion, VideoAssertion


@dataclass
class PipelineConfig:
    top_k_instance_frames: int = 4
    top_k_pair_frames: int = 4
    fact_top_k: int = 6


@dataclass(frozen=True)
class PipelineResult:
    instances: tuple[InstanceAssertion, ...]
    interactions: tuple[InteractionAssertion, ...]
    video: VideoAssertion

    def to_json_dict(self) -> dict:
        return {
            "instances": [a.to_json_dict() for a in self.instances],
            "interactions": [a.to_json_dict() for a in self.interactions],
            "video": self.video.to_json_dict(),
        }


class Pipeline:
    def __init__(
        self,
        tracker: Tracker,
        motion_fact_extractor: Optional[MotionFactExtractor] = None,
        event_filter: Optional[EventCandidateFilter] = None,
        fact_selector: Optional[FactSelector] = None,
        unary_kfa: Optional[UnaryKFA] = None,
        pairwise_kfa: Optional[PairwiseKFA] = None,
        projector: Optional[Projector] = None,
        mllm_adapter: Optional[MLLMAdapter] = None,
        output_assembler: Optional[OutputAssembler] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.tracker = tracker
        self.motion_fact_extractor = motion_fact_extractor or MotionFactExtractor()
        self.event_filter = event_filter or EventCandidateFilter()
        self.fact_selector = fact_selector or DeterministicFactSelector()
        self.unary_kfa = unary_kfa or NoOpUnaryKFA()
        self.pairwise_kfa = pairwise_kfa or NoOpPairwiseKFA()
        self.projector = projector or NoOpProjector()
        self.mllm_adapter = mllm_adapter or MockMLLMAdapter()
        self.output_assembler = output_assembler or OutputAssembler()
        self.config = config or PipelineConfig()

    def run(self, video: VideoHandle) -> PipelineResult:
        trajectories = self.tracker.track(video)
        facts = self.motion_fact_extractor.extract(trajectories)
        event_candidates = self.event_filter.find_candidates(trajectories)

        instances = tuple(
            self._run_instance(traj, facts) for traj in trajectories
        )
        interactions = tuple(
            self._run_interaction(candidate, facts) for candidate in event_candidates
        )
        video_assertion = self._run_video(trajectories, facts)

        return PipelineResult(
            instances=instances, interactions=interactions, video=video_assertion
        )

    def _run_instance(self, traj, facts) -> InstanceAssertion:
        selection = self.fact_selector.select(
            facts, SelectionContext(scope=f"instance:{traj.track_id}", top_k=self.config.fact_top_k)
        )
        kfa_selection = self.unary_kfa.select(
            traj.track_id, list(traj.per_frame), self.config.top_k_instance_frames
        )
        self.projector.project(())
        prompt_text = build_instance_prompt(traj.track_id, selection.text)
        request = MLLMRequest(
            prompt_type="instance",
            transcript_text=prompt_text,
            frame_refs=kfa_selection.key_frames,
        )
        mllm_text = self.mllm_adapter.generate(request)
        return self.output_assembler.assemble_instance(
            track_id=traj.track_id,
            mllm_text=mllm_text,
            time_span=traj.present,
            evidence_frames=kfa_selection.key_frames,
        )

    def _run_interaction(self, candidate, facts) -> InteractionAssertion:
        subject_id, object_id = candidate.edge
        selection = self.fact_selector.select(
            facts,
            SelectionContext(scope=f"pair:{subject_id},{object_id}", top_k=self.config.fact_top_k),
        )
        kfa_selection = self.pairwise_kfa.select(
            candidate.edge, candidate, self.config.top_k_pair_frames
        )
        self.projector.project(())
        prompt_text = build_interaction_prompt(subject_id, object_id, selection.text)
        request = MLLMRequest(
            prompt_type="interaction",
            transcript_text=prompt_text,
            frame_refs=kfa_selection.key_frames,
        )
        mllm_text = self.mllm_adapter.generate(request)
        frames = kfa_selection.key_frames
        time_span = (min(frames), max(frames)) if frames else (0, 0)
        return self.output_assembler.assemble_interaction(
            subject_id=subject_id,
            object_id=object_id,
            mllm_text=mllm_text,
            time_span=time_span,
            evidence_frames=frames,
        )

    def _run_video(self, trajectories, facts) -> VideoAssertion:
        involved_ids = tuple(sorted(t.track_id for t in trajectories))
        selection = self.fact_selector.select(
            facts, SelectionContext(scope="video", top_k=self.config.fact_top_k)
        )
        prompt_text = build_video_prompt(involved_ids, selection.text)
        request = MLLMRequest(prompt_type="video", transcript_text=prompt_text, frame_refs=())
        mllm_text = self.mllm_adapter.generate(request)
        return self.output_assembler.assemble_video(mllm_text, involved_ids)
