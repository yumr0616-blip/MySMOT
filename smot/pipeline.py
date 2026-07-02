"""Pipeline 编排器:把 §4 里的所有模块串成一次完整的推理流程。

每一个"可能需要学习/真实模型"的组件在构造函数里都是可选参数,
默认值就是它对应的 Stage-0 实现(确定性或 no-op)。以后升级到
Stage-1a/1b(接入真正可学习的 KFA slot、真正的 tracker/MLLM)时,
只需要把对应的参数换成真实实现传进来,Pipeline 本身的调用方式
(构造函数签名、run() 的调用方式)完全不用改——这就是 §6 里
"非破坏性升级"要求的具体落地方式。
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
    """跑一次 Pipeline 时用到的几个数量控制参数。"""

    top_k_instance_frames: int = 4  # 每个目标最多保留几张证据帧
    top_k_pair_frames: int = 4  # 每条交互边最多保留几张证据帧
    fact_top_k: int = 6  # 每次 Fact Selector 最多选几条事实进 transcript


@dataclass(frozen=True)
class PipelineResult:
    """一次 Pipeline 运行的完整输出:三类断言打包在一起。"""

    instances: tuple[InstanceAssertion, ...]
    interactions: tuple[InteractionAssertion, ...]
    video: VideoAssertion

    def to_json_dict(self) -> dict:
        """把整个结果拍平成一个可以直接 json.dumps 的 dict。"""
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
        # tracker 没有默认值——它必须由调用方显式提供(Stage-0 下通常是
        # StubTracker 注入的 GT/预置轨迹;真正跑真实 tracker 时替换成
        # 真实实现即可,其余组件的默认值都不用变)。
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
        """跑一次完整流程:
        1. 跟踪拿到轨迹
        2. 抽取运动事实
        3. 找候选交互边
        4. 对每个轨迹生成一条 instance 断言
        5. 对每条候选边生成一条 interaction 断言
        6. 生成一条 video 级概括断言
        """
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
        """针对单个轨迹:选事实 -> 选关键帧 -> 组 prompt -> 问 MLLM ->
        组装成断言。evidence_frames 直接取自 Unary KFA 的选帧结果。
        """
        selection = self.fact_selector.select(
            facts, SelectionContext(scope=f"instance:{traj.track_id}", top_k=self.config.fact_top_k)
        )
        kfa_selection = self.unary_kfa.select(
            traj.track_id, list(traj.per_frame), self.config.top_k_instance_frames
        )
        # Stage-0 下 projector 是 NoOpProjector,传入空 tuple 只是为了
        # 让调用链路完整跑通;等真正有池化向量时,这里会换成真实的
        # 池化特征而不是空 tuple。
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
        """针对一条候选交互边:选 pair 事实 -> 选双方关键帧 -> 组
        prompt -> 问 MLLM -> 组装成断言。time_span 取选中证据帧的
        最小值到最大值(候选帧本身就是由 Event Candidate Filter 排序
        过的触发帧,不会是空的)。
        """
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
        """整段视频级别:involved_ids 汇总所有出现过的 track_id(排序后
        去重),事实选择用 "video" 这个特殊 scope(DeterministicFactSelector
        会把它当作通配符,对全体事实做概括性挑选)。
        """
        involved_ids = tuple(sorted(t.track_id for t in trajectories))
        selection = self.fact_selector.select(
            facts, SelectionContext(scope="video", top_k=self.config.fact_top_k)
        )
        prompt_text = build_video_prompt(involved_ids, selection.text)
        request = MLLMRequest(prompt_type="video", transcript_text=prompt_text, frame_refs=())
        mllm_text = self.mllm_adapter.generate(request)
        return self.output_assembler.assemble_video(mllm_text, involved_ids)
