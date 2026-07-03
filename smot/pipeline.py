"""Pipeline 编排器:把 §4 里的所有模块串成一次完整的推理流程。

每一个"可能需要学习/真实模型"的组件在构造函数里都是可选参数,
默认值就是它对应的 Stage-0 实现(确定性或 no-op)。以后升级到
Stage-1a/1b(接入真正可学习的 KFA slot、真正的 tracker/MLLM)时,
只需要把对应的参数换成真实实现传进来,Pipeline 本身的调用方式
(构造函数签名、run() 的调用方式)完全不用改——这就是 §6 里
"非破坏性升级"要求的具体落地方式。

soft-token 通路(Stage-1 的关键 seam)在 Stage-0 就已经接通:
Pipeline 把选中事实的 embed 池化成一个向量喂给 projector,把
projector 的输出原样塞进 MLLMRequest.soft_tokens。Stage-0 的
NoOpProjector 返回空 tuple,所以行为上没有变化;但任何返回非空
token 的真实 projector 接进来,token 立刻会到达 MLLM 适配器,
不需要再改 Pipeline 的任何布线。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

from smot.event_filter import EventCandidateFilter
from smot.fact_selector import DeterministicFactSelector, FactSelector, SelectionContext
from smot.kfa import NoOpPairwiseKFA, NoOpUnaryKFA, PairwiseKFA, UnaryKFA
from smot.mllm import MLLMAdapter, MLLMRequest, MockMLLMAdapter
from smot.motion_facts import MotionFactExtractor
from smot.output_assembler import OutputAssembler
from smot.pair_features import build_pair_features
from smot.projector import NoOpProjector, Projector
from smot.prompts import build_instance_prompt, build_interaction_prompt, build_video_prompt
from smot.tracker import Tracker, VideoHandle
from smot.types import Fact, InstanceAssertion, InteractionAssertion, Trajectory, VideoAssertion


@dataclass
class PipelineConfig:
    """跑一次 Pipeline 时用到的几个数量控制参数。"""

    top_k_instance_frames: int = 4  # 每个目标最多保留几张证据帧
    top_k_pair_frames: int = 4  # 每条交互边最多保留几张证据帧
    fact_top_k: int = 6  # 每次 Fact Selector 最多选几条事实进 transcript


@dataclass
class CostReport:
    """一次 Pipeline 运行的成本账单。§7 把这些量列为一等公民指标
    (成本地板/学习后对比都要靠它们),所以在编排器里逐次累加,
    随 PipelineResult 一起输出。
    """

    n_vlm_calls: int = 0  # 发给 MLLM 的请求次数
    n_key_frames: int = 0  # 所有请求携带的关键帧总数(≈ 视觉 token 成本)
    n_facts_selected: int = 0  # 所有请求选入 transcript 的事实总数
    n_soft_tokens: int = 0  # 所有请求携带的 soft token 总数(Stage-0 恒为 0)

    def to_json_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PipelineResult:
    """一次 Pipeline 运行的完整输出:三类断言 + 成本账单打包在一起。"""

    instances: tuple[InstanceAssertion, ...]
    interactions: tuple[InteractionAssertion, ...]
    video: VideoAssertion
    cost: CostReport = field(default_factory=CostReport)

    def to_json_dict(self) -> dict:
        """把整个结果拍平成一个可以直接 json.dumps 的 dict。"""
        return {
            "instances": [a.to_json_dict() for a in self.instances],
            "interactions": [a.to_json_dict() for a in self.interactions],
            "video": self.video.to_json_dict(),
            "cost": self.cost.to_json_dict(),
        }


def _pool_embeds(facts: tuple[Fact, ...]) -> tuple[float, ...]:
    """把若干条事实的 embed 逐维取平均,得到喂给 projector 的池化向量。

    这是 Stage-0 的最简池化:让 projector 从第一天起就有真实的输入,
    而不是永远收到空 tuple。Stage-1 接入可学习 KFA/Fact Selector 后,
    真正的池化向量会来自 slot 的 soft 读出,届时替换这里的来源即可,
    projector 的调用位置和输出布线都不需要动。
    """
    if not facts:
        return ()
    dim = len(facts[0].embed)
    sums = [0.0] * dim
    for fact in facts:
        for i in range(dim):
            sums[i] += fact.embed[i]
    return tuple(s / len(facts) for s in sums)


def _frame_boxes(key_frames, trajs: dict[int, Trajectory]):
    """为每个证据帧收集各 track 在该帧的框,组装成 MLLMRequest.frame_boxes
    约定的形状 ((t, ((track_id, box), ...)), ...)。真实多模态适配器靠它
    在关键帧上画框做视觉 grounding;某 track 在该帧无观测(遮挡)时直接
    省略,不猜框。
    """
    boxes = []
    for t in key_frames:
        entries = tuple(
            (tid, fp.box)
            for tid, traj in sorted(trajs.items())
            if (fp := traj.frame_at(t)) is not None
        )
        boxes.append((t, entries))
    return tuple(boxes)


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
        frame_feature_fn: Optional[
            Callable[[Trajectory], Sequence[tuple[float, ...]]]
        ] = None,
        fact_transform: Optional[Callable[[list[Fact]], list[Fact]]] = None,
    ):
        # tracker 没有默认值——它必须由调用方显式提供(Stage-0 下通常是
        # StubTracker 注入的 GT/预置轨迹;真正跑真实 tracker 时替换成
        # 真实实现即可,其余组件的默认值都不用变)。
        self.tracker = tracker
        # frame_feature_fn:可学习 Unary KFA 的逐帧特征来源(Stage-1a 注入,
        # 例如 smot.frame_features.geometric_frame_features)。默认 None,
        # 此时 unary KFA 的 features 参数收到 None,保持 Stage-0 行为。
        self.frame_feature_fn = frame_feature_fn
        # fact_transform:事实抽取之后、选择/池化之前的整体变换钩子
        # (Stage-1a 用它注入 embed 归一化,见 smot.fact_norm;训练与
        # 推理必须注入同一份变换)。默认 None 保持原始 embed。
        self.fact_transform = fact_transform
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
        全程在 cost 里累加成本计数。
        """
        trajectories = self.tracker.track(video)
        # track_id 是全流程的主键(事实 scope、候选边、断言归因都靠它),
        # 重复的 id 会让下面的 traj_by_id 静默塌缩、断言错配——在入口
        # fail-fast,与 Trajectory 的构造校验同一哲学。
        seen_ids = [traj.track_id for traj in trajectories]
        if len(seen_ids) != len(set(seen_ids)):
            dupes = sorted({i for i in seen_ids if seen_ids.count(i) > 1})
            raise ValueError(f"tracker 输出了重复的 track_id: {dupes}")
        facts = self.motion_fact_extractor.extract(trajectories)
        if self.fact_transform is not None:
            facts = self.fact_transform(facts)
        event_candidates = self.event_filter.find_candidates(trajectories)
        traj_by_id = {traj.track_id: traj for traj in trajectories}
        cost = CostReport()

        instances = tuple(
            self._run_instance(traj, facts, video, cost) for traj in trajectories
        )
        # 一条候选边可能产出多条有向交互断言(真实 MLLM 以 JSON 数组
        # 回答;空数组 = 模型确认无交互,该边不产断言),拍平汇总。
        interactions = tuple(
            assertion
            for candidate in event_candidates
            for assertion in self._run_interaction(
                candidate, facts, traj_by_id, video, cost
            )
        )
        video_assertion = self._run_video(trajectories, facts, video, cost)

        return PipelineResult(
            instances=instances,
            interactions=interactions,
            video=video_assertion,
            cost=cost,
        )

    def _generate(self, request: MLLMRequest, cost: CostReport) -> str:
        """所有 MLLM 调用的唯一出口:顺手把成本计数记全,保证任何新增
        的调用路径都不会漏计。
        """
        cost.n_vlm_calls += 1
        cost.n_key_frames += len(request.frame_refs)
        cost.n_soft_tokens += len(request.soft_tokens)
        return self.mllm_adapter.generate(request)

    def _run_instance(self, traj, facts, video, cost: CostReport) -> InstanceAssertion:
        """针对单个轨迹:选事实 -> 选关键帧 -> 池化+投影出 soft token ->
        组 prompt -> 问 MLLM -> 组装成断言。evidence_frames 直接取自
        Unary KFA 的选帧结果。
        """
        selection = self.fact_selector.select(
            facts, SelectionContext(scope=f"instance:{traj.track_id}", top_k=self.config.fact_top_k)
        )
        cost.n_facts_selected += len(selection.selected_facts)
        # 逐帧特征由注入的 frame_feature_fn 提供(Stage-1a 的可学习 KFA
        # 需要它打分);未注入时传 None,NoOp KFA 也不使用。
        features = (
            self.frame_feature_fn(traj) if self.frame_feature_fn is not None else None
        )
        kfa_selection = self.unary_kfa.select(
            traj.track_id,
            list(traj.per_frame),
            self.config.top_k_instance_frames,
            features=features,
        )
        pooled = _pool_embeds(selection.selected_facts)
        if kfa_selection.soft_token is not None:
            # KFA 的 soft 读出向量拼接在事实池化向量之后一起进 projector。
            # 这是"hard top-k 搭 soft 读出梯度便车"的落地布线:被选中的
            # 帧走图像通路(离散、无梯度),soft 向量走 projector->soft
            # token 通路(可导)——两者共享同一组打分,训练信号经 soft
            # 通路回传给 KFA 打分器。NoOp KFA 返回 None,布线自然短路。
            pooled = tuple(pooled) + tuple(kfa_selection.soft_token)
        soft_tokens = self.projector.project(pooled)
        prompt_text = build_instance_prompt(traj.track_id, selection.text)
        request = MLLMRequest(
            prompt_type="instance",
            transcript_text=prompt_text,
            frame_refs=kfa_selection.key_frames,
            soft_tokens=soft_tokens,
            video_path=video.path,
            frame_boxes=_frame_boxes(
                kfa_selection.key_frames, {traj.track_id: traj}
            ),
        )
        mllm_text = self._generate(request, cost)
        return self.output_assembler.assemble_instance(
            track_id=traj.track_id,
            mllm_text=mllm_text,
            time_span=traj.present,
            evidence_frames=kfa_selection.key_frames,
        )

    def _run_interaction(
        self, candidate, facts, traj_by_id, video, cost: CostReport
    ) -> tuple[InteractionAssertion, ...]:
        """针对一条候选交互边:构造逐帧 pair 特征 -> 选 pair 事实 ->
        选双方关键帧 -> 池化+投影出 soft token -> 组 prompt -> 问 MLLM ->
        组装成断言。time_span 取选中证据帧的最小值到最大值(候选帧本身
        就是由 Event Candidate Filter 排序过的触发帧,不会是空的)。

        注意:这里解包出的 (subject_id, object_id) 只是候选边的下标
        顺序,不是模型对"谁是动作发出方"的判断——真正的方向对账在
        OutputAssembler.assemble_interaction 里做(以 MLLM 文本里
        声明的 subject/object 为准)。
        """
        subject_id, object_id = candidate.edge
        traj_i, traj_j = traj_by_id[subject_id], traj_by_id[object_id]
        pair_features = build_pair_features(traj_i, traj_j, candidate.candidate_frames)
        # scope 键统一用排序后的 id(和 MotionFactExtractor 的约定一致),
        # 与轨迹列表顺序解耦;subject/object 的方向语义只体现在 prompt 里。
        lo, hi = sorted((subject_id, object_id))
        selection = self.fact_selector.select(
            facts,
            SelectionContext(scope=f"pair:{lo},{hi}", top_k=self.config.fact_top_k),
        )
        cost.n_facts_selected += len(selection.selected_facts)
        kfa_selection = self.pairwise_kfa.select(
            candidate.edge,
            candidate,
            self.config.top_k_pair_frames,
            pair_features=pair_features,
        )
        pooled = _pool_embeds(selection.selected_facts)
        if kfa_selection.soft_token is not None:
            # 与 _run_instance 相同的布线:pairwise KFA(Stage-1b 起可学习)
            # 的 soft 读出向量拼进 projector 输入;NoOp 返回 None 时短路。
            pooled = tuple(pooled) + tuple(kfa_selection.soft_token)
        soft_tokens = self.projector.project(pooled)
        prompt_text = build_interaction_prompt(subject_id, object_id, selection.text)
        frames = kfa_selection.key_frames
        request = MLLMRequest(
            prompt_type="interaction",
            transcript_text=prompt_text,
            frame_refs=frames,
            soft_tokens=soft_tokens,
            video_path=video.path,
            frame_boxes=_frame_boxes(
                frames, {subject_id: traj_i, object_id: traj_j}
            ),
        )
        mllm_text = self._generate(request, cost)
        time_span = (min(frames), max(frames)) if frames else (0, 0)
        return self.output_assembler.assemble_interactions(
            subject_id=subject_id,
            object_id=object_id,
            mllm_text=mllm_text,
            time_span=time_span,
            evidence_frames=frames,
        )

    def _run_video(self, trajectories, facts, video, cost: CostReport) -> VideoAssertion:
        """整段视频级别:involved_ids 汇总所有出现过的 track_id(排序后
        去重),事实选择用 "video" 这个特殊 scope(DeterministicFactSelector
        会把它当作通配符,对全体事实做概括性挑选)。
        """
        involved_ids = tuple(sorted(t.track_id for t in trajectories))
        selection = self.fact_selector.select(
            facts, SelectionContext(scope="video", top_k=self.config.fact_top_k)
        )
        cost.n_facts_selected += len(selection.selected_facts)
        soft_tokens = self.projector.project(_pool_embeds(selection.selected_facts))
        prompt_text = build_video_prompt(involved_ids, selection.text)
        request = MLLMRequest(
            prompt_type="video",
            transcript_text=prompt_text,
            frame_refs=(),
            soft_tokens=soft_tokens,
            video_path=video.path,
        )
        mllm_text = self._generate(request, cost)
        return self.output_assembler.assemble_video(mllm_text, involved_ids)
