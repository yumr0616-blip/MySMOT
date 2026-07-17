"""Unary KFA / Pairwise KFA:Protocol 定义 + Stage-0 no-op 占位实现。

KFA = Key Frame Attention(关键帧注意力):从一个目标(unary)或一对
目标(pairwise)的全部观测帧里,挑出最能代表其行为/交互的少数几帧
喂给 MLLM——因为把所有帧都喂进去既昂贵又会稀释注意力,所以需要一个
"选帧"的模块,而且这个选择本身可以是可学习的(打分 + top-k)。

对应 §4/§6:slot + projector 从 Stage-1a(unary)/ Stage-1b(pairwise)
开始才是可学习的,采用 soft+hard 双读出(hard 的 top-k 离散选帧搭 soft
读出的梯度便车)。Stage-0 阶段还没有真正学习的 slot,下面这两个 no-op
占位类实现了同样的 Protocol,用固定的、不学习的规则来选帧,
soft_token 恒为 None——这样真正的可学习 KFA 接进来的时候,
Pipeline 的调用方式完全不需要改动。

接口形状说明(§4 的输入契约,Stage-1 接真实现时不需要再改签名):
  - Unary KFA 的打分依据是"逐帧实例视觉特征",所以 select() 除了
    frames(框/mask/conf)之外还接收一个与 frames 逐帧对齐的
    features 序列;Stage-0 没有视觉塔,Pipeline 传 None,NoOp 实现
    也不使用它。
  - Pairwise KFA 的打分依据是"候选边逐帧 pair 特征"(两方视觉特征 +
    确定性相对几何 RelGeom),所以 select() 除了 EventCandidate 之外
    还接收与 candidate_frames 对齐的 pair_features 序列;Stage-0 由
    Pipeline 用 smot.pair_features.build_pair_features 确定性构造
    (视觉特征分量为空,rel_geom 是真实计算值)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from smot.event_filter import EventCandidate
from smot.types import FramePresence, PairFeature


@dataclass(frozen=True)
class KeyFrameSelection:
    """一次关键帧选择的结果:选中的帧号列表 + (Stage-0 恒为 None 的)
    soft token。
    """

    key_frames: tuple[int, ...]  # hard 读出:实际渲染送入 MLLM 的帧号(离散选择)
    soft_token: tuple[float, ...] | None  # soft 读出:梯度回传通道;不学习时为 None


def _evenly_spaced(ts: Sequence[int], top_k: int) -> tuple[int, ...]:
    """从有序帧号序列里等间隔抽取最多 top_k 帧(保证首尾都被覆盖),
    两个 NoOp KFA 共用:不做任何"显著性"打分,只求覆盖整个时段,
    避免简单截断把证据帧全部偏向序列开头。
    """
    if not ts:
        return ()
    if len(ts) <= top_k:
        # 总帧数本来就不超过 top_k,全部保留即可,不需要再抽稀。
        return tuple(ts)
    # 在 [0, len(ts)-1] 的下标范围内等间隔取 top_k 个下标
    # (用 set 去重是为了防止 round() 后出现重复下标),
    # 从而得到近似均匀分布在整个时段上的关键帧。
    # step 是相邻抽样点之间的平均下标间隔;top_k==1 时特判为 0,
    # 表示只取一个点(下面 i*step 恒为 0,即取第一个下标)。
    step = (len(ts) - 1) / (top_k - 1) if top_k > 1 else 0
    indices = sorted({round(i * step) for i in range(top_k)})
    return tuple(ts[i] for i in indices)


@runtime_checkable
class UnaryKFA(Protocol):
    """Stage-1a 开始可学习。features 与 frames 逐帧对齐,是该目标在
    对应帧上的实例视觉特征向量;Stage-0 没有视觉塔时传 None。
    """

    def select(
        self,
        track_id: int,
        frames: list[FramePresence],
        top_k: int,
        features: Optional[Sequence[tuple[float, ...]]] = None,
    ) -> KeyFrameSelection: ...


@runtime_checkable
class PairwiseKFA(Protocol):
    """Stage-1b 开始可学习。pair_features 与 event_candidate 的
    candidate_frames 对齐(观测缺失的候选帧会被跳过),携带两方视觉
    特征和确定性相对几何——这是 pairwise slot 打分方向性的信息来源,
    缺了它 pairwise 选择会退化成两个 unary 特征的拼接。

    features 是 pair_features 的定长向量化(与 pair_features 逐帧对齐,
    由 Pipeline 经注入的 pair_feature_fn 构造,见 smot.pair_features.
    pair_feature_vectors)——与 Unary KFA 的 features 参数完全同一种
    角色:打分 MLP 消费的是它,PairFeature 对象只负责携带帧号等元信息。
    未注入时为 None,NoOp 实现不使用。
    """

    def select(
        self,
        edge: tuple[int, int],
        event_candidate: EventCandidate,
        top_k: int,
        pair_features: Sequence[PairFeature] = (),
        features: Optional[Sequence[tuple[float, ...]]] = None,
    ) -> KeyFrameSelection: ...


class NoOpUnaryKFA:
    """Stage-1a 才会变成可学习;这是 Stage-0 的 no-op 默认实现。
    直接在该轨迹的所有观测帧里,按下标等间隔抽取最多 top_k 帧,
    不做任何"显著性"打分(因此也不使用 features),soft_token 恒为 None。
    """

    def select(
        self,
        track_id: int,
        frames: list[FramePresence],
        top_k: int,
        features: Optional[Sequence[tuple[float, ...]]] = None,
    ) -> KeyFrameSelection:
        ts = [fp.t for fp in frames]
        return KeyFrameSelection(key_frames=_evenly_spaced(ts, top_k), soft_token=None)


class NoOpPairwiseKFA:
    """Stage-1b 才会变成可学习;这是 Stage-0 的 no-op 默认实现。
    直接复用 Event Candidate Filter 已经算好的候选帧(触发交互规则的
    那些帧),等间隔抽稀到 top_k(而不是简单截断——截断会把证据帧
    全部偏向事件开头),不再额外做"双轮廓显著性"选帧(因此也不使用
    pair_features),soft_token 恒为 None。
    """

    def select(
        self,
        edge: tuple[int, int],
        event_candidate: EventCandidate,
        top_k: int,
        pair_features: Sequence[PairFeature] = (),
        features: Optional[Sequence[tuple[float, ...]]] = None,
    ) -> KeyFrameSelection:
        chosen = _evenly_spaced(event_candidate.candidate_frames, top_k)
        return KeyFrameSelection(key_frames=chosen, soft_token=None)
