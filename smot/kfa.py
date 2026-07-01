"""Unary KFA / Pairwise KFA:Protocol 定义 + Stage-0 no-op 占位实现。

对应 §4/§6:slot + projector 从 Stage-1a(unary)/ Stage-1b(pairwise)
开始才是可学习的,采用 soft+hard 双读出(hard 的 top-k 离散选帧搭 soft
读出的梯度便车)。Stage-0 阶段还没有真正学习的 slot,下面这两个 no-op
占位类实现了同样的 Protocol,用固定的、不学习的规则来选帧,
soft_token 恒为 None——这样真正的可学习 KFA 接进来的时候,
Pipeline 的调用方式完全不需要改动。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from smot.event_filter import EventCandidate
from smot.types import FramePresence


@dataclass(frozen=True)
class KeyFrameSelection:
    """一次关键帧选择的结果:选中的帧号列表 + (Stage-0 恒为 None 的)
    soft token。
    """

    key_frames: tuple[int, ...]
    soft_token: tuple[float, ...] | None


@runtime_checkable
class UnaryKFA(Protocol):
    """Stage-1a 开始可学习。"""

    def select(self, track_id: int, frames: list[FramePresence], top_k: int) -> KeyFrameSelection: ...


@runtime_checkable
class PairwiseKFA(Protocol):
    """Stage-1b 开始可学习。"""

    def select(
        self, edge: tuple[int, int], event_candidate: EventCandidate, top_k: int
    ) -> KeyFrameSelection: ...


class NoOpUnaryKFA:
    """Stage-1a 才会变成可学习;这是 Stage-0 的 no-op 默认实现。
    直接在该轨迹的所有观测帧里,按下标等间隔抽取最多 top_k 帧,
    不做任何"显著性"打分,soft_token 恒为 None。
    """

    def select(self, track_id: int, frames: list[FramePresence], top_k: int) -> KeyFrameSelection:
        ts = [fp.t for fp in frames]
        if not ts:
            return KeyFrameSelection(key_frames=(), soft_token=None)
        if len(ts) <= top_k:
            # 总帧数本来就不超过 top_k,全部保留即可,不需要再抽稀。
            chosen = ts
        else:
            # 在 [0, len(ts)-1] 的下标范围内等间隔取 top_k 个下标
            # (用 set 去重是为了防止 round() 后出现重复下标),
            # 从而得到近似均匀分布在整个出现时段上的关键帧。
            step = (len(ts) - 1) / (top_k - 1) if top_k > 1 else 0
            indices = sorted({round(i * step) for i in range(top_k)})
            chosen = [ts[i] for i in indices]
        return KeyFrameSelection(key_frames=tuple(chosen), soft_token=None)


class NoOpPairwiseKFA:
    """Stage-1b 才会变成可学习;这是 Stage-0 的 no-op 默认实现。
    直接复用 Event Candidate Filter 已经算好的候选帧(触发交互规则的
    那些帧),截断到 top_k,不再额外做"双轮廓显著性"选帧,
    soft_token 恒为 None。
    """

    def select(
        self, edge: tuple[int, int], event_candidate: EventCandidate, top_k: int
    ) -> KeyFrameSelection:
        chosen = event_candidate.candidate_frames[:top_k]
        return KeyFrameSelection(key_frames=chosen, soft_token=None)
