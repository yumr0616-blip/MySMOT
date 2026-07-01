"""Unary KFA / Pairwise KFA: Protocols + Stage-0 no-op stubs.

Slots + projector are learnable starting Stage-1a (unary)/Stage-1b (pairwise)
per §4/§6, with soft+hard dual readout (hard top-k selection rides along the
soft readout's gradient). Stage-0 has no learned slots yet, so the no-op
stubs below implement the same Protocols with fixed, non-learned selection
rules and soft_token=None, so Pipeline's call signature won't change when
real KFA slots are wired in later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from smot.event_filter import EventCandidate
from smot.types import FramePresence


@dataclass(frozen=True)
class KeyFrameSelection:
    key_frames: tuple[int, ...]
    soft_token: tuple[float, ...] | None


@runtime_checkable
class UnaryKFA(Protocol):
    """Learnable in Stage-1a."""

    def select(self, track_id: int, frames: list[FramePresence], top_k: int) -> KeyFrameSelection: ...


@runtime_checkable
class PairwiseKFA(Protocol):
    """Learnable in Stage-1b."""

    def select(
        self, edge: tuple[int, int], event_candidate: EventCandidate, top_k: int
    ) -> KeyFrameSelection: ...


class NoOpUnaryKFA:
    """Learnable in Stage-1a; this is the Stage-0 no-op default. Returns
    evenly spaced frame indices up to top_k, soft_token=None.
    """

    def select(self, track_id: int, frames: list[FramePresence], top_k: int) -> KeyFrameSelection:
        ts = [fp.t for fp in frames]
        if not ts:
            return KeyFrameSelection(key_frames=(), soft_token=None)
        if len(ts) <= top_k:
            chosen = ts
        else:
            step = (len(ts) - 1) / (top_k - 1) if top_k > 1 else 0
            indices = sorted({round(i * step) for i in range(top_k)})
            chosen = [ts[i] for i in indices]
        return KeyFrameSelection(key_frames=tuple(chosen), soft_token=None)


class NoOpPairwiseKFA:
    """Learnable in Stage-1b; this is the Stage-0 no-op default. Returns the
    EventCandidateFilter's candidate frames directly (truncated to top_k),
    soft_token=None.
    """

    def select(
        self, edge: tuple[int, int], event_candidate: EventCandidate, top_k: int
    ) -> KeyFrameSelection:
        chosen = event_candidate.candidate_frames[:top_k]
        return KeyFrameSelection(key_frames=chosen, soft_token=None)
