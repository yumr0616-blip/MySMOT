"""Output Assembler: deterministic MLLM-text -> attributable assertion.

Deterministic per §4: maps open-vocabulary predicates to canonical labels and
assembles evidence-backed assertions (track ID + time span + evidence
frames) from raw MLLM text.
"""
from __future__ import annotations

from typing import Optional

from smot.canonical_labels import CANONICAL_MAP, map_predicate
from smot.types import InstanceAssertion, InteractionAssertion, VideoAssertion


def _extract_predicate(mllm_text: str) -> str:
    """Find the longest known predicate phrase occurring in mllm_text (case
    insensitive). Falls back to the full trimmed text when none is found, so
    an unrecognized-but-real MLLM sentence still produces a (unmapped)
    predicate rather than raising.
    """
    lowered = mllm_text.lower()
    candidates = [phrase for phrase in CANONICAL_MAP if phrase in lowered]
    if not candidates:
        return mllm_text.strip()
    return max(candidates, key=len)


class OutputAssembler:
    """Deterministic."""

    def __init__(self, canonical_map: Optional[dict[str, str]] = None):
        self.canonical_map = canonical_map or CANONICAL_MAP

    def assemble_instance(
        self,
        track_id: int,
        mllm_text: str,
        time_span: tuple[int, int],
        evidence_frames: tuple[int, ...],
    ) -> InstanceAssertion:
        return InstanceAssertion(
            track_id=track_id,
            caption=mllm_text.strip(),
            time_span=time_span,
            evidence_frames=evidence_frames,
        )

    def assemble_interaction(
        self,
        subject_id: int,
        object_id: int,
        mllm_text: str,
        time_span: tuple[int, int],
        evidence_frames: tuple[int, ...],
        confidence: float = 1.0,
    ) -> InteractionAssertion:
        predicate = _extract_predicate(mllm_text)
        return InteractionAssertion(
            subject_id=subject_id,
            object_id=object_id,
            predicate=predicate,
            canonical_label=map_predicate(predicate),
            time_span=time_span,
            evidence_frames=evidence_frames,
            confidence=confidence,
        )

    def assemble_video(
        self, mllm_text: str, involved_ids: tuple[int, ...]
    ) -> VideoAssertion:
        return VideoAssertion(summary=mllm_text.strip(), involved_ids=involved_ids)
