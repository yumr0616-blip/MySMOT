"""Frozen MLLM: Protocol + a real (deterministic) mock adapter for Stage-0.

Frozen per §4: no separate output head, generative, task distinguished by
prompt. MockMLLMAdapter stands in for a real Qwen-VL-class model; it parses
the structured ids that smot.prompts embeds into transcript_text (e.g.
"track_id=3", "subject_id=1 ... object_id=2") and returns canned-but-genuine
text built from those ids, so downstream OutputAssembler parsing is
exercised for real rather than round-tripping a hardcoded string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

_TRACK_ID_RE = re.compile(r"track_id=(\d+)")
_SUBJECT_ID_RE = re.compile(r"subject_id=(\d+)")
_OBJECT_ID_RE = re.compile(r"object_id=(\d+)")


@dataclass(frozen=True)
class MLLMRequest:
    prompt_type: str  # "instance" | "interaction" | "video"
    transcript_text: str
    frame_refs: tuple[int, ...]
    soft_tokens: tuple[tuple[float, ...], ...] = field(default_factory=tuple)


@runtime_checkable
class MLLMAdapter(Protocol):
    """Frozen."""

    def generate(self, request: MLLMRequest) -> str: ...


class MockMLLMAdapter:
    """Frozen (stub standing in for a real MLLM). Deterministic canned text
    keyed off prompt_type, derived from ids embedded in transcript_text.
    """

    def __init__(self, canned_responses: Optional[dict[str, str]] = None):
        self._canned_responses = canned_responses or {}

    def generate(self, request: MLLMRequest) -> str:
        if request.prompt_type in self._canned_responses:
            return self._canned_responses[request.prompt_type]

        if request.prompt_type == "instance":
            match = _TRACK_ID_RE.search(request.transcript_text)
            track_id = match.group(1) if match else "?"
            return f"track_id={track_id} is present and moving."

        if request.prompt_type == "interaction":
            subj = _SUBJECT_ID_RE.search(request.transcript_text)
            obj = _OBJECT_ID_RE.search(request.transcript_text)
            subject_id = subj.group(1) if subj else "?"
            object_id = obj.group(1) if obj else "?"
            return f"subject_id={subject_id} approaches object_id={object_id}."

        if request.prompt_type == "video":
            return "Two tracked objects approach each other during the video."

        raise ValueError(f"unknown prompt_type: {request.prompt_type!r}")
