"""Prompt template builders for the three MLLM task types.

Per §4/§6: the frozen MLLM has no separate output head; task is distinguished
purely by prompt. These builders assemble the transcript text (from a
FactSelector's rendered text) plus task framing that Pipeline sends as an
MLLMRequest.
"""
from __future__ import annotations


def build_instance_prompt(track_id: int, transcript_text: str) -> str:
    return (
        f"Describe the behavior of track_id={track_id} based on the following "
        f"motion facts: {transcript_text}"
    )


def build_interaction_prompt(subject_id: int, object_id: int, transcript_text: str) -> str:
    return (
        f"Describe the interaction between subject_id={subject_id} and "
        f"object_id={object_id} based on the following motion facts: {transcript_text}"
    )


def build_video_prompt(involved_ids: tuple[int, ...], transcript_text: str) -> str:
    ids_text = ", ".join(str(i) for i in involved_ids)
    return (
        f"Summarize the video involving track ids [{ids_text}] based on the "
        f"following motion facts: {transcript_text}"
    )
