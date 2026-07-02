"""冻结 MLLM:Protocol 定义 + 一个"真材实料"的 Stage-0 mock 实现。

对应 §4:MLLM 本身是冻结的,没有独立输出头,靠 prompt 区分任务、
生成式地输出文本。MockMLLMAdapter 用来代替真实的 Qwen-VL 一类模型——
它会真的从 smot.prompts 塞进 transcript_text 里的固定格式 id
(比如 "track_id=3"、"subject_id=1 ... object_id=2")用正则解析出来,
再拼出对应的"照本宣科"回复文本。这样下游的 OutputAssembler 解析逻辑
是被真实地跑过一遍的,而不是简单地把写死的字符串原样传回去。
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
    """发给 MLLM 的一次请求:任务类型 + transcript 文本(prompts.py
    构造出的完整 prompt)+ 关键帧引用 + soft token(默认空,由 Pipeline
    把 projector 的输出接进来;Stage-0 的 NoOpProjector 不产 token,
    真实 projector 的 token 会原样到达这里)。
    """

    prompt_type: str  # "instance" | "interaction" | "video"
    transcript_text: str
    frame_refs: tuple[int, ...]
    soft_tokens: tuple[tuple[float, ...], ...] = field(default_factory=tuple)


@runtime_checkable
class MLLMAdapter(Protocol):
    """冻结。"""

    def generate(self, request: MLLMRequest) -> str: ...


class MockMLLMAdapter:
    """冻结(用 mock 代替真实模型)。根据 prompt_type 分支,
    从 transcript_text 里解析出对应的 id,拼出确定性但"看起来真实"
    的回复文本。
    """

    def __init__(self, canned_responses: Optional[dict[str, str]] = None):
        # 允许调用方按 prompt_type 直接注入固定回复,用于测试里绕过
        # 正则解析、单独验证某个分支的行为。
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
            # 固定用 "approaches" 这个谓词,它同时也在
            # canonical_labels.CANONICAL_MAP 里有对应的规范化映射,
            # 方便验证 Output Assembler 的谓词提取+规范化流程。
            return f"subject_id={subject_id} approaches object_id={object_id}."

        if request.prompt_type == "video":
            return "Two tracked objects approach each other during the video."

        raise ValueError(f"unknown prompt_type: {request.prompt_type!r}")
