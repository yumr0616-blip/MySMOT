"""Output Assembler:确定性地把 MLLM 输出文本组装成可归因的断言。

对应 §4:把开放谓词映射成规范标签,并把 track ID、时间段、证据帧这些
"归因"信息和 MLLM 生成的自然语言文本一起打包,组装出最终可以直接
序列化成 JSON 的断言对象。这一步本身不调用任何模型,是纯规则处理。
"""
from __future__ import annotations

from typing import Optional

from smot.canonical_labels import CANONICAL_MAP, map_predicate
from smot.types import InstanceAssertion, InteractionAssertion, VideoAssertion


def _extract_predicate(mllm_text: str) -> str:
    """在 mllm_text 里查找最长的、已知的谓词短语(忽略大小写)。

    用"最长匹配"是为了避免像 "approaches" 命中的同时,更具体的短语
    (如果有的话)被更短的子串抢先匹配掉。如果一个已知谓词都没匹配到,
    就退回整段文本(去首尾空格)作为谓词——这样即便遇到没预料到的、
    但依然是真实句子的 MLLM 输出,也能产出一个(未被规范化的)谓词,
    而不是直接报错中断整个流程。
    """
    lowered = mllm_text.lower()
    candidates = [phrase for phrase in CANONICAL_MAP if phrase in lowered]
    if not candidates:
        return mllm_text.strip()
    return max(candidates, key=len)


class OutputAssembler:
    """确定性。"""

    def __init__(self, canonical_map: Optional[dict[str, str]] = None):
        self.canonical_map = canonical_map or CANONICAL_MAP

    def assemble_instance(
        self,
        track_id: int,
        mllm_text: str,
        time_span: tuple[int, int],
        evidence_frames: tuple[int, ...],
    ) -> InstanceAssertion:
        """单目标断言:MLLM 文本原样作为 caption,直接带上时间段和
        证据帧(由调用方——通常是 Pipeline——从 KFA 选帧结果里传入)。
        """
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
        """交互断言:先从 mllm_text 里提取谓词,再映射成规范标签,
        subject/object 的方向固定记为 "subj->obj"(即
        subject_id 是动作发出方,object_id 是接受方)。
        confidence 在 Stage-0 阶段是一个占位常数(默认 1.0),
        等真正有打分机制(比如 MLLM 输出的置信度或规则打分)时再替换。
        """
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
        """视频级断言:MLLM 文本作为整体概括,involved_ids 通常是
        Pipeline 汇总出的"本次视频里出现过的所有 track_id"。
        """
        return VideoAssertion(summary=mllm_text.strip(), involved_ids=involved_ids)
