"""Output Assembler:确定性地把 MLLM 输出文本组装成可归因的断言。

对应 §4:把开放谓词映射成规范标签,并把 track ID、时间段、证据帧这些
"归因"信息和 MLLM 生成的自然语言文本一起打包,组装出最终可以直接
序列化成 JSON 的断言对象。这一步本身不调用任何模型,是纯规则处理。
"""
from __future__ import annotations

import re
from typing import Optional

from smot.canonical_labels import CANONICAL_MAP, map_predicate
from smot.types import InstanceAssertion, InteractionAssertion, VideoAssertion

_SUBJECT_ID_RE = re.compile(r"subject_id=(\d+)")
_OBJECT_ID_RE = re.compile(r"object_id=(\d+)")

# 谓词短语命中时,如果它紧前面的词是这些否定词之一,说明 MLLM 实际上在
# 否定这个交互("never approaches"),不能提取成肯定谓词。
_NEGATION_WORDS = frozenset(
    {"not", "never", "no", "cannot", "doesn't", "don't", "didn't",
     "isn't", "aren't", "wasn't", "weren't", "won't"}
)

# MLLM 文本里解析不出(或解析出对不上号的)subject/object id 时,断言的
# 方向只能沿用上游 Event Candidate Filter 的下标顺序——那只是一个启发式,
# 不是模型的判断,所以把置信度压到这个标记值,让下游评测/分析能把
# "方向经过模型确认"和"方向只是启发式默认"两类断言区分开。
UNVERIFIED_DIRECTION_CONFIDENCE = 0.5


def _extract_predicate(mllm_text: str, canonical_map: dict[str, str]) -> str:
    """在 mllm_text 里查找最长的、已知的谓词短语(忽略大小写)。

    用"最长匹配"是为了避免像 "approaches" 命中的同时,更具体的短语
    (如果有的话)被更短的子串抢先匹配掉。命中的短语如果紧跟在否定词
    后面("never approaches"),视为未命中——把否定句提取成肯定谓词
    比提取失败更糟糕。如果一个已知谓词都没匹配到,就退回整段文本
    (去首尾空格)作为谓词——这样即便遇到没预料到的、但依然是真实
    句子的 MLLM 输出,也能产出一个(未被规范化的)谓词,而不是直接
    报错中断整个流程。
    """
    lowered = mllm_text.lower()
    candidates = []
    for phrase in canonical_map:
        idx = lowered.find(phrase)
        if idx == -1:
            continue
        preceding = lowered[:idx].split()
        if preceding and preceding[-1] in _NEGATION_WORDS:
            continue
        candidates.append(phrase)
    if not candidates:
        return mllm_text.strip()
    return max(candidates, key=len)


class OutputAssembler:
    """确定性。canonical_map 可注入自定义映射表(§7 分层 F1 评测需要
    用 synonym-merged / coarse 等不同粒度的表重跑),不注入则用全局
    默认表 CANONICAL_MAP。
    """

    def __init__(self, canonical_map: Optional[dict[str, str]] = None):
        self.canonical_map = canonical_map if canonical_map is not None else CANONICAL_MAP

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
        """交互断言:先从 mllm_text 里提取谓词,再映射成规范标签。

        subject/object 的分配需要和 MLLM 文本对账:调用方传入的
        (subject_id, object_id) 只是上游候选边的下标顺序(一个启发式,
        不是模型判断)。这里从 mllm_text 里解析 "subject_id=<n>" /
        "object_id=<n>",如果模型明确说的是相反方向,就交换双方;
        解析不出来或对不上号时,沿用启发式顺序但把置信度压到
        UNVERIFIED_DIRECTION_CONFIDENCE,让评测能区分"模型确认的方向"
        和"启发式默认的方向"。direction 字段固定为 "subj->obj"
        (即最终的 subject_id 是动作发出方)。
        """
        subj_match = _SUBJECT_ID_RE.search(mllm_text)
        obj_match = _OBJECT_ID_RE.search(mllm_text)
        if subj_match and obj_match:
            stated = (int(subj_match.group(1)), int(obj_match.group(1)))
            if stated == (object_id, subject_id):
                # MLLM 明确给出了相反的方向,以模型的判断为准。
                subject_id, object_id = object_id, subject_id
            elif stated != (subject_id, object_id):
                # 解析出的 id 和候选边对不上号(模型说到了别的目标),
                # 方向无法确认。
                confidence = min(confidence, UNVERIFIED_DIRECTION_CONFIDENCE)
        else:
            confidence = min(confidence, UNVERIFIED_DIRECTION_CONFIDENCE)

        predicate = _extract_predicate(mllm_text, self.canonical_map)
        return InteractionAssertion(
            subject_id=subject_id,
            object_id=object_id,
            predicate=predicate,
            canonical_label=map_predicate(predicate, self.canonical_map),
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
