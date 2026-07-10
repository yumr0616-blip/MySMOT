"""Output Assembler:确定性地把 MLLM 输出文本组装成可归因的断言。

对应 §4:把开放谓词映射成规范标签,并把 track ID、时间段、证据帧这些
"归因"信息和 MLLM 生成的自然语言文本一起打包,组装出最终可以直接
序列化成 JSON 的断言对象。这一步本身不调用任何模型,是纯规则处理。
"""
from __future__ import annotations

import json
import re
from typing import Optional

from smot.canonical_labels import CANONICAL_MAP, map_predicate
from smot.types import InstanceAssertion, InteractionAssertion, VideoAssertion

_SUBJECT_ID_RE = re.compile(r"subject_id=(\d+)")
_OBJECT_ID_RE = re.compile(r"object_id=(\d+)")

_JSON_DECODER = json.JSONDecoder()

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


def _is_interaction_item(obj) -> bool:
    """判断一个解析出的 JSON 值是否"长得像"一条交互记录(带 predicate 键的 dict)。"""
    return isinstance(obj, dict) and "predicate" in obj


def _find_structured_interactions(text: str) -> Optional[list[dict]]:
    """在 MLLM 输出里找出结构化的交互 JSON,统一返回对象列表。

    真实 MLLM 被要求以 JSON 数组回答交互任务(一对目标可能同时存在
    多条有向交互),但模型经常在 JSON 前后加解释文字或 ```json 围栏,
    也可能退化成单个对象,所以不能直接 json.loads 整段文本——从每个
    '[' 或 '{' 起尝试 raw_decode(它能正确处理字符串里的括号),按:
      - 解析出数组且所有元素都是带 predicate 键的对象 -> 原样返回
        (空数组 [] 是模型明确的"无交互"信号,返回空列表);
      - 解析出带 predicate 键的单个对象 -> 包成单元素列表;
    找不到返回 None,调用方退回自由文本谓词抽取路径(Mock 和不守
    指令的模型都走那条路)。
    """
    for idx in (i for i, ch in enumerate(text) if ch in "[{"):
        try:
            obj, _ = _JSON_DECODER.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and all(_is_interaction_item(o) for o in obj):
            return list(obj)
        if _is_interaction_item(obj):
            return [obj]
    return None


def _as_int(value) -> Optional[int]:
    """宽容地把 JSON 字段值转成 int(模型可能输出 "1" 或 1.0);
    转不了返回 None(视为该字段缺失)。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _reconcile_direction(
    stated: Optional[tuple[int, int]], subject_id: int, object_id: int, confidence: float
) -> tuple[int, int, float]:
    """方向对账(结构化 JSON 与正则两条解析路径共用同一套规则):

    调用方传入的 (subject_id, object_id) 只是上游候选边的下标顺序;
    stated 是模型自己声明的 (subject, object)。
      - 模型声明的恰好是相反方向 -> 交换,以模型判断为准;
      - 声明与候选边完全一致    -> 保持,置信度不动;
      - 声明缺失或对不上号      -> 保持启发式顺序,但置信度压到
        UNVERIFIED_DIRECTION_CONFIDENCE,让评测能区分"模型确认的方向"
        和"启发式默认的方向"。
    """
    if stated is None:
        return subject_id, object_id, min(confidence, UNVERIFIED_DIRECTION_CONFIDENCE)
    if stated == (object_id, subject_id):
        return object_id, subject_id, confidence
    if stated != (subject_id, object_id):
        return subject_id, object_id, min(confidence, UNVERIFIED_DIRECTION_CONFIDENCE)
    return subject_id, object_id, confidence


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
        """单断言入口(兼容旧调用方/测试):取 assemble_interactions 的
        第一条;模型明确回答"无交互"(空数组)时按文本兜底路径产出
        一条低置信断言,保证本方法总有返回值。
        """
        assertions = self.assemble_interactions(
            subject_id, object_id, mllm_text, time_span, evidence_frames, confidence
        )
        if assertions:
            return assertions[0]
        return self._assemble_from_text(
            subject_id, object_id, mllm_text, time_span, evidence_frames, confidence
        )

    def assemble_interactions(
        self,
        subject_id: int,
        object_id: int,
        mllm_text: str,
        time_span: tuple[int, int],
        evidence_frames: tuple[int, ...],
        confidence: float = 1.0,
    ) -> tuple[InteractionAssertion, ...]:
        """交互断言(复数):结构化 JSON 数组优先,自由文本谓词抽取兜底。

        真实 MLLM 被要求以 JSON 数组回答交互任务(每项 {"subject_id",
        "object_id", "predicate"}),因为一对目标往往同时存在多条有向
        交互(A 递钱给 B 的同时两人在交谈)——BenSMOT 的 gold 平均每对
        4+ 条有向断言,单断言输出的召回上限只有 ~25%。

        解析到数组时逐项组装(谓词直接取字段值,方向按各项声明的 id
        对账);空数组是模型明确的"无交互"信号,返回空元组;解析不到
        任何 JSON(Mock、或模型没守指令)时退回正则 + 已知短语最长匹配
        的文本路径,产出单条断言。同一 (subject, object, predicate) 的
        重复项去重。所有路径的方向对账规则相同(_reconcile_direction),
        direction 字段固定为 "subj->obj"(最终 subject_id 是动作发出方)。
        """
        structured = _find_structured_interactions(mllm_text)
        if structured is None:
            return (
                self._assemble_from_text(
                    subject_id, object_id, mllm_text, time_span,
                    evidence_frames, confidence,
                ),
            )

        assertions: list[InteractionAssertion] = []
        seen: set[tuple[int, int, str]] = set()  # (subj, obj, predicate小写) 去重键
        for item in structured:
            predicate = str(item["predicate"]).strip()
            if not predicate:
                continue
            stated_subj = _as_int(item.get("subject_id"))
            stated_obj = _as_int(item.get("object_id"))
            stated = (
                (stated_subj, stated_obj)
                if stated_subj is not None and stated_obj is not None
                else None
            )
            subj, obj, conf = _reconcile_direction(
                stated, subject_id, object_id, confidence
            )
            key = (subj, obj, predicate.lower())
            if key in seen:
                continue  # 同一条交互模型重复给出(或大小写不同的同义写法),只留一条
            seen.add(key)
            assertions.append(
                InteractionAssertion(
                    subject_id=subj,
                    object_id=obj,
                    predicate=predicate,
                    canonical_label=map_predicate(predicate, self.canonical_map),
                    time_span=time_span,
                    evidence_frames=evidence_frames,
                    confidence=conf,
                )
            )
        return tuple(assertions)

    def _assemble_from_text(
        self,
        subject_id: int,
        object_id: int,
        mllm_text: str,
        time_span: tuple[int, int],
        evidence_frames: tuple[int, ...],
        confidence: float,
    ) -> InteractionAssertion:
        """自由文本兜底路径(旧行为原样保留)。"""
        subj_match = _SUBJECT_ID_RE.search(mllm_text)
        obj_match = _OBJECT_ID_RE.search(mllm_text)
        stated = (
            (int(subj_match.group(1)), int(obj_match.group(1)))
            if subj_match and obj_match
            else None
        )
        predicate = _extract_predicate(mllm_text, self.canonical_map)
        subject_id, object_id, confidence = _reconcile_direction(
            stated, subject_id, object_id, confidence
        )
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
