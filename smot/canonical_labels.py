"""开放词表谓词(predicate)到规范化标签(canonical label)的映射表。

对应 §4 Output Assembler 的职责之一:MLLM 输出的谓词是开放词表、
自由文本形式的(比如"approaches"、"moves toward"含义相近但字面不同),
评测(§7)需要一个稳定的规范标签集合来做 F1 等指标统计,所以这里
提供一个确定性的映射。查不到映射时,直接退回小写去空格后的原始谓词
而不是报错——这个 fallback 行为是刻意为之并且要保持稳定,否则未来
评测代码依赖这个行为时会被破坏。
"""
from __future__ import annotations

CANONICAL_MAP: dict[str, str] = {
    "approaches": "approach",
    "is approaching": "approach",
    "moves toward": "approach",
    "moves away from": "recede",
    "recedes from": "recede",
    "touches": "contact",
    "is in contact with": "contact",
    "follows": "follow",
    "is present and moving": "present",
}


def map_predicate(raw_predicate: str) -> str:
    """把原始谓词规范化(去首尾空格、转小写)后查表;查不到就原样返回
    规范化后的字符串(fallback,不抛异常)。
    """
    normalized = raw_predicate.strip().lower()
    return CANONICAL_MAP.get(normalized, normalized)
