"""Fact Selector:Protocol 定义 + Stage-0 确定性实现。

对应 §4/§6:事实的数值(value)永远不学习(由确定性的 Motion Fact
Extractor 产出);学习的只是"该选哪些事实放进 transcript 文本"这件事,
通过一个和 KFA 结构同构的 ride-along slot 来实现(soft+hard 双读出,
hard 的离散选择搭 soft 读出的梯度便车)。

Stage-0 阶段还没有真正的可学习 slot,所以 DeterministicFactSelector
只是按一个固定的类型优先级顺序挑事实,soft_token 恒为 None。未来
Stage-1 的可学习实现会实现同一个 FactSelector Protocol,因此 Pipeline
的调用方式完全不需要改变。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from smot.types import FACT_TYPE_ORDER, Fact


@dataclass(frozen=True)
class SelectionContext:
    """描述"在什么范围内选事实"的查询上下文。

    scope 取值约定:
      - "instance:<i>"  单个目标的事实
      - "pair:<i>,<j>"  一对目标之间的事实
      - "video"         整段视频级别(见 DeterministicFactSelector 里
                         对这个特殊值的通配处理)
    """

    scope: str  # 查询范围,见上方约定
    top_k: int = 8  # 最多选出的事实条数


@dataclass(frozen=True)
class FactSelection:
    """一次选择的结果:选中的事实列表、(Stage-0 恒为 None 的)soft
    token、以及渲染成文本后的 transcript。
    """

    selected_facts: tuple[Fact, ...]  # 被选中、按优先级排序后的事实
    soft_token: tuple[float, ...] | None  # 可学习实现的软读出向量;Stage-0 恒为 None
    text: str  # 拼接好、可直接嵌入 prompt 的 transcript 文本


@runtime_checkable
class FactSelector(Protocol):
    """所有 Fact Selector 实现(确定性或可学习)必须满足的接口。"""

    def select(self, facts: list[Fact], query_context: SelectionContext) -> FactSelection:
        """从 facts 中按 query_context 挑出一批事实并渲染成文本。"""
        ...


def render_fact(fact: Fact) -> str:
    """把一条 Fact 渲染成一小段人类可读的文本,格式大致是
    "<类型>[<范围>]=<值> (t=<起>..<止>)",作为最终塞进 MLLM prompt 里的
    transcript 片段。确定性与可学习 Fact Selector 共用同一份渲染——
    学习改变的只是"选哪些",不是"长什么样"(否则 prompt 分布漂移)。
    """
    return f"{fact.type.value}[{fact.scope}]={fact.value} (t={fact.t_span[0]}..{fact.t_span[1]})"


def scoped_facts(facts: list[Fact], scope: str) -> list[Fact]:
    """按 SelectionContext.scope 过滤事实。"video" 是通配 scope:对全部
    事实池做概括,不按 scope 过滤(两个 selector 实现共用这条约定)。"""
    if scope == "video":
        return list(facts)
    return [f for f in facts if f.scope == scope]


class DeterministicFactSelector:
    """确定性(不学习)。Stage-0 默认实现:按固定的类型优先级顺序,
    从匹配 scope 的事实里挑最多 top_k 条,再渲染成文本。
    """

    def __init__(self, priority_order: tuple = FACT_TYPE_ORDER):
        self.priority_order = priority_order

    def select(self, facts: list[Fact], query_context: SelectionContext) -> FactSelection:
        scoped = scoped_facts(facts, query_context.scope)

        def sort_key(fact: Fact) -> int:
            try:
                return self.priority_order.index(fact.type)
            except ValueError:
                # 出现优先级列表里没有的类型时,排到最后而不是报错,
                # 保证未来新增事实类型时不会破坏现有排序逻辑。
                return len(self.priority_order)

        ordered = sorted(scoped, key=sort_key)
        selected = tuple(ordered[: query_context.top_k])
        text = "; ".join(render_fact(f) for f in selected)
        return FactSelection(selected_facts=selected, soft_token=None, text=text)
