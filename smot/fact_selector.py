"""Fact Selector: Protocol + Stage-0 deterministic implementation.

Per §4/§6: fact VALUES are never learned (they come from the deterministic
Motion Fact Extractor); only WHICH facts get selected into the transcript is
learned, via a ride-along slot structurally isomorphic to KFA. Stage-0 has no
learned slot yet, so DeterministicFactSelector picks facts by a fixed
priority order and always returns soft_token=None. A Stage-1 learnable
FactSelector implementation will satisfy this same Protocol, so Pipeline
never needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from smot.types import FACT_TYPE_ORDER, Fact


@dataclass(frozen=True)
class SelectionContext:
    scope: str  # "instance:<i>" | "pair:<i>,<j>" | "video"
    top_k: int = 8


@dataclass(frozen=True)
class FactSelection:
    selected_facts: tuple[Fact, ...]
    soft_token: tuple[float, ...] | None
    text: str


@runtime_checkable
class FactSelector(Protocol):
    def select(self, facts: list[Fact], query_context: SelectionContext) -> FactSelection: ...


def _render_fact(fact: Fact) -> str:
    return f"{fact.type.value}[{fact.scope}]={fact.value} (t={fact.t_span[0]}..{fact.t_span[1]})"


class DeterministicFactSelector:
    """Deterministic (not learned). Stage-0 default: selects up to top_k
    facts in a fixed type-priority order, then renders them to text.
    """

    def __init__(self, priority_order: tuple = FACT_TYPE_ORDER):
        self.priority_order = priority_order

    def select(self, facts: list[Fact], query_context: SelectionContext) -> FactSelection:
        if query_context.scope == "video":
            # Video-level queries summarize over the whole fact pool rather
            # than a single instance/pair scope.
            scoped = list(facts)
        else:
            scoped = [f for f in facts if f.scope == query_context.scope]

        def sort_key(fact: Fact) -> int:
            try:
                return self.priority_order.index(fact.type)
            except ValueError:
                return len(self.priority_order)

        ordered = sorted(scoped, key=sort_key)
        selected = tuple(ordered[: query_context.top_k])
        text = "; ".join(_render_fact(f) for f in selected)
        return FactSelection(selected_facts=selected, soft_token=None, text=text)
