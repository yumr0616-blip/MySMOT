"""Projector: Protocol + Stage-0 no-op stub.

Learnable per §4: a small residual MLP mapping a pooled KFA/Fact vector into
the frozen MLLM's input embedding space (m soft tokens x d_llm). Stage-0 has
no soft tokens at all, so NoOpProjector always returns an empty tuple.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Projector(Protocol):
    """Learnable."""

    def project(self, pooled_vector: tuple[float, ...]) -> tuple[tuple[float, ...], ...]: ...


class NoOpProjector:
    """Learnable in Stage-1a/1b; this is the Stage-0 no-op default. Returns
    no soft tokens.
    """

    def project(self, pooled_vector: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
        return ()
