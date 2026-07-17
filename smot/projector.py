"""Projector:Protocol 定义 + Stage-0 no-op 占位实现。

对应 §4:可学习组件,负责把 KFA/Fact Selector 池化后的向量,通过一个
小的残差 MLP 映射到冻结 MLLM 的输入 embedding 空间(m 个 soft token x
d_llm 维)。Stage-0 阶段完全没有 soft token 这个概念(纯靠文本
transcript + 原始关键帧),所以 NoOpProjector 恒返回空 tuple。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Projector(Protocol):
    """可学习。真实实现见 smot.ml.projector.MLPProjector。"""

    def project(self, pooled_vector: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
        """把一个池化向量映射成 m 个 soft token(每个 d_llm 维)。"""
        ...


class NoOpProjector:
    """Stage-1a/1b 才会变成可学习;这是 Stage-0 的 no-op 默认实现,
    不产生任何 soft token。
    """

    def project(self, pooled_vector: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
        # 返回空 tuple:Stage-0 的 prompt 完全靠文本 transcript,没有 soft token。
        return ()
