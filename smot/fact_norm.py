"""Fact.embed 的 norm_value 分量按数据集统计量做 z-score 归一化(stdlib)。

Fact.embed 的约定是 (type_index, norm_value, t_start_norm, t_end_norm),
其中 norm_value 在抽取时保留原始数值(types.py 里记录的 Stage-1a 前置
待办):距离/速度类数值动辄几十上百像素,而其余三个分量都在 [0, 5] 内,
不归一化的话池化向量会被 norm_value 一个分量主导,可学习组件(projector
/未来的 fact selector)的输入尺度完全失衡。

统计量来自 smot.datasets.bensmot.compute_fact_statistics(按事实类型的
n/mean/std)。训练与推理必须应用同一份统计量做同一变换——训练脚本把
统计量存进 checkpoint 的 extra,推理脚本从那里取回,两侧都经由本模块
构造的同一个变换函数注入 Pipeline 的 fact_transform 接缝。
"""
from __future__ import annotations

import dataclasses
from typing import Callable

from smot.types import Fact


def make_fact_embed_normalizer(
    stats: dict[str, dict],
) -> Callable[[list[Fact]], list[Fact]]:
    """按 {事实类型: {"mean": .., "std": ..}} 构造 embed 归一化函数。

    只变换 embed[1](norm_value 分量),fact.value 本身不动(它是给
    transcript/人看的确定性数值,归一化的只是打分用的向量表示)。
    统计表里没有的类型、以及 std<=0(常数分布)的类型原样保留——
    宁可少归一化,也不做除以近零的数值放大。
    """

    def normalize(facts: list[Fact]) -> list[Fact]:
        out: list[Fact] = []
        for fact in facts:
            stat = stats.get(fact.type.value)  # 按事实类型(如 "speed")查统计量
            if not stat or stat.get("std", 0.0) <= 0.0:
                out.append(fact)  # 没有统计量或标准差为 0:原样保留,不做变换
                continue
            embed = list(fact.embed)  # tuple 不可原地修改,先转 list
            embed[1] = (embed[1] - stat["mean"]) / stat["std"]  # 标准 z-score 公式
            # Fact 是 frozen dataclass,不能直接赋值,用 dataclasses.replace
            # 构造一个新实例(其余字段原样复制),保持不可变性契约。
            out.append(dataclasses.replace(fact, embed=tuple(embed)))
        return out

    return normalize
