"""可学习组件的 checkpoint 存取(Stage-1a 两模块 / Stage-1b 四模块)。

训练(smot.ml.training)与推理(examples/run_bensmot_real.py --checkpoint)
共用同一格式,构造参数一并存盘——加载侧不需要再记得训练时的维度配置,
配置漂移(比如换了 n_tokens 之后加载旧档)会在 load_state_dict 的形状
校验里当场暴露。

两种格式:
  Stage-1a  {kfa_*, projector_*}                       (历史格式,无 stage 键)
  Stage-1b  {stage: "1b", unary_kfa_*, pairwise_kfa_*,
             fact_selector_*, projector_*}
load_checkpoint() 按 payload 自动判别,统一返回 LoadedCheckpoint;
1a 专用函数保留,旧调用方/旧档案不受影响。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from smot.ml.fact_selector import LearnableFactSelector
from smot.ml.pairwise_kfa import LearnablePairwiseKFA
from smot.ml.projector import MLPProjector
from smot.ml.unary_kfa import LearnableUnaryKFA


def save_stage1a_checkpoint(
    path: str | Path,
    kfa: LearnableUnaryKFA,
    projector: MLPProjector,
    extra: dict | None = None,
) -> None:
    """保存两个可学习模块的权重与构造配置(extra 放训练元信息,如
    step 数、loss 曲线文件路径等,加载侧原样透传)。"""
    payload = {
        "kfa_state": kfa.state_dict(),  # 权重(nn.Module 的标准序列化格式)
        "kfa_config": {  # 重建同结构模块所需的构造参数
            "in_dim": kfa.in_dim,
            "out_dim": kfa.out_dim,
        },
        "projector_state": projector.state_dict(),
        "projector_config": {
            "in_dim": projector.in_dim,
            "d_llm": projector.d_llm,
            "n_tokens": projector.n_tokens,
        },
        "extra": extra or {},  # 训练元信息(fact_stats/epoch/step等),原样透传
    }
    torch.save(payload, str(path))


def load_stage1a_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[LearnableUnaryKFA, MLPProjector, dict]:
    """按存盘的构造配置重建模块并加载权重,返回 (kfa, projector, extra)。
    两个模块都置为 eval() 模式(推理用途;继续训练的话调用方自行 train())。
    """
    # weights_only=True:只反序列化张量/基础类型,不执行任意 pickle 代码,
    # 加载不受信任来源的 checkpoint 时更安全,这里的 payload 本身也只有
    # 张量/dict/基础类型,不需要更宽松的模式。
    payload = torch.load(str(path), map_location=device, weights_only=True)
    # 先按存盘的配置重建"空壳"模块(结构必须与训练时一致),
    # 再把权重灌进去——形状不匹配时 load_state_dict 会直接报错,
    # 这正是模块顶部说的"配置漂移会在这里当场暴露"。
    kfa = LearnableUnaryKFA(
        in_dim=payload["kfa_config"]["in_dim"],
        out_dim=payload["kfa_config"]["out_dim"],
    ).to(device)
    kfa.load_state_dict(payload["kfa_state"])
    kfa.eval()  # 推理模式:关闭 dropout 等训练专属行为
    projector = MLPProjector(
        in_dim=payload["projector_config"]["in_dim"],
        d_llm=payload["projector_config"]["d_llm"],
        n_tokens=payload["projector_config"]["n_tokens"],
    ).to(device)
    projector.load_state_dict(payload["projector_state"])
    projector.eval()
    return kfa, projector, payload.get("extra", {})


def save_stage1b_checkpoint(
    path: str | Path,
    unary_kfa: LearnableUnaryKFA,
    pairwise_kfa: LearnablePairwiseKFA,
    fact_selector: LearnableFactSelector,
    projector: MLPProjector,
    extra: dict | None = None,
) -> None:
    """保存 Stage-1b 的四个可学习模块(权重 + 构造配置 + extra 元信息)。"""
    payload = {
        "stage": "1b",  # 格式判别字段(1a 历史格式没有它)
        "unary_kfa_state": unary_kfa.state_dict(),
        "unary_kfa_config": {"in_dim": unary_kfa.in_dim, "out_dim": unary_kfa.out_dim},
        "pairwise_kfa_state": pairwise_kfa.state_dict(),
        "pairwise_kfa_config": {
            "in_dim": pairwise_kfa.in_dim,
            "out_dim": pairwise_kfa.out_dim,
        },
        "fact_selector_state": fact_selector.state_dict(),
        "fact_selector_config": {
            "in_dim": fact_selector.in_dim,
            "out_dim": fact_selector.out_dim,
        },
        "projector_state": projector.state_dict(),
        "projector_config": {
            "in_dim": projector.in_dim,
            "d_llm": projector.d_llm,
            "n_tokens": projector.n_tokens,
        },
        "extra": extra or {},
    }
    torch.save(payload, str(path))


@dataclass
class LoadedCheckpoint:
    """load_checkpoint() 的统一返回:1a 档案的 1b 专属组件为 None。
    调用方按"组件是否存在"接线 Pipeline,不需要自己分辨格式。"""

    stage: str  # "1a" | "1b"
    unary_kfa: LearnableUnaryKFA
    projector: MLPProjector
    pairwise_kfa: Optional[LearnablePairwiseKFA] = None
    fact_selector: Optional[LearnableFactSelector] = None
    extra: dict = field(default_factory=dict)


def load_checkpoint(path: str | Path, device: str = "cpu") -> LoadedCheckpoint:
    """通用加载器:按 payload 的 stage 键判别 1a/1b 格式,重建模块并加载
    权重,全部置为 eval()。"""
    payload = torch.load(str(path), map_location=device, weights_only=True)
    if payload.get("stage") != "1b":
        # 历史 Stage-1a 格式:直接复用专用加载器(键名不同)。
        kfa, projector, extra = load_stage1a_checkpoint(path, device=device)
        return LoadedCheckpoint(
            stage="1a", unary_kfa=kfa, projector=projector, extra=extra
        )

    def build(cls, prefix: str):
        module = cls(**payload[f"{prefix}_config"]).to(device)
        module.load_state_dict(payload[f"{prefix}_state"])
        module.eval()  # 推理模式;继续训练的话调用方自行 train()
        return module

    return LoadedCheckpoint(
        stage="1b",
        unary_kfa=build(LearnableUnaryKFA, "unary_kfa"),
        pairwise_kfa=build(LearnablePairwiseKFA, "pairwise_kfa"),
        fact_selector=build(LearnableFactSelector, "fact_selector"),
        projector=build(MLPProjector, "projector"),
        extra=payload.get("extra", {}),
    )
