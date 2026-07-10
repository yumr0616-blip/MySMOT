"""Stage-1a 可学习组件(unary KFA + projector)的 checkpoint 存取。

训练(smot.ml.training)与推理(examples/run_bensmot_real.py --checkpoint)
共用同一格式,构造参数一并存盘——加载侧不需要再记得训练时的维度配置,
配置漂移(比如换了 n_tokens 之后加载旧档)会在 load_state_dict 的形状
校验里当场暴露。
"""
from __future__ import annotations

from pathlib import Path

import torch

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
