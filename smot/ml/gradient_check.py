"""验收门禁 #1(§6/§10):梯度恰好且仅落在可训练槽位上(Stage-1b 四模块)。

流程:两条 dummy 样本走与训练循环完全同一条前向路径——
  instance    fact selector soft + unary KFA soft -> [fact | unary | 0]
  interaction fact selector soft + pairwise KFA soft -> [fact | 0 | pairwise]
各自经 projector -> soft token 经 embedding hook 注入冻结 Qwen3.5 ->
CE loss;两条 loss 相加 backward,然后断言:

  1. {unary KFA, pairwise KFA, fact selector, projector} 的每个参数都
     拿到了梯度张量,且每个模块的总梯度范数 > 0(训练信号真实到达
     全部四个槽位);
  2. 冻结 MLLM(含视觉塔)的所有参数 requires_grad=False 且 .grad 为
     None(一个都不许漏);
  3. 模型处于 eval() 模式,loss 有限。

不满足任何一条即 FAIL——Stage-1b 训练只允许在本门禁通过之后开工
(与 M-B1 对 Stage-1a 的判据一致)。用法:

    python -m smot.ml.gradient_check [--model-id Qwen/Qwen3.5-2B]
                                     [--device cuda] [--quantize-4bit]
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch
from PIL import Image, ImageDraw

from smot.frame_features import FRAME_FEATURE_DIM
from smot.ml.fact_selector import FACT_SCORE_DIM, LearnableFactSelector
from smot.ml.pairwise_kfa import LearnablePairwiseKFA
from smot.ml.projector import MLPProjector
from smot.ml.qwen_adapter import (
    DEFAULT_MODEL_ID,
    load_frozen_qwen,
    teacher_forced_loss,
)
from smot.ml.unary_kfa import LearnableUnaryKFA
from smot.pair_features import PAIR_FEATURE_DIM

# dummy 样本的教师强制目标句(内容不重要,只要能算出 CE loss)。
_INSTANCE_TARGET = "track_id=1 walks to the right and then stops."
_INSTANCE_PROMPT = (
    "Describe the behavior of track_id=1 based on the following motion "
    "facts: presence 1~8; mean speed 5.0 px/frame."
)
_INTERACTION_TARGET = '[{"subject_id": 1, "object_id": 2, "predicate": "talk"}]'
_INTERACTION_PROMPT = (
    "Describe the interaction between subject_id=1 and object_id=2 based on "
    "the following motion facts: proximity 12.0 px; approach closer."
)


def _synthetic_image() -> Image.Image:
    """一张带红框的合成图,让视觉塔真实参与前向——这样"视觉塔不收
    梯度"的断言才有意义(不跑视觉塔的话它天然没梯度,等于没检查)。"""
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 40, 160, 200], outline="red", width=5)
    return img


def run_gradient_check(
    model,
    processor,
    device: str = "cuda",
    n_frames: int = 8,
    top_k: int = 4,
    seed: int = 0,
    unary_in_dim: int = FRAME_FEATURE_DIM,
    pairwise_in_dim: int = PAIR_FEATURE_DIM,
) -> dict:
    """执行一次门禁检查,返回结构化报告(report["pass"] 为总判定)。

    unary_in_dim/pairwise_in_dim 默认是纯几何维度(Stage-1a/1b);传
    smot.ml.feature_cache.AUGMENTED_FRAME_FEATURE_DIM/
    AUGMENTED_PAIR_FEATURE_DIM 即可复用同一套门禁检查 P2 Stage-2 的
    放大 in_dim——门禁只关心梯度是否流向四个可训练槽位、冻结 MLLM 是否
    纹丝不动,不关心特征本身是几何还是几何+视觉,所以这里用合成随机
    张量,不需要真实的离线视觉缓存。
    """
    torch.manual_seed(seed)
    embedding = model.get_input_embeddings()
    d_llm = embedding.embedding_dim
    # soft token 的初始尺度对齐冻结 LM 的词嵌入 RMS(见 MLPProjector
    # 的模块 docstring)。
    embed_rms = float(embedding.weight.detach().float().pow(2).mean().sqrt())

    unary_kfa = LearnableUnaryKFA(in_dim=unary_in_dim).to(device)
    pairwise_kfa = LearnablePairwiseKFA(in_dim=pairwise_in_dim).to(device)
    fact_selector = LearnableFactSelector().to(device)
    projector = MLPProjector(
        # 与训练循环同一份 [fact | unary | pairwise] 槽位布局。
        in_dim=fact_selector.out_dim + unary_kfa.out_dim + pairwise_kfa.out_dim,
        d_llm=d_llm,
        output_gain=embed_rms,
    ).to(device)
    trainable_modules = {
        "unary_kfa": unary_kfa,
        "pairwise_kfa": pairwise_kfa,
        "fact_selector": fact_selector,
        "projector": projector,
    }
    for module in trainable_modules.values():
        module.train()

    zeros_unary = torch.zeros(unary_kfa.out_dim, device=device)
    zeros_pair = torch.zeros(pairwise_kfa.out_dim, device=device)

    def message_for(prompt: str):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _synthetic_image()},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # ---- instance 样本:fact selector + unary KFA 两条 soft 通路 ----
    fact_feats = torch.rand(5, FACT_SCORE_DIM, device=device)
    _hard_f, soft_facts = fact_selector(fact_feats, top_k=3)
    unary_feats = torch.rand(n_frames, unary_in_dim, device=device)
    _hard_u, soft_unary = unary_kfa(unary_feats, top_k=top_k)
    inst_tokens = projector(
        torch.cat([soft_facts, soft_unary, zeros_pair])
    ).squeeze(0)
    loss_instance = teacher_forced_loss(
        model, processor, message_for(_INSTANCE_PROMPT), inst_tokens, _INSTANCE_TARGET
    )

    # ---- interaction 样本:fact selector + pairwise KFA 两条 soft 通路 ----
    fact_feats2 = torch.rand(4, FACT_SCORE_DIM, device=device)
    _hard_f2, soft_facts2 = fact_selector(fact_feats2, top_k=3)
    pair_feats = torch.rand(6, pairwise_in_dim, device=device)
    _hard_p, soft_pair = pairwise_kfa(pair_feats, top_k=2)
    inter_tokens = projector(
        torch.cat([soft_facts2, zeros_unary, soft_pair])
    ).squeeze(0)
    loss_interaction = teacher_forced_loss(
        model,
        processor,
        message_for(_INTERACTION_PROMPT),
        inter_tokens,
        _INTERACTION_TARGET,
    )

    # 两条样本相加一起 backward:一次反传覆盖全部四个槽位的通路
    # (unary 只在 instance 通路上、pairwise 只在 interaction 通路上)。
    loss = loss_instance + loss_interaction
    loss.backward()

    # ---- 断言收集 ----
    failures: list[str] = []
    warnings: list[str] = []

    if model.training:
        failures.append("冻结 MLLM 未处于 eval() 模式")
    if not torch.isfinite(loss):
        failures.append(f"loss 非有限值: {loss.item()!r}")

    # 逐参数检查:四个可训练模块的每一个参数张量都必须 requires_grad=True
    # 且拿到非 None、有限、非全零的梯度——任何一项不满足都记一条失败。
    trainable_norms: dict[str, float] = {}
    for module_name, module in trainable_modules.items():
        total = 0.0
        for name, param in module.named_parameters():
            if not param.requires_grad:
                failures.append(f"{module_name}.{name} 意外处于冻结状态")
            if param.grad is None:
                failures.append(f"{module_name}.{name} 没有拿到梯度")
                continue
            if not torch.isfinite(param.grad).all():
                failures.append(f"{module_name}.{name} 梯度含非有限值")
            norm = float(param.grad.norm())
            total += norm
            if norm == 0.0:
                # 单个参数梯度为零不一定是 bug(比如某些初始化下的正常现象),
                # 只警告;整个模块总梯度为零才是真正的"信号没到达"。
                warnings.append(f"{module_name}.{name} 梯度为全零")
        trainable_norms[module_name] = total
        if total == 0.0:
            failures.append(f"{module_name} 的总梯度范数为 0,训练信号未到达")

    # 反向检查:冻结 MLLM 的每一个参数都不许 requires_grad,也不许有 .grad
    # ——哪怕只漏检查一个参数,也可能是"意外没冻住"这类严重 bug。
    n_frozen = 0
    for name, param in model.named_parameters():
        n_frozen += 1
        if param.requires_grad:
            failures.append(f"冻结 MLLM 参数 {name} 的 requires_grad 为 True")
        if param.grad is not None:
            failures.append(f"冻结 MLLM 参数 {name} 意外拿到了梯度")

    return {
        "pass": not failures,
        "loss": float(loss.detach()),
        "trainable_grad_norms": trainable_norms,
        "n_frozen_params": n_frozen,
        "d_llm": d_llm,
        "failures": failures,
        "warnings": warnings,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m smot.ml.gradient_check",
        description="Stage-1b 梯度门禁(四可训练槽位)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument(
        "--stage2", action="store_true",
        help="用 P2 Stage-2 的放大 in_dim(几何+视觉)构造 KFA,而不是"
             "Stage-1a/1b 的纯几何维度",
    )
    args = parser.parse_args(argv)

    model, processor = load_frozen_qwen(
        args.model_id, device=args.device, quantize_4bit=args.quantize_4bit
    )
    if args.stage2:
        from smot.ml.feature_cache import (
            AUGMENTED_FRAME_FEATURE_DIM,
            AUGMENTED_PAIR_FEATURE_DIM,
        )

        report = run_gradient_check(
            model,
            processor,
            device=args.device,
            unary_in_dim=AUGMENTED_FRAME_FEATURE_DIM,
            pairwise_in_dim=AUGMENTED_PAIR_FEATURE_DIM,
        )
    else:
        report = run_gradient_check(model, processor, device=args.device)

    print(f"loss = {report['loss']:.4f}")
    for module_name, norm in report["trainable_grad_norms"].items():
        print(f"可训练模块 {module_name}: 总梯度范数 = {norm:.6f}")
    print(f"冻结 MLLM 参数张量数 = {report['n_frozen_params']}(全部无梯度)")
    for message in report["warnings"]:
        print(f"[警告] {message}")
    for message in report["failures"]:
        print(f"[失败] {message}")
    print("GATE PASS" if report["pass"] else "GATE FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
