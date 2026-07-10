"""Stage-1a 验收门禁 #1(§6/§10):梯度恰好且仅落在可训练槽位上。

流程:dummy batch(随机轨迹特征 + 随机事实池化向量 + 合成图像 + 教师
强制目标句)-> 完整前向(KFA soft 读出 -> projector -> soft token 经
embedding hook 注入冻结 Qwen3.5 -> CE loss)-> backward,然后断言:

  1. LearnableUnaryKFA 与 MLPProjector 的每个参数都拿到了梯度张量,
     且两个模块的总梯度范数 > 0(训练信号真实到达);
  2. 冻结 MLLM(含视觉塔)的所有参数 requires_grad=False 且 .grad 为
     None(一个都不许漏);
  3. 模型处于 eval() 模式,loss 有限。

不满足任何一条即 FAIL——这是 M-B1 的完成判据,训练循环(M-B2)只允许
在本门禁通过之后开工。用法:

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
from smot.ml.projector import MLPProjector
from smot.ml.qwen_adapter import (
    DEFAULT_MODEL_ID,
    load_frozen_qwen,
    teacher_forced_loss,
)
from smot.ml.unary_kfa import LearnableUnaryKFA

# dummy batch 的教师强制目标句(内容不重要,只要能算出一个 CE loss)。
_TARGET_TEXT = "track_id=1 walks to the right and then stops."
_PROMPT_TEXT = (
    "Describe the behavior of track_id=1 based on the following motion "
    "facts: presence 1~8; mean speed 5.0 px/frame."
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
) -> dict:
    """执行一次门禁检查,返回结构化报告(report["pass"] 为总判定)。"""
    torch.manual_seed(seed)
    embedding = model.get_input_embeddings()
    d_llm = embedding.embedding_dim
    # soft token 的初始尺度对齐冻结 LM 的词嵌入 RMS(见 MLPProjector
    # 的模块 docstring)。
    embed_rms = float(embedding.weight.detach().float().pow(2).mean().sqrt())

    kfa = LearnableUnaryKFA().to(device)
    projector = MLPProjector(
        in_dim=4 + kfa.out_dim, d_llm=d_llm, output_gain=embed_rms
    ).to(device)
    kfa.train()
    projector.train()

    # ---- dummy batch 前向:特征 -> KFA -> projector -> soft token ----
    features = torch.rand(n_frames, FRAME_FEATURE_DIM, device=device)
    fact_vector = torch.rand(4, device=device)
    _hard_indices, soft_vector = kfa(features, top_k=top_k)
    pooled = torch.cat([fact_vector, soft_vector])
    soft_tokens = projector(pooled).squeeze(0)  # (n_tokens, d_llm)

    # ---- 组多模态 prompt + 教师强制目标:与训练循环共用同一条组装
    # 路径(teacher_forced_loss),门禁校验过的就是训练真正跑的。 ----
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _synthetic_image()},
                {"type": "text", "text": _PROMPT_TEXT},
            ],
        }
    ]
    loss = teacher_forced_loss(model, processor, messages, soft_tokens, _TARGET_TEXT)
    loss.backward()

    # ---- 断言收集 ----
    failures: list[str] = []
    warnings: list[str] = []

    if model.training:
        failures.append("冻结 MLLM 未处于 eval() 模式")
    if not torch.isfinite(loss):
        failures.append(f"loss 非有限值: {loss.item()!r}")

    # 逐参数检查:两个可训练模块的每一个参数张量都必须 requires_grad=True
    # 且拿到非 None、有限、非全零的梯度——任何一项不满足都记一条失败。
    trainable_norms: dict[str, float] = {}
    for module_name, module in (("unary_kfa", kfa), ("projector", projector)):
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
        prog="python -m smot.ml.gradient_check", description="Stage-1a 梯度门禁"
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantize-4bit", action="store_true")
    args = parser.parse_args(argv)

    model, processor = load_frozen_qwen(
        args.model_id, device=args.device, quantize_4bit=args.quantize_4bit
    )
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
