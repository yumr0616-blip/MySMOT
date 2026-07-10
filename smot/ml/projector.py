"""可学习 Projector(Stage-1a):池化向量 -> m 个 d_llm 维 soft token。

对应 §4:一个小的残差 MLP,把 KFA soft 读出 + 事实池化拼成的向量映射
到冻结 MLLM 的输入嵌入空间。

输出尺度设计(梯度门禁曾抓出的教训,记录在此):
  - 不做 LoRA 式的最后一层零初始化——y = W_out·h 里 W_out=0 会让
    dL/dh = W_out^T·(dL/dy) = 0,第 0 步就把上游(input_proj/block/KFA)
    的梯度全部堵死,梯度门禁直接 FAIL;
  - 而 LayerNorm 的默认单位增益又会让每个 soft token 的范数达到
    sqrt(d_llm)(≈45 @ d_llm=2048),远大于真实词嵌入的典型范数,
    注入过强会在训练早期扰乱冻结 LM;
  - 所以采用 prompt-tuning 的惯例:输出经 LayerNorm 后,用 output_gain
    (调用方传入冻结 LM 词嵌入的 RMS)初始化增益,让 soft token 在
    训练起点上就与真实 token 嵌入同尺度,梯度又全程可达。
"""
from __future__ import annotations

import torch
from torch import nn


class MLPProjector(nn.Module):
    """实现 smot.projector.Projector Protocol(推理侧 project()),同时是
    标准 nn.Module(训练侧 forward())。"""

    def __init__(
        self,
        in_dim: int,
        d_llm: int,
        n_tokens: int = 4,
        hidden_dim: int = 256,
        output_gain: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim  # 输入池化向量的维度(事实池化 4 维 [+ KFA soft 向量])
        self.d_llm = d_llm  # 冻结 MLLM 的词嵌入维度,soft token 必须匹配这个维度
        self.n_tokens = n_tokens  # 每次调用产出的 soft token 个数 m
        self.input_proj = nn.Linear(in_dim, hidden_dim)  # 把小维度输入升到隐藏维
        self.block = nn.Sequential(  # 单个残差块:LN -> Linear -> GELU -> Linear
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Linear(hidden_dim, n_tokens * d_llm)  # 一次性输出 m 个 token 拼在一起
        self.norm = nn.LayerNorm(d_llm)  # 对每个 token 的 d_llm 维分别归一化
        # output_gain 应传冻结 LM 词嵌入的 RMS(见模块 docstring):
        # LayerNorm 归一化到单位方差后,按嵌入尺度整体缩放。增益本身
        # 仍是可学习参数,这里只是初始化。
        nn.init.constant_(self.norm.weight, output_gain)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: (B, in_dim) 或 (in_dim,)。返回 (B, n_tokens, d_llm)。"""
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)  # 补 batch 维,统一按 (B, in_dim) 处理
        hidden = self.input_proj(pooled)  # (B, hidden_dim)
        hidden = hidden + self.block(hidden)  # 残差:即使 block 学不到东西也不退化
        tokens = self.out(hidden).reshape(-1, self.n_tokens, self.d_llm)  # 拆回 (B, m, d_llm)
        return self.norm(tokens)  # LayerNorm 作用在最后一维(d_llm),逐 token 独立归一化

    def project(
        self, pooled_vector: tuple[float, ...]
    ) -> tuple[tuple[float, ...], ...]:
        """推理侧入口(Projector Protocol)。

        Pipeline 在不同调用点喂进来的池化向量维度不同:instance 任务是
        事实池化(4 维)+ unary KFA soft 向量;interaction/video 在
        pairwise KFA 可学习之前(Stage-1b)只有事实池化的 4 维。不足
        in_dim 的输入按"缺失分量为零"补齐(零填充 = 该信息源缺席,
        与零初始化的语义一致);超出 in_dim 是布线错误,直接报错。
        空输入(既无事实也无 KFA 向量)返回空,不产 token。
        """
        if not pooled_vector:
            return ()
        if len(pooled_vector) > self.in_dim:
            raise ValueError(
                f"池化向量维度 {len(pooled_vector)} 超过 projector 的 in_dim "
                f"{self.in_dim}——构造 MLPProjector 时的 in_dim 与 Pipeline "
                f"布线不匹配"
            )
        padded = tuple(pooled_vector) + (0.0,) * (self.in_dim - len(pooled_vector))
        device = next(self.parameters()).device  # 与模块权重同设备(CPU/CUDA)
        with torch.no_grad():  # 推理路径不需要构建计算图,省显存/加速
            tokens = self.forward(
                torch.tensor(padded, dtype=torch.float32, device=device)
            )
        # 去掉 forward() 补上的 batch 维,搬回 CPU 转成纯 Python 嵌套 tuple
        # (Protocol 约定的返回类型是 tuple,不是 tensor——推理侧不依赖 torch)。
        return tuple(
            tuple(float(v) for v in row) for row in tokens.squeeze(0).cpu()
        )
