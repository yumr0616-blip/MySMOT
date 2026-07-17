"""可学习 Unary KFA(Stage-1a):soft+hard 双读出的关键帧注意力槽位。

对应 §4/§6 的核心机制:
  - scorer MLP 对每个观测帧的特征打一个显著性分数;
  - soft 读出:softmax(scores) 加权的特征值向量(完全可导)——它经
    Pipeline 拼进 projector 输入,变成 soft token 进入冻结 MLLM,是
    训练信号回传到 scorer 的唯一通路;
  - hard 读出:同一组分数的 top-k 选帧(离散)——被选中的帧作为图像
    证据发给 MLLM,本身不可导。
  hard 没有自己的损失:它与 soft 共享同一个 scorer,soft 通路的梯度
  "顺便"训练了 hard 的选择依据,这就是设计文档说的"硬 top-k 搭软
  读出的梯度便车"(straight-through 式)。

输入特征来自 smot.frame_features.geometric_frame_features(Stage-1a
第一版是几何/运动特征;视觉塔特征以后拼接在其后,这里只需改 in_dim)。
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from smot.frame_features import FRAME_FEATURE_DIM
from smot.kfa import KeyFrameSelection
from smot.types import FramePresence


class LearnableUnaryKFA(nn.Module):
    """实现 smot.kfa.UnaryKFA Protocol(推理侧 select()),同时是标准
    nn.Module(训练侧 forward())。"""

    def __init__(
        self,
        in_dim: int = FRAME_FEATURE_DIM,
        hidden_dim: int = 64,
        out_dim: int = 32,
    ):
        super().__init__()
        self.in_dim = in_dim  # 逐帧特征维度(默认等于 FRAME_FEATURE_DIM=9)
        self.out_dim = out_dim  # soft 读出向量的维度,会拼进 projector 的输入
        self.scorer = nn.Sequential(  # 每帧特征 -> 1 个显著性分数(标量)
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Linear(in_dim, out_dim)  # 每帧特征 -> out_dim 维"值"向量,供 soft 加权求和

    def forward(
        self, features: torch.Tensor, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features: (T, in_dim) 单条轨迹的逐帧特征。

        返回 (hard_indices, soft_vector):
          hard_indices (k,)   分数 top-k 的帧下标(离散,无梯度);
          soft_vector (out_dim,) softmax 注意力加权的 value 读出(可导)。
        """
        scores = self.scorer(features).squeeze(-1)  # (T, 1) -> (T,) 每帧一个分数
        probs = torch.softmax(scores, dim=-1)  # 分数 -> 全体 T 帧上的注意力权重,和为 1
        soft_vector = probs @ self.value(features)  # (T,) @ (T, out_dim) -> (out_dim,) 加权和
        k = min(top_k, features.shape[0])  # 帧数不足 top_k 时,最多取全部帧
        hard_indices = torch.topk(scores, k).indices  # 分数最高的 k 个帧下标(离散选择)
        return hard_indices, soft_vector

    def select(
        self,
        track_id: int,
        frames: list[FramePresence],
        top_k: int,
        features: Optional[Sequence[tuple[float, ...]]] = None,
    ) -> KeyFrameSelection:
        """推理侧入口(UnaryKFA Protocol)。

        features 必须由 Pipeline 的 frame_feature_fn 注入——缺失时直接
        报错而不是静默退化成等间隔选帧:静默退化会让"模型训好了但推理
        没用上"这类错误完全不可见,宁可当场炸。
        """
        if not frames:
            return KeyFrameSelection(key_frames=(), soft_token=None)
        if features is None:
            raise ValueError(
                "LearnableUnaryKFA 需要逐帧特征:请在构造 Pipeline 时注入 "
                "frame_feature_fn(例如 smot.frame_features."
                "geometric_frame_features)"
            )
        if len(features) != len(frames):
            raise ValueError(
                f"features 数量({len(features)})与 frames 数量({len(frames)})"
                f"不一致(track_id={track_id})"
            )
        device = next(self.parameters()).device
        feats = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():  # 推理路径(select)不需要梯度;训练时走 forward() 本身
            hard_indices, soft_vector = self.forward(feats, top_k)
        # hard_indices 是打分产生的下标顺序(未必按时间排序),转回帧号后
        # 重新按时间排序,保证下游(渲染、prompt)看到的证据帧是时间有序的。
        key_frames = tuple(sorted(frames[i].t for i in hard_indices.tolist()))
        soft_token = tuple(float(v) for v in soft_vector.cpu())  # 转成纯 Python tuple,脱离 torch
        return KeyFrameSelection(key_frames=key_frames, soft_token=soft_token)
