"""可学习 Pairwise KFA(Stage-1b):候选边关键帧注意力槽位。

与 smot.ml.unary_kfa.LearnableUnaryKFA 完全同构的 soft+hard 双读出:
  - scorer MLP 对候选边的每个共同观测帧的 pair 特征打显著性分数;
  - soft 读出:softmax(scores) 加权的 value 向量(可导)——经 Pipeline
    拼进 projector 输入的 pairwise 槽位,是训练信号回传的唯一通路;
  - hard 读出:同一组分数的 top-k 选帧(离散),作为 interaction 调用的
    图像证据发给 MLLM,搭 soft 通路的梯度便车。

输入特征来自 smot.pair_features.pair_feature_vectors(相对几何 + 时间,
方向性信息就在 rel_pos/rel_vel/orient 的符号里;视觉塔特征是后续增强,
接入时只需改 in_dim)。
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from smot.event_filter import EventCandidate
from smot.kfa import KeyFrameSelection, _evenly_spaced
from smot.pair_features import PAIR_FEATURE_DIM
from smot.types import PairFeature


class LearnablePairwiseKFA(nn.Module):
    """实现 smot.kfa.PairwiseKFA Protocol(推理侧 select()),同时是标准
    nn.Module(训练侧 forward())。"""

    def __init__(
        self,
        in_dim: int = PAIR_FEATURE_DIM,
        hidden_dim: int = 64,
        out_dim: int = 32,
    ):
        super().__init__()
        self.in_dim = in_dim  # 逐帧 pair 特征维度(默认等于 PAIR_FEATURE_DIM=8)
        self.out_dim = out_dim  # soft 读出向量的维度,占 projector 输入的 pairwise 槽位
        self.scorer = nn.Sequential(  # 每帧 pair 特征 -> 1 个显著性分数(标量)
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Linear(in_dim, out_dim)  # 每帧 pair 特征 -> out_dim 维"值"向量

    def forward(
        self, features: torch.Tensor, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features: (T, in_dim) 单条候选边的逐帧 pair 特征。

        返回 (hard_indices, soft_vector):
          hard_indices (k,)   分数 top-k 的帧下标(离散,无梯度);
          soft_vector (out_dim,) softmax 注意力加权的 value 读出(可导)。
        """
        scores = self.scorer(features).squeeze(-1)  # (T, 1) -> (T,)
        probs = torch.softmax(scores, dim=-1)  # 全体 T 帧上的注意力权重,和为 1
        soft_vector = probs @ self.value(features)  # (out_dim,) 加权和
        k = min(top_k, features.shape[0])  # 帧数不足 top_k 时,最多取全部帧
        hard_indices = torch.topk(scores, k).indices  # 离散选择,无自有梯度
        return hard_indices, soft_vector

    def select(
        self,
        edge: tuple[int, int],
        event_candidate: EventCandidate,
        top_k: int,
        pair_features: Sequence[PairFeature] = (),
        features: Optional[Sequence[tuple[float, ...]]] = None,
    ) -> KeyFrameSelection:
        """推理侧入口(PairwiseKFA Protocol)。

        features 必须由 Pipeline 的 pair_feature_fn 注入——缺失时直接
        报错而不是静默退化(与 LearnableUnaryKFA.select 同一哲学:
        "模型训好了但推理没用上"必须当场炸,不能藏在评测数字里)。

        pair_features 为空(候选帧上双方无共同观测,算不出 pair 特征)
        是数据侧的合法情况,不是布线错误:hard 退回候选帧等间隔抽稀
        (仍给 MLLM 看图),soft 读出为全零向量(= 该信息源缺席,与
        projector 槽位的补零语义一致)。
        """
        if not pair_features:
            chosen = _evenly_spaced(event_candidate.candidate_frames, top_k)
            return KeyFrameSelection(
                key_frames=chosen, soft_token=(0.0,) * self.out_dim
            )
        if features is None:
            raise ValueError(
                "LearnablePairwiseKFA 需要向量化 pair 特征:请在构造 Pipeline "
                "时注入 pair_feature_fn(例如 smot.pair_features."
                "pair_feature_vectors)"
            )
        if len(features) != len(pair_features):
            raise ValueError(
                f"features 数量({len(features)})与 pair_features 数量"
                f"({len(pair_features)})不一致(edge={edge})"
            )
        device = next(self.parameters()).device
        feats = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():  # 推理路径不需要梯度;训练时走 forward() 本身
            hard_indices, soft_vector = self.forward(feats, top_k)
        # 打分下标 -> 帧号,重新按时间排序(下游渲染/prompt 期望时间有序)。
        key_frames = tuple(
            sorted(pair_features[i].t for i in hard_indices.tolist())
        )
        soft_token = tuple(float(v) for v in soft_vector.cpu())
        return KeyFrameSelection(key_frames=key_frames, soft_token=soft_token)
