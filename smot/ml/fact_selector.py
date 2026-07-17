"""可学习 Fact Selector(Stage-1b):事实选择槽位(soft+hard 双读出)。

对应 §4/§6:事实的 value 永远不学习(确定性抽取),学习的只是"该选
哪些事实放进 transcript"。机制与 KFA 槽位完全同构:
  - scorer MLP 对 scope 内的每条事实打分;
  - soft 读出:softmax(scores) 加权的 value 向量(可导)——它取代
    Stage-0/1a 的"事实 embed 均值池化"成为 projector 输入的 fact 槽位,
    是训练信号回传到 scorer 的唯一通路(均值池化是它的一个特例:
    权重全等 + value 为恒等映射);
  - hard 读出:同一组分数的 top-k 事实,渲染成 transcript 文本进 prompt
    (离散,无自有梯度,搭 soft 通路的便车)。

渲染约定:选中的事实按 DeterministicFactSelector 同款优先级顺序排序后
用同一个 render_fact 渲染——学习改变"选哪些",不改变 transcript 的
文本分布(顺序/格式),避免 prompt 风格随训练漂移。

每条事实的打分特征 = embed(4 维,norm_value 已按数据集统计量 z-score,
见 smot.fact_norm)+ 类型 one-hot(线性打分器直接可用的类型信号;
embed[0] 的序数 type_index 保留不动,信息冗余无害)。
"""
from __future__ import annotations

import torch
from torch import nn

from smot.fact_selector import FactSelection, SelectionContext, render_fact, scoped_facts
from smot.types import FACT_TYPE_ORDER, Fact

# 打分特征维度:embed 固定 4 维 + 事实类型 one-hot。
FACT_SCORE_DIM = 4 + len(FACT_TYPE_ORDER)


def fact_scoring_features(facts: list[Fact]) -> tuple[tuple[float, ...], ...]:
    """把每条事实压成 FACT_SCORE_DIM 维打分向量(与输入逐条对齐)。
    训练循环与推理侧 select() 共用,保证两侧看到同一种特征。"""
    rows: list[tuple[float, ...]] = []
    for fact in facts:
        onehot = [0.0] * len(FACT_TYPE_ORDER)
        try:
            onehot[FACT_TYPE_ORDER.index(fact.type)] = 1.0
        except ValueError:
            pass  # 未知类型:one-hot 全零(embed 分量仍在,不至于完全无信号)
        rows.append(tuple(fact.embed) + tuple(onehot))
    return tuple(rows)


def fact_priority_indices(facts: list[Fact]) -> tuple[int, ...]:
    """每条事实的类型优先级下标(DeterministicFactSelector 同款顺序,
    未知类型排最后)。训练循环在样本构造期预计算它,推理侧 select()
    在线计算——两侧经 order_selected 使用同一个排序契约。"""
    out = []
    for fact in facts:
        try:
            out.append(FACT_TYPE_ORDER.index(fact.type))
        except ValueError:
            out.append(len(FACT_TYPE_ORDER))
    return tuple(out)


def order_selected(
    hard_indices: list[int], priorities: tuple[int, ...]
) -> list[int]:
    """把 hard top-k 选出的事实下标按(类型优先级, 原始顺序)重排——
    这是渲染进 transcript 的顺序契约:学习决定"选哪些",顺序/格式与
    确定性实现保持一致,避免 prompt 文本分布随训练漂移。"""
    return sorted(hard_indices, key=lambda i: (priorities[i], i))


class LearnableFactSelector(nn.Module):
    """实现 smot.fact_selector.FactSelector Protocol(推理侧 select()),
    同时是标准 nn.Module(训练侧 forward())。"""

    def __init__(
        self,
        in_dim: int = FACT_SCORE_DIM,
        hidden_dim: int = 32,
        out_dim: int = 16,
    ):
        super().__init__()
        self.in_dim = in_dim  # 每条事实的打分特征维度(默认 FACT_SCORE_DIM=9)
        self.out_dim = out_dim  # soft 读出向量的维度,占 projector 输入的 fact 槽位
        self.scorer = nn.Sequential(  # 每条事实 -> 1 个分数(标量)
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Linear(in_dim, out_dim)  # 每条事实 -> out_dim 维"值"向量

    def forward(
        self, features: torch.Tensor, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features: (N, in_dim) scope 内全部事实的打分特征。

        返回 (hard_indices, soft_vector):
          hard_indices (k,)   分数 top-k 的事实下标(离散,无梯度);
          soft_vector (out_dim,) softmax 注意力加权的 value 读出(可导)。
        """
        scores = self.scorer(features).squeeze(-1)  # (N, 1) -> (N,)
        probs = torch.softmax(scores, dim=-1)  # scope 内全部事实上的注意力权重
        soft_vector = probs @ self.value(features)  # (out_dim,) 加权和
        k = min(top_k, features.shape[0])
        hard_indices = torch.topk(scores, k).indices  # 离散选择,无自有梯度
        return hard_indices, soft_vector

    def zero_soft(self) -> tuple[float, ...]:
        """"该信息源缺席"的全零 soft 读出(scope 内没有任何事实时用),
        维度与正常读出一致,保证 projector 输入的槽位布局不塌缩。"""
        return (0.0,) * self.out_dim

    def select(self, facts: list[Fact], query_context: SelectionContext) -> FactSelection:
        """推理侧入口(FactSelector Protocol)。scope 过滤与渲染格式都与
        DeterministicFactSelector 完全一致,唯一不同是"选哪 top_k 条"
        由打分决定,且 soft_token 是真实读出而非 None。"""
        scoped = scoped_facts(facts, query_context.scope)
        if not scoped:
            return FactSelection(selected_facts=(), soft_token=self.zero_soft(), text="")
        device = next(self.parameters()).device
        feats = torch.tensor(
            fact_scoring_features(scoped), dtype=torch.float32, device=device
        )
        with torch.no_grad():  # 推理路径不需要梯度;训练时走 forward() 本身
            hard_indices, soft_vector = self.forward(feats, query_context.top_k)
        # 选中下标按(类型优先级, 原始顺序)重排后渲染:与确定性实现的
        # transcript 顺序约定一致(见 order_selected)。
        chosen = order_selected(
            hard_indices.tolist(), fact_priority_indices(scoped)
        )
        selected = tuple(scoped[i] for i in chosen)
        text = "; ".join(render_fact(f) for f in selected)
        soft_token = tuple(float(v) for v in soft_vector.cpu())
        return FactSelection(selected_facts=selected, soft_token=soft_token, text=text)
