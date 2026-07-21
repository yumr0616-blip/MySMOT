"""P4 前置：预测轨迹 <-> gold 轨迹的 IoU 匹配对齐。

现状问题：smot.eval 的交互 F1 / instance 覆盖率全靠 subject_id /
object_id / track_id **直接相等**配对预测和 gold——StubTracker 直接读
GT 轨迹时这天然成立（预测 id 就是 gold id）。换成真实跟踪器
（YOLO/RT-DETR+ByteTrack 或 SAM2）后，跟踪器会分配自己的一套 id，与
gold id 毫无关系，直接相等匹配会整体失效。

这个模块解决"配对"这一步本身，不改 smot.eval：调用方在两边轨迹都还
在内存里时（拿到预测轨迹 + 有 gold 轨迹可比对的评测场景）调用
match_tracks() 拿到 pred_id -> gold_id 的映射，把预测断言里的 id 重写
成 gold id 之后，再走现有的 smot.eval 全套逻辑——eval.py 因此不需要
认识"预测 id 和 gold id 可能不是同一套体系"这件事。

之所以不放进 smot.eval：id 匹配需要逐帧空间坐标（box），而 eval.py
消费的是 PipelineResult 序列化后的 JSON，其中的 instance/interaction
断言只有 time_span（时间跨度），没有逐帧 box——序列化之前的 Trajectory
对象才有完整的 per_frame 坐标，这一步必须在那个阶段做。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from smot._geometry import iou
from smot.types import Trajectory


def _track_iou(pred_traj: Trajectory, gold_traj: Trajectory) -> float:
    """两条轨迹的"轨迹级 IoU":逐帧 box IoU 在两者观测帧号**并集**上取
    均值——只有一方有观测的帧按 0 计入。

    用并集而不是交集，是为了不让"只有 1 帧共同观测且刚好对得很准"的
    短暂重叠被算成完美匹配：分母是并集帧数，覆盖时间越短、共同观测帧
    占比越低，分数自然被拉低，这与轨迹匹配"整体轨迹要对得上"的直觉
    一致（呼应 smot.eval._time_iou 对 time_span 的并集处理,但这里是
    逐帧空间量而不是单一时间区间)。
    """
    pred_by_t = {fp.t: fp.box for fp in pred_traj.per_frame}
    gold_by_t = {fp.t: fp.box for fp in gold_traj.per_frame}
    all_ts = set(pred_by_t) | set(gold_by_t)
    if not all_ts:
        return 0.0
    total = sum(
        iou(pred_by_t[t], gold_by_t[t]) for t in all_ts if t in pred_by_t and t in gold_by_t
    )
    return total / len(all_ts)


@dataclass(frozen=True)
class TrackMatchResult:
    """一次匹配的完整结果。"""

    id_map: dict[int, int]  # 预测 track_id -> 匹配上的 gold track_id
    matched_ious: dict[int, float]  # 预测 track_id -> 该匹配的轨迹级 IoU
    unmatched_pred: tuple[int, ...]  # 没匹配上任何 gold 的预测 track_id(误跟踪/幻觉轨迹,FP)
    unmatched_gold: tuple[int, ...]  # 没被任何预测匹配上的 gold track_id(漏跟踪,FN)

    def stats(self) -> dict:
        """轨迹级 precision/recall/F1 + 匹配上那部分的平均 IoU(质量,
        不只是"配没配上")。分母为 0 时约定为 0,与 smot.eval.PRF 同一
        约定,避免调用方逐处判空。
        """
        n_matched = len(self.id_map)
        n_pred = n_matched + len(self.unmatched_pred)
        n_gold = n_matched + len(self.unmatched_gold)
        precision = n_matched / n_pred if n_pred else 0.0
        recall = n_matched / n_gold if n_gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        mean_iou = (
            sum(self.matched_ious.values()) / n_matched if n_matched else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_matched": n_matched,
            "n_pred": n_pred,
            "n_gold": n_gold,
            "mean_matched_iou": mean_iou,
        }


def match_tracks(
    pred_trajectories: Sequence[Trajectory],
    gold_trajectories: Sequence[Trajectory],
    iou_threshold: float = 0.5,
) -> TrackMatchResult:
    """预测轨迹与 gold 轨迹的贪心 IoU 匹配。

    贪心而非 Hungarian 最优匹配:每轮在所有"仍未匹配"的 (pred, gold)
    对里选轨迹级 IoU 最高的一对,IoU 达到阈值才接受、双方移出候选池,
    重复到没有候选对能达到阈值为止。单视频轨迹数通常是个位数,贪心和
    最优解在这个规模下几乎不会给出不同结果,换 Hungarian(需要 scipy,
    当前不是依赖)收益很小——这里保持 smot 核心包 stdlib-only 的既有
    风格(见 README "Stack" 一节)。

    iou_threshold 以下的最高分也不接受:轨迹级 IoU 太低说明这对根本
    不是同一个目标,应该分别计入 FP/FN 而不是牵强凑一对。
    """
    pending: list[tuple[float, int, int]] = []
    for p in pred_trajectories:
        for g in gold_trajectories:
            score = _track_iou(p, g)
            if score >= iou_threshold:
                pending.append((score, p.track_id, g.track_id))
    pending.sort(key=lambda x: x[0], reverse=True)  # 高分优先

    id_map: dict[int, int] = {}
    matched_ious: dict[int, float] = {}
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    for score, pred_id, gold_id in pending:
        if pred_id in used_pred or gold_id in used_gold:
            continue  # 双方有一个已经被更高分的一对占用,让给已成立的匹配
        id_map[pred_id] = gold_id
        matched_ious[pred_id] = score
        used_pred.add(pred_id)
        used_gold.add(gold_id)

    unmatched_pred = tuple(
        sorted(p.track_id for p in pred_trajectories if p.track_id not in used_pred)
    )
    unmatched_gold = tuple(
        sorted(g.track_id for g in gold_trajectories if g.track_id not in used_gold)
    )
    return TrackMatchResult(id_map, matched_ious, unmatched_pred, unmatched_gold)


def remap_assertion_ids(assertions: list[dict], id_map: dict[int, int]) -> list[dict]:
    """把序列化断言(instance/interaction)里的 track_id / subject_id /
    object_id 按 id_map 重写成 gold id,供后续接入 smot.eval 时使用
    ——不在 id_map 里的 id(即 unmatched_pred,匹配阶段已判定为 FP)保
    留原值,调用方应在此之前已经用 unmatched_pred 单独计入 tracking
    指标,这里不重复处理、也不静默丢弃这些断言(留给下游 interaction/
    instance 匹配去处理,它们对不上任何 gold id,自然算不中)。
    """
    remapped = []
    for a in assertions:
        a = dict(a)
        for key in ("track_id", "subject_id", "object_id"):
            if key in a and a[key] in id_map:
                a[key] = id_map[a[key]]
        remapped.append(a)
    return remapped
