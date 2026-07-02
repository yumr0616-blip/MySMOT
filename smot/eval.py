"""§7 评测:交互断言的分层 F1、方向准确率、instance 归因覆盖、成本汇总。

评测消费的是 PipelineResult.to_json_dict() 序列化后的落盘产物(纯 JSON
dict/list 形状),不依赖运行时 dataclass——预测文件和标注文件只要满足
§5 的 JSON schema 就能评,与产生它们的代码解耦。

分层 F1 的实现方式:同一个匹配函数,注入不同粒度的 label_map 重跑
(这正是 OutputAssembler 支持注入 canonical_map 的对偶设计):
  - strict          : label_map=None,方向必须一致
  - synonym-merged  : 注入同义词合并表
  - coarse          : 注入粗粒度表(默认用下面的 COARSE_MAP 占位示例)

用法:
    python -m smot.eval pred.json gold.json
文件内容可以是单个 PipelineResult 的 JSON,也可以是它们的列表
(多视频,按下标逐一配对)。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

# coarse 层的占位示例映射:把细粒度规范标签再折叠到粗类。真实评测时
# 应换成随数据集标注规范制定的表(和 CANONICAL_MAP 一样是可注入的)。
COARSE_MAP: dict[str, str] = {
    "approach": "move",
    "recede": "move",
    "follow": "move",
    "contact": "contact",
}


@dataclass(frozen=True)
class PRF:
    """一次匹配的 precision / recall / F1 及其计数基数。"""

    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gold: int
    n_matched: int

    @classmethod
    def from_counts(cls, n_matched: int, n_pred: int, n_gold: int) -> "PRF":
        precision = n_matched / n_pred if n_pred else 0.0
        recall = n_matched / n_gold if n_gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return cls(precision, recall, f1, n_pred, n_gold, n_matched)

    def to_json_dict(self) -> dict:
        return asdict(self)


def _mapped_label(assertion: dict, label_map: Optional[dict[str, str]]) -> str:
    """取断言的规范标签,经 label_map 折叠(表里没有的标签原样保留,
    identity fallback——和 canonical_labels.map_predicate 同一约定)。
    """
    label = assertion["canonical_label"]
    if label_map is not None:
        label = label_map.get(label, label)
    return label


def _interaction_key(
    assertion: dict, label_map: Optional[dict[str, str]], require_direction: bool
) -> tuple:
    ids = (assertion["subject_id"], assertion["object_id"])
    if not require_direction:
        ids = tuple(sorted(ids))
    return (*ids, _mapped_label(assertion, label_map))


def match_interactions(
    pred: list[dict],
    gold: list[dict],
    label_map: Optional[dict[str, str]] = None,
    require_direction: bool = True,
) -> PRF:
    """交互断言的一对一匹配 PRF。匹配键 = (subject_id, object_id, 折叠后
    标签);require_direction=False 时 id 对按无序处理。同键的多条断言按
    多重集配对(min 计数),保证一条 gold 不会被两条 pred 重复"吃掉"。
    """
    pred_keys = Counter(
        _interaction_key(a, label_map, require_direction) for a in pred
    )
    gold_keys = Counter(
        _interaction_key(a, label_map, require_direction) for a in gold
    )
    n_matched = sum((pred_keys & gold_keys).values())
    return PRF.from_counts(n_matched, len(pred), len(gold))


def direction_accuracy(
    pred: list[dict],
    gold: list[dict],
    label_map: Optional[dict[str, str]] = None,
) -> Optional[float]:
    """方向准确率(§7):先按"无序 id 对 + 标签"匹配,再统计匹配成功的
    断言里有序 (subject_id, object_id) 一致的比例——把"事件找对了"和
    "方向说对了"两个能力解耦。没有任何无序匹配时返回 None(无从谈
    方向对错)。
    """
    pred_directed = Counter(_interaction_key(a, label_map, True) for a in pred)
    gold_directed = Counter(_interaction_key(a, label_map, True) for a in gold)
    pred_undirected = Counter(_interaction_key(a, label_map, False) for a in pred)
    gold_undirected = Counter(_interaction_key(a, label_map, False) for a in gold)

    n_matched = sum((pred_undirected & gold_undirected).values())
    if n_matched == 0:
        return None
    # 有序键的多重集交集必然是无序匹配的子集,所以直接可比。
    n_correct = sum((pred_directed & gold_directed).values())
    return n_correct / n_matched


def _time_iou(span_a: list[int], span_b: list[int]) -> float:
    """两个闭区间帧号段的时间 IoU(按包含的帧数计,闭区间所以 +1)。"""
    inter = min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]) + 1
    union = max(span_a[1], span_b[1]) - min(span_a[0], span_b[0]) + 1
    if union <= 0:
        return 0.0
    return max(0, inter) / union


def instance_coverage(pred: list[dict], gold: list[dict]) -> dict:
    """instance 断言的最小归因指标(没有 LLM judge 时可用的部分):
    - track_coverage:gold 里的 track_id 有多大比例在 pred 里拿到了 caption
    - mean_time_iou:双方都覆盖的 track 上,time_span 的平均时间 IoU
    caption 文本本身的质量需要 LLM judge(§7),不在此处评。
    """
    pred_by_id = {a["track_id"]: a for a in pred}
    gold_by_id = {a["track_id"]: a for a in gold}
    if not gold_by_id:
        return {"track_coverage": 0.0, "mean_time_iou": 0.0, "n_pred": len(pred), "n_gold": 0}
    common = sorted(set(pred_by_id) & set(gold_by_id))
    ious = [
        _time_iou(pred_by_id[i]["time_span"], gold_by_id[i]["time_span"]) for i in common
    ]
    return {
        "track_coverage": len(common) / len(gold_by_id),
        "mean_time_iou": sum(ious) / len(ious) if ious else 0.0,
        "n_pred": len(pred),
        "n_gold": len(gold_by_id),
    }


def aggregate_cost(payloads: list[dict]) -> dict:
    """跨多个 PipelineResult 的 cost 字段做汇总(总和 + 均值)。§7 把
    这些量列为一等公民指标,是"learned 选择省 token"论证的对照基线。
    """
    costs = [p.get("cost", {}) for p in payloads]
    keys = sorted({k for c in costs for k in c})
    total = {k: sum(c.get(k, 0) for c in costs) for k in keys}
    n = len(costs)
    mean = {k: total[k] / n for k in keys} if n else {}
    return {"n_videos": n, "total": total, "mean": mean}


def _as_list(payload) -> list[dict]:
    """允许输入是单个 PipelineResult dict 或它们的列表,统一成列表。"""
    if isinstance(payload, dict):
        return [payload]
    return list(payload)


def evaluate(
    pred_payload,
    gold_payload,
    synonym_map: Optional[dict[str, str]] = None,
    coarse_map: Optional[dict[str, str]] = None,
) -> dict:
    """总入口:对(可能多视频的)预测/标注做全套 §7 指标。多视频时按
    下标逐一配对,交互匹配计数跨视频累加后再算 PRF(micro 平均)。
    """
    preds = _as_list(pred_payload)
    golds = _as_list(gold_payload)
    if len(preds) != len(golds):
        raise ValueError(
            f"预测与标注的视频数不一致: {len(preds)} vs {len(golds)}"
        )
    if coarse_map is None:
        coarse_map = COARSE_MAP

    tiers = {
        "strict": (None, True),
        "synonym_merged": (synonym_map, True),
        "coarse": (coarse_map, False),
    }
    tier_counts = {name: [0, 0, 0] for name in tiers}  # matched, n_pred, n_gold
    dir_correct_total, dir_matched_total = 0, 0
    coverage_accum: list[dict] = []

    for pred, gold in zip(preds, golds):
        p_inter, g_inter = pred.get("interactions", []), gold.get("interactions", [])
        for name, (label_map, require_direction) in tiers.items():
            prf = match_interactions(p_inter, g_inter, label_map, require_direction)
            counts = tier_counts[name]
            counts[0] += prf.n_matched
            counts[1] += prf.n_pred
            counts[2] += prf.n_gold
        # 方向准确率的分子/分母也跨视频累加(micro 平均)。
        pred_und = Counter(_interaction_key(a, None, False) for a in p_inter)
        gold_und = Counter(_interaction_key(a, None, False) for a in g_inter)
        dir_matched_total += sum((pred_und & gold_und).values())
        pred_dir = Counter(_interaction_key(a, None, True) for a in p_inter)
        gold_dir = Counter(_interaction_key(a, None, True) for a in g_inter)
        dir_correct_total += sum((pred_dir & gold_dir).values())
        coverage_accum.append(
            instance_coverage(pred.get("instances", []), gold.get("instances", []))
        )

    n_gold_tracks = sum(c["n_gold"] for c in coverage_accum)
    covered = sum(c["track_coverage"] * c["n_gold"] for c in coverage_accum)
    iou_weighted = [
        c["mean_time_iou"] for c in coverage_accum if c["n_gold"] > 0
    ]
    return {
        "interaction_f1": {
            name: PRF.from_counts(*tier_counts[name]).to_json_dict() for name in tiers
        },
        "direction_accuracy": (
            dir_correct_total / dir_matched_total if dir_matched_total else None
        ),
        "instance": {
            "track_coverage": covered / n_gold_tracks if n_gold_tracks else 0.0,
            "mean_time_iou": (
                sum(iou_weighted) / len(iou_weighted) if iou_weighted else 0.0
            ),
        },
        "cost": aggregate_cost(preds),
    }


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("用法: python -m smot.eval <pred.json> <gold.json>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        pred = json.load(f)
    with open(argv[1], encoding="utf-8") as f:
        gold = json.load(f)
    print(json.dumps(evaluate(pred, gold), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
