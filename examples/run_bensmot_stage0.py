"""BenSMOT 真实数据 + Stage-0 全默认 Pipeline 的基线脚本。

对下载好的 BenSMOT 数据(或其任意子目录)逐序列跑一遍 Stage-0 管线:
GT 轨迹经 StubTracker 注入(= 冻结 tracker 输出、控制跟踪变量的实验
设置),MLLM 仍是 Mock——因此这份基线给出的是"确定性管线 + 零语义
理解"的成本与指标地板,M-A2 接入真实 MLLM 后的第一组对照就是它。

产出(写入 --out-dir):
    pred.json     每序列一个 PipelineResult payload 的列表
    gold.json     同序、同形的 gold payload 列表
    metrics.json  smot.eval 的全套 §7 指标

用法:
    python examples/run_bensmot_stage0.py <BenSMOT根目录或子目录> \
        [--limit N] [--out-dir out/bensmot_stage0] [--skip-errors]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 让脚本无论从哪个工作目录运行,都能找到仓库根目录下的 smot 包。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smot.datasets.bensmot import (
    BENSMOT_SYNONYM_MAP,
    build_gold_payloads,
    load_split,
    sequence_to_video_handle,
)
from smot.eval import evaluate
from smot.event_filter import EventCandidateFilter, adaptive_proximity_gate
from smot.pipeline import Pipeline
from smot.tracker import StubTracker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("root", help="BenSMOT 根目录(或任意包含序列的子目录)")
    parser.add_argument("--limit", type=int, default=None, help="最多处理的序列数")
    parser.add_argument("--out-dir", default=os.path.join("out", "bensmot_stage0"))
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="解析失败的序列跳过并报告,而不是整体中止",
    )
    args = parser.parse_args()

    sequences, errors = load_split(
        args.root, limit=args.limit, on_error="skip" if args.skip_errors else "raise"
    )
    for seq_dir, message in errors:
        print(f"[跳过] {seq_dir}: {message}", file=sys.stderr)
    if not sequences:
        print(f"在 {args.root} 下没有找到可用序列", file=sys.stderr)
        return 1

    preds: list[dict] = []
    for seq in sequences:
        trajectories = list(seq.trajectories)
        # GT 轨迹当作冻结 tracker 的输出。事件过滤与真实脚本同配置
        # (自适应邻近门限 + 邻近跳变触发),保证 Mock 基线与真实 MLLM
        # 的成本/指标对比在同一候选集上进行;其余组件 Stage-0 默认值。
        pipeline = Pipeline(
            tracker=StubTracker(trajectories),
            event_filter=EventCandidateFilter(
                proximity_gate=adaptive_proximity_gate(trajectories),
                proximity_trigger=True,
            ),
        )
        result = pipeline.run(sequence_to_video_handle(seq))
        payload = result.to_json_dict()
        payload["sequence"] = seq.name  # 与 gold 对齐的溯源键
        preds.append(payload)
        print(
            f"[完成] {seq.name}: {len(result.instances)} instance, "
            f"{len(result.interactions)} interaction, cost={result.cost.to_json_dict()}",
            file=sys.stderr,
        )

    golds = build_gold_payloads(sequences)
    # preds 与 golds 按序列列表同一顺序构造,evaluate() 按下标逐一配对。
    metrics = evaluate(preds, golds, synonym_map=BENSMOT_SYNONYM_MAP)

    os.makedirs(args.out_dir, exist_ok=True)
    for filename, payload in (
        ("pred.json", preds),
        ("gold.json", golds),
        ("metrics.json", metrics),
    ):
        with open(os.path.join(args.out_dir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n结果已写入 {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
