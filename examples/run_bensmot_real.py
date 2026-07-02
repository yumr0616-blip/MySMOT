"""BenSMOT + 真实冻结 Qwen3.5 的端到端推理脚本(M-A2/M-B2 共用)。

与 run_bensmot_stage0.py 的唯一区别:MLLM 从 Mock 换成真实的
QwenMLLMAdapter(关键帧图像 + 画框 grounding + 结构化 JSON 输出),
可选地再注入训练好的 Stage-1a 组件(--checkpoint)。产出与 stage0
脚本同构(pred/gold/metrics 三个 JSON),两组结果可以直接对比——
这就是"真实 MLLM 相对 Mock 地板的提升"与"学习后相对确定性选择的
提升"两条对照的载体。

用法:
    python examples/run_bensmot_real.py <BenSMOT根目录或子目录> \
        [--limit N] [--out-dir out/bensmot_real] [--model-id Qwen/Qwen3.5-2B] \
        [--quantize-4bit] [--checkpoint stage1a.pt] [--skip-errors]

需要 ml 依赖(venv + torch cu128 + transformers,见 README)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smot.datasets.bensmot import (
    build_gold_payloads,
    load_split,
    sequence_to_video_handle,
)
from smot.eval import evaluate
from smot.event_filter import EventCandidateFilter, adaptive_proximity_gate
from smot.frame_features import geometric_frame_features
from smot.pipeline import Pipeline
from smot.tracker import StubTracker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("root", help="BenSMOT 根目录(或任意包含序列的子目录)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default=os.path.join("out", "bensmot_real"))
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Stage-1a checkpoint(注入训练好的 unary KFA + projector)",
    )
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args()

    sequences, errors = load_split(
        args.root, limit=args.limit, on_error="skip" if args.skip_errors else "raise"
    )
    for seq_dir, message in errors:
        print(f"[跳过] {seq_dir}: {message}", file=sys.stderr)
    if not sequences:
        print(f"在 {args.root} 下没有找到可用序列", file=sys.stderr)
        return 1

    # 重依赖 import 放在参数解析之后:--help 不需要 torch。
    from smot.ml.qwen_adapter import QwenMLLMAdapter

    adapter = QwenMLLMAdapter(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        quantize_4bit=args.quantize_4bit,
    )

    kfa = projector = None
    if args.checkpoint:
        from smot.ml.checkpoint import load_stage1a_checkpoint

        kfa, projector, extra = load_stage1a_checkpoint(args.checkpoint, device="cuda")
        print(f"[checkpoint] 已加载 {args.checkpoint}: {extra}", file=sys.stderr)

    preds: list[dict] = []
    for seq in sequences:
        t_max = max(traj.present[1] for traj in seq.trajectories)
        trajectories = list(seq.trajectories)
        pipeline_kwargs: dict = {
            "tracker": StubTracker(trajectories),
            "mllm_adapter": adapter,
            # 真实数据:邻近门限按目标尺度自适应,并开启邻近跳变触发器
            # (覆盖交谈类无离散事件的交互)。
            "event_filter": EventCandidateFilter(
                proximity_gate=adaptive_proximity_gate(trajectories),
                proximity_trigger=True,
            ),
        }
        if kfa is not None:
            pipeline_kwargs.update(
                unary_kfa=kfa,
                projector=projector,
                # 逐帧特征的时间归一化用本序列的全局最大帧号。
                frame_feature_fn=lambda traj, _tm=t_max: geometric_frame_features(
                    traj, t_max=_tm
                ),
            )
        pipeline = Pipeline(**pipeline_kwargs)
        result = pipeline.run(sequence_to_video_handle(seq))
        payload = result.to_json_dict()
        payload["sequence"] = seq.name
        preds.append(payload)
        print(
            f"[完成] {seq.name}: cost={result.cost.to_json_dict()}", file=sys.stderr
        )
        for assertion in result.instances:
            print(f"    instance[{assertion.track_id}]: {assertion.caption}", file=sys.stderr)
        for assertion in result.interactions:
            print(
                f"    interaction[{assertion.subject_id}->{assertion.object_id}]: "
                f"{assertion.predicate} (canonical={assertion.canonical_label}, "
                f"conf={assertion.confidence})",
                file=sys.stderr,
            )
        print(f"    video: {result.video.summary}", file=sys.stderr)

    golds = build_gold_payloads(sequences)
    metrics = evaluate(preds, golds)

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
