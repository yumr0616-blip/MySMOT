"""BenSMOT + 真实冻结 Qwen3.5 的端到端推理脚本(M-A2/M-B2/M-B3 共用)。

与 run_bensmot_stage0.py 的唯一区别:MLLM 从 Mock 换成真实的
QwenMLLMAdapter(关键帧图像 + 画框 grounding + 结构化 JSON 输出),
可选地再注入训练好的可学习组件(--checkpoint,1a/1b 格式自动判别:
1a 注入 unary KFA + projector,1b 额外注入 pairwise KFA + fact
selector + pair 特征向量化接缝)。产出与 stage0 脚本同构(pred/gold/
metrics 三个 JSON),各组结果可以直接对比——这就是"真实 MLLM 相对
Mock 地板的提升"与"学习后相对确定性选择的提升"两条对照的载体。

用法:
    python examples/run_bensmot_real.py <BenSMOT根目录或子目录> \
        [--limit N] [--out-dir out/bensmot_real] [--model-id Qwen/Qwen3.5-2B] \
        [--quantize-4bit] [--checkpoint stage1b.pt] [--skip-errors]

需要 ml 依赖(venv + torch cu128 + transformers,见 README)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smot.datasets.bensmot import (
    BENSMOT_SYNONYM_MAP,
    build_gold_payloads,
    load_split,
    sequence_to_video_handle,
)
from smot.eval import evaluate
from smot.event_filter import EventCandidateFilter, adaptive_proximity_gate
from smot.frame_features import geometric_frame_features
from smot.pair_features import pair_feature_vectors
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
        help="训练好的 checkpoint(1a/1b 格式自动判别,注入对应可学习组件)",
    )
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 <out-dir>/pred.jsonl 续跑:跳过已完成序列(须与上次同一"
             "配置/checkpoint,脚本不校验)",
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

    # 重依赖 import 放在参数解析之后:--help 不需要 torch。
    from smot.ml.qwen_adapter import QwenMLLMAdapter

    adapter = QwenMLLMAdapter(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        quantize_4bit=args.quantize_4bit,
    )

    loaded = fact_transform = None
    if args.checkpoint:
        from smot.fact_norm import make_fact_embed_normalizer
        from smot.ml.checkpoint import load_checkpoint

        loaded = load_checkpoint(args.checkpoint, device="cuda")
        # 训练时的 fact 统计量随 checkpoint 保存,推理侧必须做同一份
        # embed 归一化,否则可学习组件收到的输入分布与训练时不一致。
        if loaded.extra.get("fact_stats"):
            fact_transform = make_fact_embed_normalizer(loaded.extra["fact_stats"])
        print(
            f"[checkpoint] 已加载 {args.checkpoint}(stage={loaded.stage}): "
            f"epochs={loaded.extra.get('epochs')},"
            f" steps={loaded.extra.get('steps')}", file=sys.stderr,
        )

    # ---- 增量落盘:逐序列一行 JSON(行缓冲),被杀最多丢当前这一段;
    # --resume 时读回已完成的序列直接跳过(P-eng,防随机进程误杀)。----
    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "pred.jsonl")
    done: dict[str, dict] = {}
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 被杀时可能留下半行,静默丢弃(该段会重跑)
                done[payload["sequence"]] = payload
        if done:
            print(f"[续跑] 已完成 {len(done)} 段,跳过", file=sys.stderr)

    jsonl_file = open(  # noqa: SIM115 - 生命周期跨整个循环,结束处显式关闭
        jsonl_path, "a" if args.resume else "w", encoding="utf-8", buffering=1
    )
    for seq in sequences:
        if seq.name in done:
            continue
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
        if loaded is not None:  # 只有传了 --checkpoint 才用可学习组件,否则维持 Stage-0 NoOp 默认值
            pipeline_kwargs.update(
                unary_kfa=loaded.unary_kfa,
                projector=loaded.projector,
                # 逐帧特征的时间归一化用本序列的全局最大帧号。
                # 用默认参数 _tm=t_max 把当前循环变量的值"冻结"进闭包——
                # 直接引用外层 t_max 会有经典的 Python 闭包晚绑定问题
                # (循环结束后所有 lambda 都会看到同一个、最后一次迭代的 t_max)。
                frame_feature_fn=lambda traj, _tm=t_max: geometric_frame_features(
                    traj, t_max=_tm
                ),
                fact_transform=fact_transform,
            )
            if loaded.pairwise_kfa is not None:
                # Stage-1b 组件:可学习 pairwise KFA + fact selector,以及
                # pairwise 打分需要的向量化特征接缝(时间归一化同样按本
                # 序列的 t_max 冻结进闭包,与训练侧 build_examples 一致)。
                pipeline_kwargs.update(
                    pairwise_kfa=loaded.pairwise_kfa,
                    fact_selector=loaded.fact_selector,
                    pair_feature_fn=lambda pfs, _tm=t_max: pair_feature_vectors(
                        pfs, t_max=_tm
                    ),
                )
        pipeline = Pipeline(**pipeline_kwargs)
        result = pipeline.run(sequence_to_video_handle(seq))
        payload = result.to_json_dict()
        payload["sequence"] = seq.name
        done[seq.name] = payload
        jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
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

    jsonl_file.close()
    # 按 sequences 的规范顺序聚合(而不是 jsonl 的写入顺序):续跑与
    # 一次跑完产出的 pred.json 逐字节一致。
    preds = [done[seq.name] for seq in sequences]
    golds = build_gold_payloads(sequences)
    metrics = evaluate(preds, golds, synonym_map=BENSMOT_SYNONYM_MAP)

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
