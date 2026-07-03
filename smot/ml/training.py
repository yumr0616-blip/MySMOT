"""Stage-1a 训练循环(M-B2):在 BenSMOT 上教师强制训练 {unary KFA, projector}。

对应 §6 的训练配方:
  - 冻结件:MLLM 主体 + 视觉塔(load_frozen_qwen 保证 requires_grad=False
    + eval());可训练的只有 LearnableUnaryKFA 和 MLPProjector;
  - 单一 CE loss,三种任务(instance / interaction / video)共享同一个
    LM head,只靠 prompt 措辞区分——没有任何独立输出头;
  - 前向组装复用 teacher_forced_loss(与梯度门禁完全同一条路径)与
    compose_prompt_text(与推理适配器完全同一份 prompt 组装);
  - instance 任务的证据帧来自当前 KFA 的 hard top-k(随训练演进),
    soft 读出向量拼接事实池化进 projector——梯度经 soft 通路回传,
    hard 选帧"搭便车";interaction/video 任务没有 unary soft 向量,
    KFA 分量按零补齐(与推理侧 projector.project 的补零语义一致);
  - Fact.embed 的 norm_value 分量用训练集统计量 z-score(smot.fact_norm),
    统计量随 checkpoint 一起保存,推理侧取回后做同一变换。

产出(--out-dir):
    stage1a.pt        checkpoint(权重 + 构造配置 + fact 统计量)
    loss_log.jsonl    每步一行 {"step", "epoch", "task", "loss"}

用法:
    python -m smot.ml.training <BenSMOT根目录> --out-dir out/stage1a \
        [--limit N] [--epochs 2] [--lr 1e-4] [--max-frames 2] [...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from smot.datasets.bensmot import (
    BenSMOTSequence,
    compute_fact_statistics,
    load_split,
)
from smot.fact_norm import make_fact_embed_normalizer
from smot.fact_selector import DeterministicFactSelector, SelectionContext
from smot.frame_features import geometric_frame_features
from smot.kfa import _evenly_spaced
from smot.ml.checkpoint import save_stage1a_checkpoint
from smot.ml.frames import ImageDirFrameProvider, annotate_boxes
from smot.ml.projector import MLPProjector
from smot.ml.qwen_adapter import (
    DEFAULT_MODEL_ID,
    compose_prompt_text,
    load_frozen_qwen,
    teacher_forced_loss,
)
from smot.ml.unary_kfa import LearnableUnaryKFA
from smot.motion_facts import MotionFactExtractor
from smot.pipeline import _pool_embeds
from smot.prompts import (
    build_instance_prompt,
    build_interaction_prompt,
    build_video_prompt,
)


@dataclass
class _Example:
    """一条训练样本。instance 任务带逐帧特征(KFA 每步重新选帧);
    interaction/video 的证据帧在构造时固定。"""

    task: str  # "instance" | "interaction" | "video"
    seq: BenSMOTSequence
    prompt_body: str  # prompts.py 构造的任务 prompt(不含图例/指令后缀)
    pooled_facts: tuple[float, ...]  # 4 维事实池化(已归一化)
    target_text: str
    # instance 专用
    track_id: Optional[int] = None
    features: tuple[tuple[float, ...], ...] = ()
    # interaction 专用(固定证据帧;video 两者皆空)
    static_frame_ts: tuple[int, ...] = ()
    box_track_ids: tuple[int, ...] = ()


def build_examples(
    sequences: list[BenSMOTSequence],
    fact_stats: dict,
    fact_top_k: int = 6,
    pair_top_k: int = 2,
) -> list[_Example]:
    """把转换后的序列展开成三类训练样本(gold 文本缺失的条目跳过)。"""
    extractor = MotionFactExtractor()
    selector = DeterministicFactSelector()
    normalize = make_fact_embed_normalizer(fact_stats)
    examples: list[_Example] = []

    for seq in sequences:
        trajectories = list(seq.trajectories)
        facts = normalize(extractor.extract(trajectories))
        t_max = max(traj.present[1] for traj in trajectories)

        def pooled_for(scope: str) -> tuple[float, ...]:
            selection = selector.select(
                facts, SelectionContext(scope=scope, top_k=fact_top_k)
            )
            return (
                _pool_embeds(tuple(selection.selected_facts)) or (0.0,) * 4,
                selection.text,
            )

        # ---- instance:每条有 gold caption 的轨迹一条样本 ----
        for traj in trajectories:
            caption = seq.instance_captions.get(traj.track_id, "").strip()
            if not caption or not traj.per_frame:
                continue
            pooled, transcript = pooled_for(f"instance:{traj.track_id}")
            examples.append(
                _Example(
                    task="instance",
                    seq=seq,
                    prompt_body=build_instance_prompt(traj.track_id, transcript),
                    pooled_facts=pooled,
                    target_text=caption,
                    track_id=traj.track_id,
                    features=geometric_frame_features(traj, t_max=t_max),
                )
            )

        # ---- interaction:每个无序目标对一条样本,目标是该对全部有向
        # gold 断言的 JSON 数组——与推理侧的输出契约(assemble_interactions
        # 解析的数组)完全同构,一次请求学会"列出这一对的所有交互"。----
        traj_by_id = {traj.track_id: traj for traj in trajectories}
        by_pair: dict[tuple[int, int], list] = {}
        for inter in seq.interactions:
            key = tuple(sorted((inter.subject_id, inter.object_id)))
            by_pair.setdefault(key, []).append(inter)
        for (lo, hi), inters in by_pair.items():
            common_ts = sorted(
                {fp.t for fp in traj_by_id[lo].per_frame}
                & {fp.t for fp in traj_by_id[hi].per_frame}
            )
            pooled, transcript = pooled_for(f"pair:{lo},{hi}")
            # subject/object 的 prompt 顺序沿用第一条 gold 的方向(推理时
            # 是候选边顺序;方向语义靠数组项里的 id 表达,与顺序解耦)。
            first = inters[0]
            target = json.dumps(
                [
                    {
                        "subject_id": inter.subject_id,
                        "object_id": inter.object_id,
                        "predicate": inter.predicate,
                    }
                    for inter in inters
                ],
                ensure_ascii=False,
            )
            examples.append(
                _Example(
                    task="interaction",
                    seq=seq,
                    prompt_body=build_interaction_prompt(
                        first.subject_id, first.object_id, transcript
                    ),
                    pooled_facts=pooled,
                    target_text=target,
                    static_frame_ts=_evenly_spaced(common_ts, pair_top_k),
                    box_track_ids=(lo, hi),
                )
            )

        # ---- video:整段概括(与推理一致,不带图像) ----
        if seq.video_caption.strip():
            involved = tuple(sorted(traj.track_id for traj in trajectories))
            pooled, transcript = pooled_for("video")
            examples.append(
                _Example(
                    task="video",
                    seq=seq,
                    prompt_body=build_video_prompt(involved, transcript),
                    pooled_facts=pooled,
                    target_text=seq.video_caption.strip(),
                )
            )
    return examples


class _FrameRenderer:
    """按序列缓存帧提供者;imgs 目录缺失时静默退化为纯文本样本
    (合成 fixture 没有图像,真实 BenSMOT 一定有)。"""

    def __init__(self, max_side: int = 640):
        self._providers: dict[str, Optional[ImageDirFrameProvider]] = {}
        self._max_side = max_side

    def render(self, seq: BenSMOTSequence, frame_ts, boxes_by_t) -> list:
        provider = self._providers.get(seq.seq_dir, _MISSING)
        if provider is _MISSING:
            imgs_dir = Path(seq.seq_dir) / "imgs"
            provider = (
                ImageDirFrameProvider(imgs_dir) if imgs_dir.is_dir() else None
            )
            self._providers[seq.seq_dir] = provider
        if provider is None:
            return []
        images = []
        for t in frame_ts:
            image = provider.frame(t)
            boxes = boxes_by_t.get(t)
            if boxes:
                image = annotate_boxes(image, boxes)
            if max(image.size) > self._max_side:
                ratio = self._max_side / max(image.size)
                image = image.resize(
                    (round(image.width * ratio), round(image.height * ratio))
                )
            images.append(image)
        return images


_MISSING = object()


def example_loss(
    model,
    processor,
    kfa: LearnableUnaryKFA,
    projector: MLPProjector,
    example: _Example,
    renderer: _FrameRenderer,
    top_k_frames: int,
    device: str,
):
    """一条样本的完整前向:KFA(仅 instance)-> projector -> 渲染证据帧 ->
    共享的 teacher_forced_loss。返回可反传的 loss。"""
    traj_by_id = {traj.track_id: traj for traj in example.seq.trajectories}
    pooled4 = torch.tensor(example.pooled_facts, dtype=torch.float32, device=device)

    if example.task == "instance":
        features = torch.tensor(example.features, dtype=torch.float32, device=device)
        hard_indices, soft_vector = kfa(features, top_k=top_k_frames)
        traj = traj_by_id[example.track_id]
        frame_ts = tuple(sorted(traj.per_frame[i].t for i in hard_indices.tolist()))
        box_track_ids = (example.track_id,)
        pooled = torch.cat([pooled4, soft_vector])
    else:
        # 无 unary soft 向量的任务:KFA 分量按零补齐(与推理侧
        # projector.project 的缺失语义一致),KFA 不参与本条样本的梯度。
        frame_ts = example.static_frame_ts
        box_track_ids = example.box_track_ids
        pooled = torch.cat(
            [pooled4, torch.zeros(kfa.out_dim, dtype=torch.float32, device=device)]
        )

    soft_tokens = projector(pooled).squeeze(0)  # (n_tokens, d_llm)

    frame_boxes = tuple(
        (
            t,
            tuple(
                (tid, traj_by_id[tid].frame_at(t).box)
                for tid in box_track_ids
                if traj_by_id[tid].frame_at(t) is not None
            ),
        )
        for t in frame_ts
    )
    boxes_by_t = {t: dict(entries) for t, entries in frame_boxes}
    images = renderer.render(example.seq, frame_ts, boxes_by_t) if frame_ts else []

    text = compose_prompt_text(
        example.task, example.prompt_body, frame_boxes, has_images=bool(images)
    )
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    return teacher_forced_loss(model, processor, messages, soft_tokens, example.target_text)


def train(args) -> dict:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device

    sequences, errors = load_split(
        args.root, limit=args.limit, on_error="skip" if args.skip_errors else "raise"
    )
    for seq_dir, message in errors:
        print(f"[跳过] {seq_dir}: {message}", file=sys.stderr)
    if not sequences:
        raise SystemExit(f"在 {args.root} 下没有找到可用序列")

    # fact 统计量从训练序列上算(推理侧经 checkpoint 取回同一份)。
    fact_stats = compute_fact_statistics(sequences)
    examples = build_examples(sequences, fact_stats)
    if not examples:
        raise SystemExit("没有可训练样本(gold 文本全部缺失?)")
    task_counts = {
        task: sum(1 for e in examples if e.task == task)
        for task in ("instance", "interaction", "video")
    }
    print(f"[数据] 序列 {len(sequences)},样本 {task_counts}", file=sys.stderr)

    model, processor = load_frozen_qwen(
        args.model_id, device=device, quantize_4bit=args.quantize_4bit
    )
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    embedding = model.get_input_embeddings()
    embed_rms = float(embedding.weight.detach().float().pow(2).mean().sqrt())

    kfa = LearnableUnaryKFA().to(device)
    projector = MLPProjector(
        in_dim=len(examples[0].pooled_facts) + kfa.out_dim,
        d_llm=embedding.embedding_dim,
        n_tokens=args.n_tokens,
        output_gain=embed_rms,
    ).to(device)
    kfa.train()
    projector.train()
    trainable = list(itertools.chain(kfa.parameters(), projector.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[模型] 可训练参数 {n_trainable/1e3:.1f}K,d_llm={embedding.embedding_dim},"
          f" embed_rms={embed_rms:.4f}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "loss_log.jsonl"
    renderer = _FrameRenderer(max_side=args.image_max_side)

    step = 0
    epoch_means: list[float] = []
    with open(log_path, "w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(examples)))
            rng.shuffle(order)
            losses: list[float] = []
            for index in order:
                example = examples[index]
                loss = example_loss(
                    model, processor, kfa, projector, example,
                    renderer, args.top_k_frames, device,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                loss_value = float(loss.detach())
                losses.append(loss_value)
                log_file.write(
                    json.dumps(
                        {"step": step, "epoch": epoch, "task": example.task,
                         "loss": round(loss_value, 6)}
                    )
                    + "\n"
                )
                if step % args.log_every == 0:
                    recent = losses[-args.log_every:]
                    print(
                        f"[epoch {epoch}] step {step}: "
                        f"loss(recent mean) = {sum(recent)/len(recent):.4f}",
                        file=sys.stderr,
                    )
            mean_loss = sum(losses) / len(losses)
            epoch_means.append(mean_loss)
            print(f"[epoch {epoch}] mean loss = {mean_loss:.4f}", file=sys.stderr)
            save_stage1a_checkpoint(
                out_dir / "stage1a.pt",
                kfa,
                projector,
                extra={
                    "fact_stats": fact_stats,
                    "epochs": epoch,
                    "steps": step,
                    "epoch_mean_losses": epoch_means,
                    "model_id": args.model_id,
                },
            )

    print(
        f"[完成] checkpoint -> {out_dir / 'stage1a.pt'};"
        f" loss 曲线 -> {log_path}",
        file=sys.stderr,
    )
    return {"epoch_mean_losses": epoch_means, "steps": step}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m smot.ml.training", description="Stage-1a 训练循环"
    )
    parser.add_argument("root", help="BenSMOT 根目录(或任意包含序列的子目录)")
    parser.add_argument("--out-dir", default="out/stage1a")
    parser.add_argument("--limit", type=int, default=None, help="最多加载的序列数")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--n-tokens", type=int, default=4, help="soft token 数 m")
    parser.add_argument(
        "--top-k-frames", type=int, default=2,
        help="instance 任务每步送入 MLLM 的关键帧数(训练时从紧,控显存)",
    )
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = train(args)
    means = report["epoch_mean_losses"]
    if len(means) >= 2 and means[-1] < means[0]:
        print(f"loss 下降: {means[0]:.4f} -> {means[-1]:.4f}")
    else:
        print(f"loss 未见下降(epoch 均值: {[round(m, 4) for m in means]}),"
              f"检查学习率/数据量")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
