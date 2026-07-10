"""Stage-1b 训练循环(M-B3):在 BenSMOT 上教师强制训练全部四个可学习
槽位 {unary KFA, pairwise KFA, fact selector, projector}。

对应 §6 的训练配方:
  - 冻结件:MLLM 主体 + 视觉塔(load_frozen_qwen 保证 requires_grad=False
    + eval());可训练的只有四个小模块(合计 ~2.3M 参数);
  - 单一 CE loss,三种任务(instance / interaction / video)共享同一个
    LM head,只靠 prompt 措辞区分——没有任何独立输出头;
  - 前向组装复用 teacher_forced_loss(与梯度门禁完全同一条路径)与
    compose_prompt_text(与推理适配器完全同一份 prompt 组装);
  - projector 输入是 [fact | unary | pairwise] 槽位布局(与
    smot.pipeline._compose_pooled 的推理侧布局同一契约,由单元测试
    锁住):本任务不产出的槽位显式补零;
  - fact selector 每步在线跑:soft 读出占 fact 槽位(梯度通路),
    hard top-k 渲染 transcript 进 prompt(离散,随训练演进)——
    transcript 文本不再是样本构造期固定的;
  - instance 证据帧来自 unary KFA 的 hard top-k,interaction 证据帧
    来自 pairwise KFA 的 hard top-k(帧候选与推理侧同源:优先用与
    run_bensmot_real 同配置的 EventCandidateFilter 触发帧,gold 对
    没有触发候选时退回双方共同观测帧);
  - Fact.embed 的 norm_value 分量用训练集统计量 z-score(smot.fact_norm),
    统计量随 checkpoint 一起保存,推理侧取回后做同一变换。

产出(--out-dir):
    stage1b.pt        checkpoint(四模块权重 + 构造配置 + fact 统计量)
    loss_log.jsonl    每步一行 {"step", "epoch", "task", "loss"}

用法:
    python -m smot.ml.training <BenSMOT根目录> --out-dir out/stage1b \
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
from smot.event_filter import EventCandidateFilter, adaptive_proximity_gate
from smot.fact_norm import make_fact_embed_normalizer
from smot.fact_selector import render_fact, scoped_facts
from smot.frame_features import geometric_frame_features
from smot.ml.checkpoint import save_stage1b_checkpoint
from smot.ml.fact_selector import (
    LearnableFactSelector,
    fact_priority_indices,
    fact_scoring_features,
    order_selected,
)
from smot.ml.frames import ImageDirFrameProvider, annotate_boxes
from smot.ml.pairwise_kfa import LearnablePairwiseKFA
from smot.ml.projector import MLPProjector
from smot.ml.qwen_adapter import (
    DEFAULT_MODEL_ID,
    compose_prompt_text,
    load_frozen_qwen,
    teacher_forced_loss,
)
from smot.ml.unary_kfa import LearnableUnaryKFA
from smot.motion_facts import MotionFactExtractor
from smot.pair_features import build_pair_features, pair_feature_vectors
from smot.prompts import (
    build_instance_prompt,
    build_interaction_prompt,
    build_video_prompt,
)


@dataclass
class _Example:
    """一条训练样本。prompt 文本不再在构造期固定——事实 transcript 由
    fact selector 每步在线选出,这里只携带"可供选择的原料"。"""

    task: str  # "instance" | "interaction" | "video"
    seq: BenSMOTSequence  # 该样本所属的完整序列(渲染帧时需要访问 imgs/ 目录)
    target_text: str  # 教师强制的目标句(gold caption/JSON/summary)
    prompt_ids: tuple[int, ...]  # prompt 里出现的 id:instance=(tid,);
    # interaction=(subject_id, object_id);video=involved_ids
    # ---- 事实选择原料(scope 内全部事实,selector 每步从中选) ----
    fact_feats: tuple[tuple[float, ...], ...]  # (N, FACT_SCORE_DIM) 打分特征
    fact_texts: tuple[str, ...]  # 每条渲染好的 transcript 片段(与 fact_feats 对齐)
    fact_priorities: tuple[int, ...]  # 每条的类型优先级下标(渲染排序用)
    # ---- instance 专用 ----
    track_id: Optional[int] = None  # 描述哪个目标(仅 instance 任务)
    features: tuple[tuple[float, ...], ...] = ()  # 逐帧几何特征,喂给 unary KFA 打分
    # ---- interaction 专用 ----
    pair_feats: tuple[tuple[float, ...], ...] = ()  # (T, PAIR_FEATURE_DIM) 向量化 pair 特征
    pair_ts: tuple[int, ...] = ()  # 与 pair_feats 对齐的帧号(hard 选帧 -> 帧号)
    box_track_ids: tuple[int, ...] = ()  # 需要画框的 track_id(instance 是自己,interaction 是双方)


def build_examples(
    sequences: list[BenSMOTSequence],
    fact_stats: dict,
) -> list[_Example]:
    """把转换后的序列展开成三类训练样本(gold 文本缺失的条目跳过)。"""
    extractor = MotionFactExtractor()
    normalize = make_fact_embed_normalizer(fact_stats)
    examples: list[_Example] = []

    for seq in sequences:
        trajectories = list(seq.trajectories)
        facts = normalize(extractor.extract(trajectories))
        t_max = max(traj.present[1] for traj in trajectories)
        # 与推理侧(run_bensmot_real.py)完全同配置的事件过滤器:训练时
        # pairwise KFA 看到的候选帧分布必须与推理一致,否则打分器学到的
        # 是另一种帧分布。
        candidates = EventCandidateFilter(
            proximity_gate=adaptive_proximity_gate(trajectories),
            proximity_trigger=True,
        ).find_candidates(trajectories)
        cand_frames = {c.edge: c.candidate_frames for c in candidates}

        def fact_bundle(scope: str):
            scoped = scoped_facts(facts, scope)
            return (
                fact_scoring_features(scoped),
                tuple(render_fact(f) for f in scoped),
                fact_priority_indices(scoped),
            )

        # ---- instance:每条有 gold caption 的轨迹一条样本 ----
        for traj in trajectories:
            caption = seq.instance_captions.get(traj.track_id, "").strip()
            if not caption or not traj.per_frame:
                continue
            feats, texts, priors = fact_bundle(f"instance:{traj.track_id}")
            examples.append(
                _Example(
                    task="instance",
                    seq=seq,
                    target_text=caption,
                    prompt_ids=(traj.track_id,),
                    fact_feats=feats,
                    fact_texts=texts,
                    fact_priorities=priors,
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
            # 帧候选与推理同源:事件过滤器触发帧优先(edge 已按 i<j 排序,
            # 与 (lo, hi) 直接可比);gold 对未被过滤器命中时退回双方共同
            # 观测帧(不丢训练样本——评测的召回上限不该由过滤器决定)。
            frames_source = cand_frames.get((lo, hi))
            if not frames_source:
                frames_source = sorted(
                    {fp.t for fp in traj_by_id[lo].per_frame}
                    & {fp.t for fp in traj_by_id[hi].per_frame}
                )
            pfs = build_pair_features(traj_by_id[lo], traj_by_id[hi], frames_source)
            feats, texts, priors = fact_bundle(f"pair:{lo},{hi}")
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
                    target_text=target,
                    prompt_ids=(first.subject_id, first.object_id),
                    fact_feats=feats,
                    fact_texts=texts,
                    fact_priorities=priors,
                    pair_feats=pair_feature_vectors(pfs, t_max=t_max),
                    pair_ts=tuple(pf.t for pf in pfs),
                    box_track_ids=(lo, hi),
                )
            )

        # ---- video:整段概括(与推理一致,不带图像) ----
        if seq.video_caption.strip():
            involved = tuple(sorted(traj.track_id for traj in trajectories))
            feats, texts, priors = fact_bundle("video")
            examples.append(
                _Example(
                    task="video",
                    seq=seq,
                    target_text=seq.video_caption.strip(),
                    prompt_ids=involved,
                    fact_feats=feats,
                    fact_texts=texts,
                    fact_priorities=priors,
                )
            )
    return examples


class _FrameRenderer:
    """按序列缓存帧提供者;imgs 目录缺失时静默退化为纯文本样本
    (合成 fixture 没有图像,真实 BenSMOT 一定有)。

    每个提供者的帧 LRU 压到 2:样本是跨序列 shuffle 的,帧缓存命中率
    本来就低,而全尺寸解码帧很大(1080p RGB ≈ 6MB/帧)——默认 32 帧
    × 上百个序列曾把训练进程 RAM 吹到数 GB。JPEG 重复解码只是毫秒级,
    宁可重解码。"""

    def __init__(self, max_side: int = 640):
        self._providers: dict[str, Optional[ImageDirFrameProvider]] = {}
        self._max_side = max_side

    def render(self, seq: BenSMOTSequence, frame_ts, boxes_by_t) -> list:
        provider = self._providers.get(seq.seq_dir, _MISSING)
        if provider is _MISSING:
            imgs_dir = Path(seq.seq_dir) / "imgs"
            provider = (
                ImageDirFrameProvider(imgs_dir, cache_size=2)
                if imgs_dir.is_dir()
                else None
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


@dataclass
class _Modules:
    """四个可学习槽位打包传递(训练循环内部用)。"""

    unary_kfa: LearnableUnaryKFA
    pairwise_kfa: LearnablePairwiseKFA
    fact_selector: LearnableFactSelector
    projector: MLPProjector

    def all(self) -> tuple:
        return (self.unary_kfa, self.pairwise_kfa, self.fact_selector, self.projector)


def example_loss(
    model,
    processor,
    modules: _Modules,
    example: _Example,
    renderer: _FrameRenderer,
    top_k_frames: int,
    top_k_pair_frames: int,
    fact_top_k: int,
    device: str,
):
    """一条样本的完整前向:fact selector(全任务)-> 任务对应 KFA ->
    [fact | unary | pairwise] 槽位组装 -> projector -> 渲染证据帧 ->
    共享的 teacher_forced_loss。返回可反传的 loss。

    可学习模块全部走 forward()(不是 select()):训练需要 soft 向量留在
    计算图里,select() 内部的 no_grad 会切断梯度,只适合推理。
    """
    traj_by_id = {traj.track_id: traj for traj in example.seq.trajectories}

    # ---- fact 槽位:soft 读出(可导)+ hard top-k 渲染 transcript ----
    if example.fact_feats:
        fact_tensor = torch.tensor(
            example.fact_feats, dtype=torch.float32, device=device
        )
        hard_facts, soft_facts = modules.fact_selector(fact_tensor, top_k=fact_top_k)
        chosen = order_selected(hard_facts.tolist(), example.fact_priorities)
        transcript = "; ".join(example.fact_texts[i] for i in chosen)
    else:
        # scope 内没有任何事实:fact 槽位补零(信息源缺席),transcript 为空。
        soft_facts = torch.zeros(
            modules.fact_selector.out_dim, dtype=torch.float32, device=device
        )
        transcript = ""

    zeros_unary = torch.zeros(
        modules.unary_kfa.out_dim, dtype=torch.float32, device=device
    )
    zeros_pair = torch.zeros(
        modules.pairwise_kfa.out_dim, dtype=torch.float32, device=device
    )

    if example.task == "instance":
        prompt_body = build_instance_prompt(example.prompt_ids[0], transcript)
        features = torch.tensor(example.features, dtype=torch.float32, device=device)
        hard_frames, soft_unary = modules.unary_kfa(features, top_k=top_k_frames)
        traj = traj_by_id[example.track_id]
        frame_ts = tuple(sorted(traj.per_frame[i].t for i in hard_frames.tolist()))
        box_track_ids = (example.track_id,)
        pooled = torch.cat([soft_facts, soft_unary, zeros_pair])
    elif example.task == "interaction":
        prompt_body = build_interaction_prompt(*example.prompt_ids, transcript)
        if example.pair_feats:
            pair_tensor = torch.tensor(
                example.pair_feats, dtype=torch.float32, device=device
            )
            hard_frames, soft_pair = modules.pairwise_kfa(
                pair_tensor, top_k=top_k_pair_frames
            )
            frame_ts = tuple(sorted(example.pair_ts[i] for i in hard_frames.tolist()))
        else:
            # 双方无共同观测帧(退化数据):纯文本样本 + pairwise 槽位补零,
            # 与 LearnablePairwiseKFA.select 对空 pair_features 的语义一致。
            soft_pair = zeros_pair
            frame_ts = ()
        box_track_ids = example.box_track_ids
        pooled = torch.cat([soft_facts, zeros_unary, soft_pair])
    else:  # video:没有任何 KFA 分量,两个槽位都补零
        prompt_body = build_video_prompt(example.prompt_ids, transcript)
        frame_ts = ()
        box_track_ids = ()
        pooled = torch.cat([soft_facts, zeros_unary, zeros_pair])

    soft_tokens = modules.projector(pooled).squeeze(0)  # (n_tokens, d_llm)

    # 逐证据帧收集 box_track_ids 里每个 track 在该帧的框(缺观测则跳过),
    # 与 Pipeline._frame_boxes 的形状约定完全一致,供画框/推理侧复用同一份 compose_prompt_text。
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

    # 与推理侧 QwenMLLMAdapter 完全同构的 prompt 组装(同一个 compose_prompt_text),
    # 这是"训练看到的输入分布 == 推理看到的输入分布"的关键保证。
    text = compose_prompt_text(
        example.task, prompt_body, frame_boxes, has_images=bool(images)
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
        # 用计算换显存:反向传播时重算前向激活值而不是全部缓存,
        # use_reentrant=False 是 PyTorch 推荐的新实现(兼容性更好)。
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    embedding = model.get_input_embeddings()
    embed_rms = float(embedding.weight.detach().float().pow(2).mean().sqrt())

    unary_kfa = LearnableUnaryKFA().to(device)
    pairwise_kfa = LearnablePairwiseKFA().to(device)
    fact_selector = LearnableFactSelector().to(device)
    projector = MLPProjector(
        # [fact | unary | pairwise] 槽位布局(与 Pipeline._compose_pooled
        # 的推理侧组装同一契约,tests/test_ml_modules.py 锁住两侧一致)。
        in_dim=fact_selector.out_dim + unary_kfa.out_dim + pairwise_kfa.out_dim,
        d_llm=embedding.embedding_dim,
        n_tokens=args.n_tokens,
        output_gain=embed_rms,  # 见 MLPProjector 模块 docstring:对齐冻结 LM 词嵌入尺度
    ).to(device)
    modules = _Modules(unary_kfa, pairwise_kfa, fact_selector, projector)
    for module in modules.all():
        module.train()
    trainable = list(
        itertools.chain.from_iterable(m.parameters() for m in modules.all())
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)  # 只优化四个槽位,冻结 MLLM 不在内
    n_trainable = sum(p.numel() for p in trainable)
    print(
        f"[模型] 可训练参数 {n_trainable/1e3:.1f}K,d_llm={embedding.embedding_dim},"
        f" embed_rms={embed_rms:.4f},projector in_dim={modules.projector.in_dim}",
        file=sys.stderr,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "loss_log.jsonl"
    renderer = _FrameRenderer(max_side=args.image_max_side)

    step = 0
    epoch_means: list[float] = []
    empty_cache_every = 20  # 周期清空 CUDA 缓存分配器,缓解变长样本的碎片化

    def save(epoch: int, stamp: bool = False) -> None:
        extra = {
            "fact_stats": fact_stats,
            "epochs": epoch,
            "steps": step,
            "epoch_mean_losses": epoch_means,
            "model_id": args.model_id,
        }
        # 周期性存档一律覆盖 stage1b.pt;epoch 末尾(stamp=True)额外留
        # 一份带 epoch 戳的副本——不同训练预算(1 epoch vs 2 epochs)的
        # checkpoint 都能拿去评测对比,不会被后续 epoch 覆盖。
        targets = [out_dir / "stage1b.pt"]
        if stamp:
            targets.append(out_dir / f"stage1b_epoch{epoch}.pt")
        for target in targets:
            save_stage1b_checkpoint(
                target,
                modules.unary_kfa,
                modules.pairwise_kfa,
                modules.fact_selector,
                modules.projector,
                extra=extra,
            )

    # buffering=1(行缓冲):每条 loss 记录立刻落盘,外部监控/断点续查
    # 才能看到真实进度(块缓冲曾让日志停在 flush 点,掩盖了停摆位置)。
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(examples)))
            rng.shuffle(order)
            losses: list[float] = []
            for index in order:
                example = examples[index]
                loss = example_loss(
                    model, processor, modules, example, renderer,
                    args.top_k_frames, args.top_k_pair_frames, args.fact_top_k,
                    device,
                )
                loss.backward()  # 单样本一步(无 batch 维度累积),梯度只流向四个槽位
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_norm)  # 防止个别样本梯度爆炸
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # set_to_none 比清零张量更省一次写显存

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
                # 变长多模态样本在 8GB 卡上会让缓存分配器逐步碎片化,
                # 顶到显存上限后 WDDM 把溢出页搬到系统内存,单步会从
                # 秒级劣化到分钟级(实测停摆)——周期性清空缓存换一点
                # 重分配开销,保持峰值远离天花板。
                if step % empty_cache_every == 0 and device.startswith("cuda"):
                    torch.cuda.empty_cache()
                if step % args.log_every == 0:
                    recent = losses[-args.log_every:]
                    print(
                        f"[epoch {epoch}] step {step}: "
                        f"loss(recent mean) = {sum(recent)/len(recent):.4f}",
                        file=sys.stderr, flush=True,
                    )
                # 周期性存档:长跑被环境超时/断电杀掉时,最多损失
                # save_every 步的进度,而不是整个未完成的 epoch。
                if args.save_every and step % args.save_every == 0:
                    save(epoch)
            mean_loss = sum(losses) / len(losses)
            epoch_means.append(mean_loss)
            print(f"[epoch {epoch}] mean loss = {mean_loss:.4f}",
                  file=sys.stderr, flush=True)
            save(epoch, stamp=True)

    print(
        f"[完成] checkpoint -> {out_dir / 'stage1b.pt'};"
        f" loss 曲线 -> {log_path}",
        file=sys.stderr,
    )
    return {"epoch_mean_losses": epoch_means, "steps": step}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m smot.ml.training", description="Stage-1b 训练循环"
    )
    parser.add_argument("root", help="BenSMOT 根目录(或任意包含序列的子目录)")
    parser.add_argument("--out-dir", default="out/stage1b")
    parser.add_argument("--limit", type=int, default=None, help="最多加载的序列数")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--n-tokens", type=int, default=4, help="soft token 数 m")
    parser.add_argument(
        "--top-k-frames", type=int, default=2,
        help="instance 任务每步送入 MLLM 的关键帧数(训练时从紧,控显存)",
    )
    parser.add_argument(
        "--top-k-pair-frames", type=int, default=2,
        help="interaction 任务每步送入 MLLM 的关键帧数(pairwise KFA hard 选帧)",
    )
    parser.add_argument(
        "--fact-top-k", type=int, default=6,
        help="fact selector 每步选进 transcript 的事实条数",
    )
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-every", type=int, default=100,
        help="每 N 步周期性保存 checkpoint(0 关闭,只在 epoch 末保存)",
    )
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
