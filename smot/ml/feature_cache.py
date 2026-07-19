"""P2 · Stage-2:离线视觉塔特征缓存,供两个 KFA 的打分输入从纯几何
升级为几何+外观(§4 设计文档预留的 vis_i/vis_j 接口)。

方案 A(与推理同源,无新依赖):复用冻结 Qwen3.5 自带的视觉塔
model.model.visual,取 merger 之前的 hidden_size=1152 层(比
out_hidden_size=4096 的 merger 输出省一半以上存储/参数),对每个
track 在某一帧的 bbox crop 做前向,patch token 做 mean pool 得到
定长向量。

缓存粒度是 (track_id, 帧号) -> 向量,unary 和 pairwise 共享同一份底层
缓存(同一个 crop 只算一次):
  - unary 侧逐帧特征本该覆盖轨迹的每一帧,但视觉塔前向不便宜——按
    UNARY_STRIDE 稀疏采样,读时对未采样帧做最近邻回填(见
    NearestFillCache)。
  - pairwise 侧只需要 EventCandidateFilter 选出的候选帧(本来就稀疏),
    直接对这些帧上的双方 track 都建缓存条目,读时精确命中,理论上
    不需要回填(仍留最近邻兜底,防止候选帧集合在训练/推理两侧因参数
    差异而不完全一致)。

存盘格式:每条序列一个 npz,键 "{track_id}_{t}" -> (1152,) float32
数组,见 save_cache/load_cache。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from smot.datasets.bensmot import load_split
from smot.event_filter import EventCandidateFilter, adaptive_proximity_gate
from smot.frame_features import FRAME_FEATURE_DIM, geometric_frame_features
from smot.ml.frames import ImageDirFrameProvider
from smot.pair_features import PAIR_FEATURE_DIM, pair_feature_vectors
from smot.types import Box, PairFeature, Trajectory

VISUAL_FEATURE_DIM = 1152  # = model.config.vision_config.hidden_size(merger 之前)
DEFAULT_UNARY_STRIDE = 6  # 每隔几个观测帧抽一次 unary 视觉特征(候选帧集合不受此影响)


def _clamp_box(box: Box, width: int, height: int) -> Optional[Box]:
    """把 bbox 夹到图像范围内;夹后退化(宽或高 <= 0)返回 None。"""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width))
    y2 = max(0.0, min(y2, height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def _crop(image: Image.Image, box: Box) -> Optional[Image.Image]:
    clamped = _clamp_box(box, image.width, image.height)
    if clamped is None:
        return None
    return image.crop(tuple(round(v) for v in clamped))


@torch.inference_mode()
def batched_visual_features(
    model, processor, images: list[Image.Image], device: str, batch_size: int = 32
) -> np.ndarray:
    """对一批(可不同尺寸的)crop 做视觉塔前向 + mean pool,返回 (N, 1152)。

    Qwen 的图像处理器把变分辨率图片各自 patchify 后拼成一条 pixel_values,
    image_grid_thw 记录每张图的 patch 网格(t,h,w);视觉塔一次前向能处理
    整批,输出按 grid_thw 算出的 patch 数切回单图再做 mean pool——这与
    真正推理时"整段视频一次性喂多帧"的路径是同一个视觉塔前向,只是这里
    只取 merger 之前的 last_hidden_state。
    """
    if not images:
        return np.zeros((0, VISUAL_FEATURE_DIM), dtype=np.float32)
    out_chunks: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor.image_processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        grid_thw = inputs["image_grid_thw"].to(device)
        visual_out = model.model.visual(pixel_values, grid_thw)
        patch_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        for chunk in torch.split(visual_out.last_hidden_state, patch_counts, dim=0):
            out_chunks.append(chunk.float().mean(dim=0).cpu().numpy())
    return np.stack(out_chunks, axis=0).astype(np.float32)


def _unary_sample_frames(traj: Trajectory, stride: int) -> list[int]:
    """稀疏采样一条轨迹要建视觉缓存的帧号:每隔 stride 个观测帧一个,
    首尾总是包含(回填时的边界锚点,避免首尾大段外插)。"""
    ts = [fp.t for fp in traj.per_frame]
    if not ts:
        return []
    sampled = set(ts[::stride])
    sampled.add(ts[0])
    sampled.add(ts[-1])
    return sorted(sampled)


def build_sequence_cache(
    seq, model, processor, device: str, unary_stride: int = DEFAULT_UNARY_STRIDE
) -> dict[str, np.ndarray]:
    """对一条 BenSMOT 序列建完整的 (track_id, t) -> 视觉特征缓存。

    需要建哪些 (track_id, t):
      - unary:每条轨迹按 stride 稀疏采样;
      - pairwise:EventCandidateFilter 选出的候选边在各自候选帧上,
        双方 track 都要有条目(与推理侧 pairwise KFA 的候选帧同源,
        用同一个 filter 配置——不然训练看到的候选帧分布和推理不一致)。
    两者取并集去重,避免同一个 (track_id, t) 的 crop 被算两次。
    """
    imgs_dir = Path(seq.seq_dir) / "imgs"
    if not imgs_dir.is_dir():
        return {}  # 合成 fixture 没有图像;真实 BenSMOT 一定有
    provider = ImageDirFrameProvider(imgs_dir, cache_size=8)
    trajectories = list(seq.trajectories)
    traj_by_id = {t.track_id: t for t in trajectories}

    needed: set[tuple[int, int]] = set()
    for traj in trajectories:
        for t in _unary_sample_frames(traj, unary_stride):
            needed.add((traj.track_id, t))

    if len(trajectories) >= 2:
        candidates = EventCandidateFilter(
            proximity_gate=adaptive_proximity_gate(trajectories),
            proximity_trigger=True,
        ).find_candidates(trajectories)
        for cand in candidates:
            i, j = cand.edge
            for t in cand.candidate_frames:
                needed.add((i, t))
                needed.add((j, t))

    keys: list[str] = []
    crops: list[Image.Image] = []
    frame_cache: dict[int, Image.Image] = {}
    for track_id, t in sorted(needed):
        traj = traj_by_id.get(track_id)
        fp = traj.frame_at(t) if traj is not None else None
        if fp is None:
            continue  # 该 track 在这帧没有观测(候选帧来自另一方触发),跳过
        image = frame_cache.get(t)
        if image is None:
            try:
                image = provider.frame(t)
            except KeyError:
                continue  # 标注帧号超出实际图像范围(数据噪声),跳过而非报错
            frame_cache[t] = image
        crop = _crop(image, fp.box)
        if crop is None:
            continue  # 退化 box(见 gt.txt 负宽高历史问题),跳过
        keys.append(f"{track_id}_{t}")
        crops.append(crop)

    if not crops:
        return {}
    vecs = batched_visual_features(model, processor, crops, device)
    return {k: vecs[i] for i, k in enumerate(keys)}


def cache_path(out_dir: str, split: str, seq_name: str) -> Path:
    # seq.name 形如 "活动类别/序列名",按原样建子目录,序列名做文件名。
    activity, _, stem = seq_name.rpartition("/")
    return Path(out_dir) / split / activity / f"{stem}.npz"


def save_cache(path: Path, features: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


class NearestFillCache:
    """把 save_cache 存的扁平 "{track_id}_{t}" -> 向量 字典,按 track_id
    重新索引成有序 (t, 向量) 列表,支持按帧号最近邻查找。

    unary 侧稀疏采样导致大部分帧没有精确条目,靠这个做回填;pairwise
    侧候选帧理论上精确命中,但训练/推理两侧的候选帧计算若因参数漂移
    有一两帧出入,也靠这个兜底而不是直接报错——视觉分量本就是几何分量
    的增强,缺一帧退化成"用邻近帧的外观"是合理近似,不是错误。
    """

    def __init__(self, raw: dict[str, np.ndarray]):
        by_track: dict[int, list[tuple[int, np.ndarray]]] = {}
        for key, vec in raw.items():
            track_str, _, t_str = key.rpartition("_")
            by_track.setdefault(int(track_str), []).append((int(t_str), vec))
        self._by_track = {
            tid: sorted(entries, key=lambda e: e[0]) for tid, entries in by_track.items()
        }

    def lookup(self, track_id: int, t: int) -> Optional[np.ndarray]:
        entries = self._by_track.get(track_id)
        if not entries:
            return None
        # entries 已按 t 排序,线性扫找最近——每条轨迹的条目数很小
        # (稀疏采样 + 候选帧),没必要上二分。
        best = min(entries, key=lambda e: abs(e[0] - t))
        return best[1]

    def __bool__(self) -> bool:
        return bool(self._by_track)


AUGMENTED_FRAME_FEATURE_DIM = FRAME_FEATURE_DIM + VISUAL_FEATURE_DIM
AUGMENTED_PAIR_FEATURE_DIM = PAIR_FEATURE_DIM + 2 * VISUAL_FEATURE_DIM
_ZERO_VISUAL = (0.0,) * VISUAL_FEATURE_DIM  # 缓存缺失时的兜底(不是"零等于中性",只是没有更好选择)


def augmented_frame_features(
    traj: Trajectory,
    cache: NearestFillCache,
    t_max: Optional[int] = None,
    scale: float = 1000.0,
) -> tuple[tuple[float, ...], ...]:
    """geometric_frame_features 的增强版:每帧再拼上该 track 在这一帧
    (最近邻回填后)的视觉塔特征。维度 = FRAME_FEATURE_DIM + 1152。"""
    geo = geometric_frame_features(traj, t_max=t_max, scale=scale)
    out = []
    for fp, g in zip(traj.per_frame, geo):
        vis = cache.lookup(traj.track_id, fp.t)
        out.append(g + (tuple(float(v) for v in vis) if vis is not None else _ZERO_VISUAL))
    return tuple(out)


def augmented_pair_feature_vectors(
    pair_features: Sequence[PairFeature],
    cache: NearestFillCache,
    t_max: Optional[int] = None,
    scale: float = 1000.0,
) -> tuple[tuple[float, ...], ...]:
    """pair_feature_vectors 的增强版:每帧再拼上双方 track 在这一帧
    (最近邻回填后)的视觉塔特征(vis_i, vis_j)。维度 = PAIR_FEATURE_DIM
    + 2*1152。edge 里的 track_id 顺序即 vis_i/vis_j 的顺序,与
    PairFeature.edge=(i,j) 的既有约定一致。"""
    geo = pair_feature_vectors(pair_features, t_max=t_max, scale=scale)
    out = []
    for pf, g in zip(pair_features, geo):
        i, j = pf.edge
        vis_i = cache.lookup(i, pf.t)
        vis_j = cache.lookup(j, pf.t)
        vi = tuple(float(v) for v in vis_i) if vis_i is not None else _ZERO_VISUAL
        vj = tuple(float(v) for v in vis_j) if vis_j is not None else _ZERO_VISUAL
        out.append(g + vi + vj)
    return tuple(out)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m smot.ml.feature_cache",
        description="逐序列离线预提视觉塔特征(P2-1)",
    )
    parser.add_argument("root", help="BenSMOT 根目录(含 train/test 子目录)")
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("--out-dir", default="out/feat_cache")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--unary-stride", type=int, default=DEFAULT_UNARY_STRIDE)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="跳过已存在缓存文件的序列"
    )
    args = parser.parse_args(argv)

    split_root = str(Path(args.root) / args.split)
    sequences, errors = load_split(
        split_root, limit=args.limit, on_error="skip" if args.skip_errors else "raise"
    )
    for seq_dir, message in errors:
        print(f"[跳过] {seq_dir}: {message}", file=sys.stderr)
    if not sequences:
        print(f"在 {split_root} 下没有找到可用序列", file=sys.stderr)
        return 1

    from smot.ml.qwen_adapter import DEFAULT_MODEL_ID, load_frozen_qwen

    model, processor = load_frozen_qwen(
        args.model_id or DEFAULT_MODEL_ID, device=args.device
    )

    n_done = n_skipped = n_empty = 0
    for seq in sequences:
        out_path = cache_path(args.out_dir, args.split, seq.name)
        if args.resume and out_path.exists():
            n_skipped += 1
            continue
        features = build_sequence_cache(seq, model, processor, args.device, args.unary_stride)
        if not features:
            n_empty += 1
            print(f"[空] {seq.name}: 无可用图像/crop", file=sys.stderr)
            continue
        save_cache(out_path, features)
        n_done += 1
        print(
            f"[完成] {seq.name}: {len(features)} 条缓存 -> {out_path}",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"[汇总] 完成 {n_done},跳过(已存在) {n_skipped},空 {n_empty},"
        f"共 {len(sequences)} 段",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
