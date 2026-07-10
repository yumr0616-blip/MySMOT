"""帧提供者:把 MLLMRequest 里的 video_path + frame_refs 变成真正的图像,
并提供画框 grounding 工具(把 track_id 和画面里的具体目标对应起来)。

BenSMOT 的视频以 imgs/ 图像目录发布(帧号从 1 开始,与 gt.txt 一致),
所以 ImageDirFrameProvider 是主力;VideoFileFrameProvider 用于以后对
任意视频文件做推理。
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image, ImageDraw

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# 按 track_id 取模分配的画框颜色盘(区分度优先,顺序即 id%len 的映射)。
_PALETTE = ("red", "blue", "lime", "orange", "magenta", "cyan", "yellow")


def color_for_track(track_id: int) -> str:
    """给某个 track 一个确定性的画框颜色。prompt 里的颜色图例必须与
    画到帧上的颜色一致,两处都从这里取。"""
    return _PALETTE[track_id % len(_PALETTE)]


@runtime_checkable
class FrameProvider(Protocol):
    def frame(self, t: int) -> Image.Image:
        """按帧号 t(1-based)取一帧图像。"""
        ...


class ImageDirFrameProvider:
    """图像目录形式的视频(BenSMOT 的 imgs/ 布局)。

    文件名 stem 全部可解析为整数时,按数字直接映射帧号(000001.jpg ->
    帧 1);否则退回"排序后第 i 个文件 = 第 i 帧(1-based)"。带一个小
    LRU 缓存——同一帧常被相邻的 instance/interaction 请求重复引用。
    注意:缓存返回的是共享对象,调用方不得原地修改(annotate_boxes
    总是在副本上画)。
    """

    def __init__(self, dir_path: str | Path, cache_size: int = 32):
        files = sorted(
            p for p in Path(dir_path).iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"{dir_path} 下没有图像文件")
        try:
            # 文件名(去扩展名)全是数字 -> 直接拿数字当帧号(BenSMOT 的
            # 000001.jpg 布局),这样帧号与 gt.txt 里的帧号天然一致。
            self._by_t = {int(p.stem): p for p in files}
        except ValueError:
            # 有非数字文件名 -> 退回"排序后第 i 个文件当第 i 帧"(1-based)。
            self._by_t = {i: p for i, p in enumerate(files, 1)}
        self._cache: OrderedDict[int, Image.Image] = OrderedDict()  # LRU:有序字典模拟
        self._cache_size = cache_size

    def frame(self, t: int) -> Image.Image:
        cached = self._cache.get(t)
        if cached is not None:
            self._cache.move_to_end(t)  # 标记为最近使用,推到淘汰队列末尾
            return cached
        path = self._by_t.get(t)
        if path is None:
            available = sorted(self._by_t)
            raise KeyError(
                f"帧 {t} 不存在;可用帧号范围 {available[0]}~{available[-1]}"
            )
        image = Image.open(path).convert("RGB")  # 统一转 RGB,屏蔽调色板/灰度等格式差异
        self._cache[t] = image
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # 淘汰最久未使用的一项(队首)
        return image


class VideoFileFrameProvider:
    """视频文件形式(cv2.VideoCapture)。帧号 1-based,与 MOT 约定对齐。"""

    def __init__(self, path: str | Path):
        import cv2  # 延迟到实例化再要求 opencv

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {path}")

    def frame(self, t: int) -> Image.Image:
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, t - 1)  # t 是 1-based,cv2 是 0-based
        ok, bgr = self._cap.read()
        if not ok:
            raise KeyError(f"帧 {t} 超出视频范围")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)  # cv2 默认 BGR,PIL 需要 RGB
        return Image.fromarray(rgb)


def provider_for(path: str | Path) -> FrameProvider:
    """按路径类型选择帧提供者:目录 -> 图像目录,文件 -> 视频文件。"""
    p = Path(path)
    if p.is_dir():
        return ImageDirFrameProvider(p)
    if p.is_file():
        return VideoFileFrameProvider(p)
    raise FileNotFoundError(f"帧来源不存在: {path}")


def annotate_boxes(
    image: Image.Image, boxes: dict[int, tuple[float, float, float, float]]
) -> Image.Image:
    """在图像副本上为每个 track 画彩色框和 id 标签,返回副本(原图不动)。

    这是 id -> 画面目标的视觉 grounding:prompt 文本里会给出同一套
    颜色图例("id=1 is the red box"),模型据此把 track_id 对应到画面
    里的具体的人。
    """
    annotated = image.copy()  # 绝不原地改原图:同一帧对象可能被多个请求共享(见缓存)
    draw = ImageDraw.Draw(annotated)
    for track_id, box in sorted(boxes.items()):  # 排序保证多次调用画框顺序确定、可复现
        color = color_for_track(track_id)
        draw.rectangle(box, outline=color, width=3)
        # 标签画在框左上角外侧;max(..., 0) 防止框贴着图像顶边时文字画出画布外。
        draw.text((box[0] + 3, max(box[1] - 14, 0)), f"id={track_id}", fill=color)
    return annotated
