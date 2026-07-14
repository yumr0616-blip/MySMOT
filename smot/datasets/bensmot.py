"""BenSMOT 原始标注 -> smot 内部类型的转换器(纯 stdlib)。

BenSMOT(ECCV 2024, "Beyond MOT: Semantic Multi-Object Tracking")的
原始发布形式是逐序列目录:

    BenSMOT/
    ├── train/<活动类别>/<序列名>/
    │   ├── gt/gt.txt              # MOTChallenge 格式的轨迹标注
    │   ├── imgs/000001.jpg ...    # 逐帧图像
    │   ├── video_caption.txt      # 整段视频的一句话概括
    │   ├── instance_captions.txt  # 每行 "名字: 描述",如 "woman0: ..."
    │   └── interactions.graphml   # 轨迹间交互关系图(GraphML)
    └── test/...

本模块把它们转换成:
  - Trajectory 元组         —— 当作冻结 tracker 的输出注入 Pipeline
                               (GT 轨迹,即"控制跟踪变量"的实验设置)
  - §5 gold 评测 payload    —— 与 PipelineResult.to_json_dict() 同形,
                               直接喂给 smot.eval
  - fact 数值统计(mean/std)—— 供 Stage-1a 前对 Fact.embed 的 norm_value
                               做数据集级归一化(设计文档记录的显式待办)

格式事实(2026-07 对全部 3282 段真实序列做过全量调查后确认):
  1. instance_captions.txt 行序 <-> gt.txt 中升序排列的 track_id 一一
     对应;95% 的序列两边数量相等,其余多为 caption 少于轨迹(背景
     人物有轨迹无描述)——映射按 zip 截断,没有 caption 的轨迹不进
     gold instances。
  2. interactions.graphml 的边属性键固定为 "relationship",值是
     WordNet synset 列表(如 "look.v.01,talk.v.01";约 1.7% 混有
     "cooperation"/"recieve" 之类的裸词或点号笔误)。一条边拆成
     多条断言,谓词取 synset 的 lemma;双向交互两条边各自标注。
  3. 92.7% 序列的 graphml 节点名与 caption 实例名一致(casefold 后);
     其余名字对不上(如 caption 写 woman0、graphml 写 man1 的标注
     笔误)——名字全部可解析时用名字映射,否则整体退回"节点文档
     顺序 <-> track_id 升序"的位置映射。
  4. 帧号是稀疏采样的(1,2,4,6,...),imgs/ 的文件名数字即帧号,
     gt.txt 只在有图像的帧上给框——Trajectory 本就支持稀疏观测。
  5. 122 段序列缺失全部语义标注文件,--skip-errors 模式下跳过。

命令行用法:
    python -m smot.datasets.bensmot probe <序列目录>          # 格式探查
    python -m smot.datasets.bensmot gold <根目录> -o gold.json
    python -m smot.datasets.bensmot stats <根目录> -o stats.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from smot.canonical_labels import map_predicate
from smot.motion_facts import MotionFactExtractor
from smot.tracker import VideoHandle
from smot.types import (
    FramePresence,
    InstanceAssertion,
    InteractionAssertion,
    Trajectory,
    VideoAssertion,
)

# GraphML 的 XML 命名空间(ElementTree 查询时必须显式带上)。
_GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}

# 从边属性里挑谓词时的键名提示词,按优先级排列:属性键名包含越靠前的
# 提示词,越可能是谓词字段。全部不命中时退回"第一个非数值字符串属性"。
_PREDICATE_KEY_HINTS = ("relation", "predicate", "interaction", "verb", "label", "caption")

# 从节点属性里解析实例显示名的键名提示词。
_NODE_NAME_KEY_HINTS = ("name", "label", "instance")

# 边属性里表示交互起止帧的键名(精确匹配,避免 "restart" 之类误命中)。
_SPAN_START_KEYS = frozenset({"start", "start_frame", "begin", "t_start", "from"})
_SPAN_END_KEYS = frozenset({"end", "end_frame", "finish", "t_end", "to"})

# §7 synonym-merged 档的合并表(对预测和 gold 同时应用)。按 gold 谓词
# 分布(look/talk/smile/converse/listen/hold/embrace/...)把明显同义的
# lemma 折叠到同一代表词:模型说 "chat" 而标注写 "converse"/"talk" 时,
# strict 档记错、synonym 档应记对。表刻意保守——只合并无争议的同义词,
# "smile"/"laugh"、"touch"/"hold" 这类近义但可区分的保持独立。
BENSMOT_SYNONYM_MAP: dict[str, str] = {
    "converse": "talk",
    "chat": "talk",
    "speak": "talk",
    "communicate": "talk",
    "watch": "look",
    "observe": "look",
    "see": "look",
    "stare": "look",
    "gaze": "look",
    "hug": "embrace",
    "cuddle": "embrace",
    "shake hands": "handshake",
    "cooperate": "collaborate",
    "cooperation": "collaborate",
    "hand": "give",
    "pass": "give",
    "receive": "accept",
    "recieve": "accept",  # 标注里的真实拼写错误
    "hear": "listen",
    "grab": "hold",
    "grip": "hold",
    # ---- P1 扩充(2026-07-14):由 100 段大评测的"未匹配 pred×gold
    # 共现统计"挖掘 + 人工审核,方法与证据量见 out/report/next_steps.md。
    # 表面归一化(模型输出风格差异,非语义差异):
    "look at": "look",  # 共现 17 次;"look at" 在 pred 词表出现 35 次
    "smile at": "smile",
    # 语义同义:
    "pet": "caress",  # 共现 8 次(抚摸宠物场景)
    "seize": "hold",  # 共现 6 次;与既有 grab/grip -> hold 同簇
    "return": "give",  # return.v.06 = 归还,语义是"给回去"(递还现金实证)
    "hash out": "talk",  # hash_out.v.01 的 WordNet 释义即 "talk over in detail"
    # 标注笔误(与上面 recieve 同类):
    "reveive": "accept",
    "smile.0.01": "smile",  # 坏 synset 名(应为 smile.v.01),按词干归一
}


@dataclass(frozen=True)
class BenSMOTSequence:
    """一段 BenSMOT 序列转换后的全部内容。"""

    name: str  # "活动类别/序列名",用于人类可读的溯源
    seq_dir: str  # 序列目录的绝对路径
    trajectories: tuple[Trajectory, ...]
    instance_names: dict[int, str]  # track_id -> 标注里的实例名(如 "woman0")
    instance_captions: dict[int, str]  # track_id -> gold caption
    video_caption: str
    interactions: tuple[InteractionAssertion, ...]  # gold 交互断言
    num_frames: int


@dataclass(frozen=True)
class _GraphMLEdge:
    """GraphML 边的中间表示:源/目标节点 id + 属性名->值的字典。"""

    source: str
    target: str
    data: dict[str, str]


# ---------------------------------------------------------------------------
# gt.txt(MOTChallenge 格式)
# ---------------------------------------------------------------------------

def parse_gt_txt(path: str | os.PathLike) -> tuple[Trajectory, ...]:
    """解析 MOTChallenge 格式的 gt.txt 为 Trajectory 元组(按 track_id 升序)。

    每行: frame,id,x,y,w,h[,consider,class,visibility]
      - 框从 xywh(左上角+宽高)转成本项目的 xyxy(左上角+右下角);
      - 第 7 列是 MOT 的 consider 标志,为 0 的行(distractor 等)跳过;
      - 第 9 列 visibility(0~1)如存在则作为 FramePresence.conf;
      - 同一 (id, frame) 出现多行时保留第一行(真实标注不应出现,
        容忍而不放大)。
    Trajectory 构造器本身会对帧号排序性/区间做 fail-fast 校验。
    """
    rows_by_id: dict[int, dict[int, FramePresence]] = {}
    with open(path, encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no}: gt 行列数不足 6: {line!r}")
            try:
                t = int(float(parts[0]))  # int(float(...)):容忍 "1.0" 这类带小数点的整数写法
                track_id = int(float(parts[1]))
                x, y, w, h = (float(v) for v in parts[2:6])  # 左上角坐标 + 宽高
                consider = float(parts[6]) if len(parts) >= 7 and parts[6] else 1.0  # 第7列缺失时默认"考虑"
                conf = float(parts[8]) if len(parts) >= 9 and parts[8] else 1.0  # 第9列缺失时默认满置信度
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: 无法解析 gt 行: {line!r}") from exc
            if consider == 0.0:
                continue  # MOT 的 consider=0 标记该行不应参与评测(如 distractor 类)
            # 真实标注里约 90 行(全量扫描确认)宽/高为负值——框拟合工具
            # 偶发的角点错序,不是转换器能预判的格式假设。按 min/max
            # 归一化角点(而非原样传播 x2<x1 的退化框):后续任何按
            # (x1,y1,x2,y2) 假设作画/求交/求 IoU 的代码都默认框合法,
            # 不归一化会在 PIL 画框时 ValueError,或让 IoU/距离静默算错。
            x1, x2 = (x, x + w) if w >= 0 else (x + w, x)
            y1, y2 = (y, y + h) if h >= 0 else (y + h, y)
            per_frame = rows_by_id.setdefault(track_id, {})
            per_frame.setdefault(  # setdefault:同一 (id, frame) 重复出现时保留第一行
                t, FramePresence(t=t, box=(x1, y1, x2, y2), conf=conf)
            )
    trajectories = []
    for track_id in sorted(rows_by_id):  # 按 track_id 升序,格式事实 #1 的映射依赖这个顺序
        frames = tuple(
            rows_by_id[track_id][t] for t in sorted(rows_by_id[track_id])  # 按帧号升序排列
        )
        trajectories.append(
            Trajectory(
                track_id=track_id,
                present=(frames[0].t, frames[-1].t),  # 首末观测帧号作为出现区间
                per_frame=frames,
            )
        )
    return tuple(trajectories)


# ---------------------------------------------------------------------------
# instance_captions.txt / video_caption.txt
# ---------------------------------------------------------------------------

def parse_instance_captions(path: str | os.PathLike) -> list[tuple[str, str]]:
    """解析 instance_captions.txt,返回 [(实例名, caption), ...](保持行序)。

    行格式为 "名字: 描述";用 partition 只按第一个冒号切,caption 内部
    可以再出现冒号。空行跳过;没有冒号的行直接报错(格式假设被破坏时
    尽早暴露,而不是静默丢数据)。
    """
    entries: list[tuple[str, str]] = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            name, sep, caption = line.partition(":")
            if not sep or not name.strip():
                raise ValueError(
                    f"{path}:{line_no}: 期望 '实例名: 描述' 格式,得到: {line!r}"
                )
            entries.append((name.strip(), caption.strip()))
    return entries


def map_names_to_track_ids(
    names: list[str], track_ids: list[int], context: str = ""
) -> dict[str, int]:
    """实例名 -> track_id 的映射(格式事实 #1:行序 <-> id 升序)。

    数量不一致是真实数据的常态(约 5% 序列的背景人物有轨迹无 caption,
    个别序列 caption 反而更多)——按 zip 截断:第 i 行 caption 对应第 i
    小的 track_id,多出来的一侧丢弃。重复实例名仍然 fail-fast(那说明
    行序映射本身不可信)。
    """
    if len(names) != len(set(names)):
        raise ValueError(f"{context}: instance_captions 中存在重复实例名: {names}")
    return {name: tid for name, tid in zip(names, sorted(track_ids))}


# ---------------------------------------------------------------------------
# interactions.graphml
# ---------------------------------------------------------------------------

def _parse_graphml(path: str | os.PathLike) -> tuple[dict[str, str], list[_GraphMLEdge]]:
    """解析 GraphML,返回 (节点id->显示名, 边列表)。

    <key> 声明把属性 id(如 "d0")翻译成属性名(如 "relation");节点的
    显示名优先取其 name/label/instance 属性,没有属性时就用节点 id 本身
    (BenSMOT 的节点 id 预期直接就是 "woman0" 这类实例名)。
    """
    root = ET.parse(path).getroot()
    # GraphML 的 <key> 元素声明了属性 id(如 "d0")到人类可读属性名
    # (如 "relation")的映射;<data key="d0"> 里的 "d0" 必须经这张表翻译。
    key_names = {
        k.get("id"): (k.get("attr.name") or k.get("id"))
        for k in root.findall("g:key", _GRAPHML_NS)
    }

    def data_dict(element) -> dict[str, str]:
        """把一个 <node>/<edge> 元素下的全部 <data> 子元素翻译成 {属性名: 值}。"""
        return {
            key_names.get(d.get("key"), d.get("key") or ""): (d.text or "").strip()
            for d in element.findall("g:data", _GRAPHML_NS)
        }

    node_names: dict[str, str] = {}
    edges: list[_GraphMLEdge] = []
    for graph in root.findall("g:graph", _GRAPHML_NS):
        for node in graph.findall("g:node", _GRAPHML_NS):
            node_id = node.get("id") or ""
            data = data_dict(node)
            display = node_id  # 默认显示名就是节点 id 本身
            # 按提示词优先级依次找:命中即用该属性值覆盖 display 并跳出双重循环。
            for hint in _NODE_NAME_KEY_HINTS:
                for key, value in data.items():
                    if hint in key.casefold() and value:
                        display = value
                        break
                if display != node_id:
                    break
            node_names[node_id] = display
        for edge in graph.findall("g:edge", _GRAPHML_NS):
            edges.append(
                _GraphMLEdge(
                    source=edge.get("source") or "",
                    target=edge.get("target") or "",
                    data=data_dict(edge),
                )
            )
    return node_names, edges


def _looks_numeric(value: str) -> bool:
    """value 能否解析成浮点数——用来把"数值型属性"(如帧号)与"文本型
    属性"(如谓词)区分开,而不是硬编码属性名。"""
    try:
        float(value)
        return True
    except ValueError:
        return False


# WordNet synset 形态:lemma.词性.两位序号,如 "look.v.01"、"shake_hands.v.01"。
_SYNSET_RE = re.compile(r"([a-zA-Z_'\-]+)\.[a-z]\.\d{2}")


def parse_predicates(value: str) -> tuple[str, ...]:
    """把边属性值解析成谓词元组(格式事实 #2)。

    值的主体形态是逗号分隔的 synset 列表("look.v.01,talk.v.01"),
    谓词取 lemma(下划线还原成空格);真实标注里混有约 1.7% 的脏条目:
    裸词("cooperation"、拼错的 "recieve")原样保留(小写),
    点号连写("clap.v.04.take.v.04")靠 findall 拆出全部 synset。
    同一条边内去重、保持出现顺序。
    """
    out: list[str] = []
    for part in value.split(","):  # 主体形态是逗号分隔的 synset 列表
        part = part.strip()
        if not part:
            continue
        synsets = _SYNSET_RE.findall(part)  # 一个 part 里可能连写多个 synset(标注笔误)
        if synsets:
            out.extend(s.replace("_", " ").lower() for s in synsets)  # lemma 下划线还原成空格
        else:
            out.append(part.replace("_", " ").lower())  # 不匹配 synset 形态的裸词,原样小写保留
    seen: set[str] = set()
    # `p in seen or seen.add(p)` 是保序去重的惯用写法:or 的短路特性让
    # add() 只在"没见过"时执行(add 返回 None,恒为假,不影响 or 的结果),
    # 同时把 p 记进 seen 供后续元素判重。
    unique = [p for p in out if not (p in seen or seen.add(p))]
    return tuple(unique)


def _pick_predicate(data: dict[str, str], context: str) -> str:
    """从边属性里挑出谓词字符串(格式假设 #2)。

    先按 _PREDICATE_KEY_HINTS 的键名提示找;全不命中时退回第一个
    非数值的字符串属性;还找不到就报错并列出所有属性,让 probe 输出
    能直接定位真实数据里谓词到底放在哪个键下。
    """
    for hint in _PREDICATE_KEY_HINTS:
        for key, value in data.items():
            if hint in key.casefold() and value:
                return value
    for value in data.values():
        if value and not _looks_numeric(value):
            return value
    raise ValueError(f"{context}: 边属性中找不到谓词字符串,属性为: {data}")


def _span_from_edge_data(data: dict[str, str]) -> Optional[tuple[int, int]]:
    """若边属性带起止帧(键名精确匹配 _SPAN_*_KEYS),解析为闭区间。"""
    start: Optional[int] = None
    end: Optional[int] = None
    for key, value in data.items():
        k = key.casefold()
        if k in _SPAN_START_KEYS and _looks_numeric(value):
            start = int(float(value))
        elif k in _SPAN_END_KEYS and _looks_numeric(value):
            end = int(float(value))
    if start is None or end is None:
        return None
    return (min(start, end), max(start, end))


def _default_pair_span(traj_a: Trajectory, traj_b: Trajectory) -> tuple[int, int]:
    """边上没有起止帧属性时的兜底 time_span:两条轨迹 present 区间的交集
    (交互只可能发生在双方都在场时);完全不相交时退化为并集区间,
    保证 span 始终合法(评测的 time IoU 对 gold 的这个字段并不使用,
    这里只求形状正确、语义尽量合理)。
    """
    lo = max(traj_a.present[0], traj_b.present[0])
    hi = min(traj_a.present[1], traj_b.present[1])
    if lo > hi:
        lo = min(traj_a.present[0], traj_b.present[0])
        hi = max(traj_a.present[1], traj_b.present[1])
    return (lo, hi)


def _resolve_graphml_nodes(
    node_names: dict[str, str],
    name_to_id: dict[str, int],
    track_ids: list[int],
) -> dict[str, int]:
    """graphml 节点 id -> track_id(格式事实 #3 的两级策略)。

    优先名字映射(casefold 容忍大小写/空格差异,覆盖 92.7% 的序列);
    任何一个节点名解析失败,就整体退回位置映射:节点在文档中的出现
    顺序 <-> track_id 升序(调查确认标注工具按人物顺序生成节点,数字
    后缀是按类别各自计数的,不能单独当顺序用)。位置映射下节点数超出
    轨迹数的溢出节点不进映射,引用它们的边由调用方丢弃。
    """
    norm = {name.strip().casefold(): tid for name, tid in name_to_id.items()}  # 大小写/空格不敏感的查找表

    def by_name(node_id: str) -> Optional[int]:
        display = node_names.get(node_id, node_id) or node_id
        return norm.get(display.strip().casefold())

    resolved = {nid: by_name(nid) for nid in node_names}
    if resolved and all(tid is not None for tid in resolved.values()):
        return resolved  # type: ignore[return-value]  # 全部节点都按名字解析成功,直接采用
    # 只要有一个节点名对不上,就不信任名字映射,整体退回位置映射
    # (格式事实 #3:标注工具按人物顺序生成节点,不是按数字后缀排序)。
    ordered = sorted(track_ids)
    return {
        nid: ordered[i] for i, nid in enumerate(node_names) if i < len(ordered)
    }


def graphml_to_interactions(
    path: str | os.PathLike,
    name_to_id: dict[str, int],
    traj_by_id: dict[int, Trajectory],
) -> tuple[InteractionAssertion, ...]:
    """把 interactions.graphml 转成 gold InteractionAssertion 元组。

    - 边方向 source -> target 解释为 subject -> object,双向交互在
      标注里本来就是两条边;
    - 一条边的 relationship 值经 parse_predicates 拆成多个谓词,
      每个谓词一条断言;
    - canonical_label 经 map_predicate 规范化(synset lemma 查不到映射
      时保留小写原文,与预测侧同一套规则,保证 strict 层能对得上);
    - 节点解析不出 track_id(位置映射下的溢出节点)或自环的边直接
      丢弃——那是标注对不上号的边,比错配成随机 track 更安全;
    - gold 没有"证据帧"概念,evidence_frames 置空(评测不消费该字段)。
    """
    node_names, edges = _parse_graphml(path)
    node_to_tid = _resolve_graphml_nodes(
        node_names, name_to_id, list(traj_by_id)
    )

    assertions: list[InteractionAssertion] = []
    for index, edge in enumerate(edges):
        context = f"{path} 第 {index} 条边"
        subject_id = node_to_tid.get(edge.source)
        object_id = node_to_tid.get(edge.target)
        if subject_id is None or object_id is None or subject_id == object_id:
            continue  # 解析不出 id(溢出节点)或自环边,丢弃比错配到随机 track 更安全
        value = _pick_predicate(edge.data, context)
        time_span = _span_from_edge_data(edge.data) or _default_pair_span(
            traj_by_id[subject_id], traj_by_id[object_id]
        )
        for predicate in parse_predicates(value):
            assertions.append(
                InteractionAssertion(
                    subject_id=subject_id,
                    object_id=object_id,
                    predicate=predicate,
                    canonical_label=map_predicate(predicate),
                    time_span=time_span,
                    evidence_frames=(),
                )
            )
    return tuple(assertions)


# ---------------------------------------------------------------------------
# 序列级 / 数据集级入口
# ---------------------------------------------------------------------------

def load_sequence(seq_dir: str | os.PathLike) -> BenSMOTSequence:
    """加载一个 BenSMOT 序列目录的全部标注。

    gt.txt 与 instance_captions.txt 是必需的(缺失直接 FileNotFoundError);
    video_caption.txt / interactions.graphml 允许缺失(分别退化为空串/
    空交互,真实数据里可能存在无交互的序列)。
    """
    seq = Path(seq_dir)
    trajectories = parse_gt_txt(seq / "gt" / "gt.txt")
    entries = parse_instance_captions(seq / "instance_captions.txt")
    name_to_id = map_names_to_track_ids(
        [name for name, _ in entries],
        [traj.track_id for traj in trajectories],
        context=str(seq),
    )
    caption_by_id = {name_to_id[name]: caption for name, caption in entries}

    video_caption_path = seq / "video_caption.txt"
    video_caption = (
        video_caption_path.read_text(encoding="utf-8-sig").strip()
        if video_caption_path.is_file()
        else ""  # 允许缺失,退化为空概括(不进 video 任务的训练/评测)
    )

    traj_by_id = {traj.track_id: traj for traj in trajectories}
    graphml_path = seq / "interactions.graphml"
    interactions = (
        graphml_to_interactions(graphml_path, name_to_id, traj_by_id)
        if graphml_path.is_file()
        else ()  # 允许缺失,退化为"这段序列没有交互标注"
    )

    imgs_dir = seq / "imgs"
    if imgs_dir.is_dir():
        num_frames = sum(1 for p in imgs_dir.iterdir() if p.is_file())
    else:
        # 没有图像目录(如某些测试 fixture)时,用轨迹里出现过的最大帧号兜底。
        num_frames = max((traj.present[1] for traj in trajectories), default=0)

    return BenSMOTSequence(
        name="/".join(seq.resolve().parts[-2:]),
        seq_dir=str(seq.resolve()),
        trajectories=trajectories,
        instance_names={tid: name for name, tid in name_to_id.items()},
        instance_captions=caption_by_id,
        video_caption=video_caption,
        interactions=interactions,
        num_frames=num_frames,
    )


def iter_sequences(root: str | os.PathLike) -> Iterator[Path]:
    """递归找出 root 下所有的序列目录(判据:包含 gt/gt.txt)。

    对目录名排序,保证遍历顺序确定(gold 与 pred 的多视频配对靠下标,
    顺序必须稳定可复现)。root 本身是序列目录时直接返回它。
    """
    root_path = Path(root)
    if (root_path / "gt" / "gt.txt").is_file():
        yield root_path
        return
    for dirpath, dirnames, _filenames in os.walk(root_path):
        dirnames.sort()  # 原地排序 os.walk 的下钻列表,决定后续子目录的遍历顺序
        current = Path(dirpath)
        if (current / "gt" / "gt.txt").is_file():
            dirnames.clear()  # 序列目录内部不再下钻
            yield current


def load_split(
    root: str | os.PathLike,
    limit: Optional[int] = None,
    on_error: str = "raise",
) -> tuple[list[BenSMOTSequence], list[tuple[str, str]]]:
    """加载 root 下的(至多 limit 个)序列。

    on_error="raise" 时任何一个序列解析失败都直接抛出(默认,fail-fast);
    on_error="skip" 时收集 (序列路径, 错误信息) 继续加载——真实数据集
    难免有个别脏标注,整体转换不应被单点卡死,但跳过必须显式可见。
    """
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error 只能是 'raise' 或 'skip',得到 {on_error!r}")
    sequences: list[BenSMOTSequence] = []
    errors: list[tuple[str, str]] = []
    for seq_dir in iter_sequences(root):
        if limit is not None and len(sequences) >= limit:
            break
        try:
            sequences.append(load_sequence(seq_dir))
        except Exception as exc:  # noqa: BLE001 - 跳过模式下按序列粒度容错
            if on_error == "raise":
                raise
            errors.append((str(seq_dir), f"{type(exc).__name__}: {exc}"))
    return sequences, errors


def sequence_to_video_handle(seq: BenSMOTSequence, fps: float = 1.0) -> VideoHandle:
    """给 Pipeline.run() 用的 VideoHandle(path 指向 imgs 目录)。"""
    return VideoHandle(
        path=str(Path(seq.seq_dir) / "imgs"), num_frames=seq.num_frames, fps=fps
    )


def sequence_to_gold_payload(seq: BenSMOTSequence) -> dict:
    """把一段序列的 gold 标注转成与 PipelineResult.to_json_dict() 同形的
    payload(可直接落盘、直接喂 smot.eval)。复用 §5 的断言 dataclass
    来构造,保证字段名/结构与预测侧永远同步,而不是手写第二份 schema。

    gold instances 只包含有 caption 的轨迹:没有语义标注的背景人物不该
    进 coverage 指标的分母(预测侧描述了它们也不算错,只是 gold 没有
    可对照的内容)。
    """
    instances = [
        InstanceAssertion(
            track_id=traj.track_id,
            caption=seq.instance_captions[traj.track_id],
            time_span=traj.present,
            evidence_frames=(),  # gold 无证据帧概念,评测也不消费此字段
        ).to_json_dict()
        for traj in seq.trajectories
        if traj.track_id in seq.instance_captions
    ]
    video = VideoAssertion(
        summary=seq.video_caption,
        involved_ids=tuple(sorted(traj.track_id for traj in seq.trajectories)),
    ).to_json_dict()
    return {
        "sequence": seq.name,  # 溯源用的附加键,评测端忽略未知键
        "instances": instances,
        "interactions": [a.to_json_dict() for a in seq.interactions],
        "video": video,
    }


def build_gold_payloads(sequences: list[BenSMOTSequence]) -> list[dict]:
    """多序列版 gold payload(顺序与传入序列一致——评测按下标配对)。"""
    return [sequence_to_gold_payload(seq) for seq in sequences]


# ---------------------------------------------------------------------------
# fact 数值统计(Stage-1a 归一化用)
# ---------------------------------------------------------------------------

class _RunningStat:
    """Welford 在线均值/方差(总体标准差),避免两遍扫描或数值溢出。"""

    __slots__ = ("n", "mean", "_m2")  # 三个累计量之外不存原始数据,内存 O(1) 而非 O(样本数)

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0  # 与当前 mean 的离差平方和(sum of squared deviations)

    def add(self, x: float) -> None:
        """喂入一个新样本,增量更新 n/mean/_m2(Welford 算法核心三行)。"""
        self.n += 1
        delta = x - self.mean  # 新样本相对"旧"均值的离差
        self.mean += delta / self.n  # 更新均值
        self._m2 += delta * (x - self.mean)  # 注意第二个 (x - self.mean) 用的是"新"均值——
        # 这个看似不对称的写法正是 Welford 算法数值稳定的关键,等价于但避免了
        # sum((x-mean)^2) 两遍扫描或 sum(x^2)-n*mean^2 的大数相减误差。

    @property
    def std(self) -> float:
        """总体标准差(除以 n,不是 n-1 的样本标准差)。"""
        return math.sqrt(self._m2 / self.n) if self.n else 0.0


def compute_fact_statistics(sequences: list[BenSMOTSequence]) -> dict:
    """对全体序列跑 MotionFactExtractor,按事实类型统计 embed 的
    norm_value 分量(embed[1],目前是未归一化的原始数值)的 n/mean/std。

    这是设计文档记录的 Stage-1a 前置事项:可学习 Fact Selector 的打分
    输入需要跨数据集可比的数值尺度,训练侧将用这里的统计量做 z-score。
    """
    extractor = MotionFactExtractor()
    stats: dict[str, _RunningStat] = {}  # 按事实类型(如 "speed")各维护一份在线统计
    for seq in sequences:
        for fact in extractor.extract(list(seq.trajectories)):
            stats.setdefault(fact.type.value, _RunningStat()).add(float(fact.embed[1]))
    return {
        fact_type: {"n": s.n, "mean": s.mean, "std": s.std}
        for fact_type, s in sorted(stats.items())  # 排序保证输出 JSON 的键顺序确定、可复现
    }


# ---------------------------------------------------------------------------
# 格式探查(probe)
# ---------------------------------------------------------------------------

def describe_sequence(seq_dir: str | os.PathLike) -> str:
    """对一个序列目录做尽力而为的格式探查,输出人类可读的报告。

    专门用于在真实数据到手后验证本模块的两个格式假设;任何一段解析
    失败只记录错误、不中断其余段落(探查工具自身绝不能因为格式意外
    而崩溃,那正是它要报告的东西)。
    """
    seq = Path(seq_dir)
    lines: list[str] = [f"== BenSMOT 序列探查: {seq} =="]

    def section(title: str, body_fn) -> None:
        lines.append(f"-- {title} --")
        try:
            body = body_fn()
        except Exception as exc:  # noqa: BLE001 - 探查工具按段落容错
            lines.append(f"  [解析失败] {type(exc).__name__}: {exc}")
            return
        lines.extend(f"  {row}" for row in body)

    def files_body() -> list[str]:
        expected = [
            Path("gt") / "gt.txt",
            Path("imgs"),
            Path("video_caption.txt"),
            Path("instance_captions.txt"),
            Path("interactions.graphml"),
        ]
        rows = []
        for rel in expected:
            p = seq / rel
            if p.is_dir():
                rows.append(f"{rel}: 目录, {sum(1 for _ in p.iterdir())} 项")
            elif p.is_file():
                rows.append(f"{rel}: 文件, {p.stat().st_size} 字节")
            else:
                rows.append(f"{rel}: 缺失")
        return rows

    def gt_body() -> list[str]:
        with open(seq / "gt" / "gt.txt", encoding="utf-8-sig") as f:
            first = f.readline().strip()
            n_rows = 1 + sum(1 for _ in f)
        trajectories = parse_gt_txt(seq / "gt" / "gt.txt")
        return [
            f"共 {n_rows} 行; 首行: {first!r} ({len(first.split(','))} 列)",
            f"轨迹数 {len(trajectories)}; track_id: "
            f"{[t.track_id for t in trajectories]}",
            f"帧号范围: {min(t.present[0] for t in trajectories)}"
            f" ~ {max(t.present[1] for t in trajectories)}",
        ]

    def captions_body() -> list[str]:
        entries = parse_instance_captions(seq / "instance_captions.txt")
        rows = [f"{name}: {caption[:60]}" for name, caption in entries]
        trajectories = parse_gt_txt(seq / "gt" / "gt.txt")
        mapping = map_names_to_track_ids(
            [n for n, _ in entries],
            [t.track_id for t in trajectories],
            context=str(seq),
        )
        rows.append(f"名字->track_id 映射(行序<->id升序假设): {mapping}")
        return rows

    def graphml_body() -> list[str]:
        node_names, edges = _parse_graphml(seq / "interactions.graphml")
        rows = [f"节点: {node_names}"]
        entries = parse_instance_captions(seq / "instance_captions.txt")
        trajectories = parse_gt_txt(seq / "gt" / "gt.txt")
        name_to_id = map_names_to_track_ids(
            [n for n, _ in entries],
            [t.track_id for t in trajectories],
            context=str(seq),
        )
        node_to_tid = _resolve_graphml_nodes(
            node_names, name_to_id, [t.track_id for t in trajectories]
        )
        by_name = {
            name.strip().casefold() for name in name_to_id
        } >= {
            (node_names.get(n, n) or n).strip().casefold() for n in node_names
        }
        rows.append(
            f"节点->track_id 映射({'名字匹配' if by_name else '位置回退'}): "
            f"{node_to_tid}"
        )
        for i, edge in enumerate(edges[:10]):
            value = _pick_predicate(edge.data, f"边[{i}]")
            rows.append(
                f"边[{i}]: {edge.source} -> {edge.target}, 原始 {value!r} "
                f"-> 谓词 {list(parse_predicates(value))}"
            )
        if len(edges) > 10:
            rows.append(f"... 共 {len(edges)} 条边")
        return rows

    section("文件清单", files_body)
    section("gt.txt", gt_body)
    section("instance_captions.txt", captions_body)
    section("interactions.graphml", graphml_body)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m smot.datasets.bensmot",
        description="BenSMOT 标注转换/探查工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="对单个序列目录做格式探查")
    p_probe.add_argument("seq_dir")

    for name, help_text in (
        ("gold", "把标注转成 gold 评测 payload(JSON 列表)"),
        ("stats", "统计 fact 数值分布(Stage-1a 归一化用)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("root")
        p.add_argument("-o", "--out", required=True)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument(
            "--skip-errors",
            action="store_true",
            help="解析失败的序列跳过并在 stderr 报告,而不是整体中止",
        )

    args = parser.parse_args(argv)

    if args.command == "probe":
        print(describe_sequence(args.seq_dir))
        return 0

    sequences, errors = load_split(
        args.root, limit=args.limit, on_error="skip" if args.skip_errors else "raise"
    )
    for seq_dir, message in errors:
        print(f"[跳过] {seq_dir}: {message}", file=sys.stderr)
    if not sequences:
        print(f"在 {args.root} 下没有找到可用序列", file=sys.stderr)
        return 1

    if args.command == "gold":
        payload = build_gold_payloads(sequences)
    else:  # stats
        payload = compute_fact_statistics(sequences)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(
        f"已写入 {args.out}(序列数 {len(sequences)},跳过 {len(errors)})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
