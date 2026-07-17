"""核心数据结构定义,供所有 SMOT 模块共享。

这些类型是设计文档 §5 中 JSON schema 的严格镜像:Trajectory(轨迹)、
Fact(运动事实)、PairFeature(pair 特征),以及三种最终输出断言
(instance / interaction / video)。全部使用 frozen dataclass(不可变),
这样可以放心地在各模块之间传递,不用担心被意外修改;同时也方便直接
序列化成 JSON。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# 边界框类型:(x1, y1, x2, y2),左上角+右下角坐标
Box = tuple[float, float, float, float]


def _asdict_json(obj) -> dict:
    """把一个(可能嵌套的)frozen dataclass 转成可以直接 json.dumps 的 dict。

    dataclasses.asdict() 本身会递归处理嵌套 dataclass,但它转换后 tuple
    还是 tuple(不是 list),而且 Enum 也不会自动变成普通值。这里统一做
    一次后处理,避免每个 assertion 类里都各写一遍转换逻辑。
    """
    raw = dataclasses.asdict(obj)
    return _tuples_to_lists(raw)


def _tuples_to_lists(value):
    """递归地把 dict/list/tuple 中的 tuple 转成 list,Enum 转成其 value。

    这是 _asdict_json 的实现细节:JSON 里没有 tuple 这个概念,数组和
    Enum 成员也不能直接被 json.dumps 处理,所以需要提前拍平。
    """
    if isinstance(value, dict):
        return {k: _tuples_to_lists(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tuples_to_lists(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True)
class FramePresence:
    """某个轨迹在某一帧 t 的观测结果:框、可选的 mask(RLE 编码字符串)、
    以及跟踪器给出的置信度。
    """

    t: int  # 帧号(从 0 开始,或数据集原生编号——与 Trajectory.present 同一套坐标)
    box: Box  # (x1, y1, x2, y2) 像素坐标边界框
    mask: Optional[str] = None  # RLE 字符串;没有 mask 时为 None
    conf: float = 1.0  # 跟踪器给出的置信度,[0, 1]


@dataclass(frozen=True)
class Trajectory:
    """冻结 Tracker 的输出:一个目标(track_id)在整段视频里的出现轨迹。

    present 是该目标出现的起止帧号区间 (t_in, t_out)(闭区间);
    per_frame 是它在这个区间内每一帧的具体观测(可能因为遮挡等原因不是
    每一帧都有观测,所以是稀疏列表而不是定长数组)。
    """

    track_id: int  # 目标的唯一编号,贯穿全部下游断言(instance/interaction)
    present: tuple[int, int]  # (t_in, t_out),闭区间
    per_frame: tuple[FramePresence, ...]  # 按 t 严格升序排列的稀疏观测列表

    def __post_init__(self):
        """构造时做数据契约校验。下游多个模块(净位移取首末帧、KFA 等
        间隔抽帧、approach 首尾距离对比)都隐式依赖 per_frame 按帧号
        升序;真实 tracker/数据加载器一旦输出乱序或帧号越界的数据,
        这些计算会"静默算错"而不报任何异常——所以在数据入口处直接
        fail-fast,把问题暴露在构造时而不是评测数字上。
        """
        if self.present[0] > self.present[1]:
            raise ValueError(
                f"Trajectory(track_id={self.track_id}): present 区间起点 "
                f"{self.present[0]} 大于终点 {self.present[1]}"
            )
        ts = [fp.t for fp in self.per_frame]
        for a, b in zip(ts, ts[1:]):
            if b <= a:
                raise ValueError(
                    f"Trajectory(track_id={self.track_id}): per_frame 帧号必须"
                    f"严格递增,发现 {a} 之后出现 {b}"
                )
        if ts and (ts[0] < self.present[0] or ts[-1] > self.present[1]):
            raise ValueError(
                f"Trajectory(track_id={self.track_id}): per_frame 帧号范围 "
                f"[{ts[0]}, {ts[-1]}] 超出 present 区间 {self.present}"
            )
        # 顺手建一个帧号 -> 观测的索引,让 frame_at 是 O(1)——事件过滤、
        # pair 事实、pair 特征构造都在 O(n^2) 的 pair 循环里逐帧调 frame_at,
        # 线性扫描会把每对的复杂度推到 O(n^2)。frozen dataclass 里用
        # object.__setattr__ 绕过不可变限制;_by_t 不是 dataclass field,
        # 不影响相等性/repr/dataclasses.asdict。
        object.__setattr__(self, "_by_t", {fp.t: fp for fp in self.per_frame})

    def frame_at(self, t: int) -> Optional[FramePresence]:
        """按帧号 t 查找该轨迹在这一帧的观测;找不到返回 None(表示该帧
        该目标缺失,例如被遮挡)。
        """
        return self._by_t.get(t)


class FactType(str, Enum):
    """Motion Fact Extractor 能抽取的事实类型枚举。"""

    PRESENCE = "presence"  # 出现时段
    NET_MOTION = "net_motion"  # 首帧到末帧的净位移
    SPEED = "speed"  # 平均速度(像素/帧)
    PROXIMITY = "proximity"  # 两个目标之间的距离(pair 事实)
    APPROACH = "approach"  # 两个目标是靠近/远离/保持不变(pair 事实)


# DeterministicFactSelector 使用的固定优先级顺序,同时也决定了
# Fact.embed 里 type_index 分量的取值(即该类型在这个元组中的下标)。
FACT_TYPE_ORDER: tuple[FactType, ...] = (
    FactType.PROXIMITY,
    FactType.APPROACH,
    FactType.NET_MOTION,
    FactType.SPEED,
    FactType.PRESENCE,
)


@dataclass(frozen=True)
class Fact:
    """Motion Fact Extractor 的输出。value 是确定性计算得到的(不学习,
    不会出现幻觉);embed 只是为了给未来 Stage-1 的可学习 Fact Selector
    提供一个可打分的向量,本身不参与 value 的计算。

    embed 约定(固定为 4 个 float):
        (type_index, norm_value, t_span_start_norm, t_span_end_norm)
    其中 type_index 是该事实类型在 FACT_TYPE_ORDER 中的下标;
    t_span 两个分量已按视频时长归一化到 [0, 1](否则长视频的原始帧号
    数值会淹没另外两个维度的打分信号);norm_value 暂时是原始数值
    (跨数据集的数值归一化需要数据集统计量,等真正训练 Fact Selector
    时再补,这是 Stage-1a 验收清单上的显式事项)。
    """

    type: FactType
    scope: str  # "instance:<i>"(单目标事实) 或 "pair:<i>,<j>"(pair 事实)
    value: object  # 确定性算出的原始取值,类型随 type 而变(数值/字符串/元组)
    t_span: tuple[int, int]  # 该事实成立的帧号区间(闭区间)
    embed: tuple[float, ...]  # 供 Fact Selector 打分用的定长向量,见下方约定


@dataclass(frozen=True)
class RelGeom:
    """一对目标在某一帧的相对几何关系,全部是确定性计算结果。"""

    rel_pos: tuple[float, float]  # 相对位置(j 相对 i 的位移)
    dist: float  # 中心点距离
    rel_vel: tuple[float, float]  # 相对速度
    orient: float  # 相对朝向角(弧度)
    overlap: float  # 重叠度(IoU)


@dataclass(frozen=True)
class PairFeature:
    """候选边(edge)在某一候选帧 t 上的 pair 级特征:两个目标各自的
    视觉特征向量,加上确定性算出的相对几何关系。
    """

    edge: tuple[int, int]  # (track_id_i, track_id_j),候选交互对
    t: int  # 该特征所属的候选帧号
    vis_i: tuple[float, ...]  # 目标 i 在帧 t 的视觉特征向量
    vis_j: tuple[float, ...]  # 目标 j 在帧 t 的视觉特征向量
    rel_geom: RelGeom  # i、j 之间确定性算出的相对几何关系


@dataclass(frozen=True)
class InstanceAssertion:
    """单目标行为描述断言:track_id + 一句话 caption + 时间段 + 证据帧。
    每一条描述都能回指到具体的目标、具体的时间段和具体的证据帧,
    这就是"可归因"的落地形式。
    """

    track_id: int  # 描述对象的目标编号
    caption: str  # MLLM 生成的一句话行为描述
    time_span: tuple[int, int]  # 描述所覆盖的帧号区间(闭区间)
    evidence_frames: tuple[int, ...]  # 生成该描述时实际喂给 MLLM 的关键帧号
    type: str = "instance"  # 输出 JSON 里的判别字段,固定值

    def to_json_dict(self) -> dict:
        """序列化为可直接 json.dumps 的 dict(tuple 转 list)。"""
        return _asdict_json(self)


@dataclass(frozen=True)
class InteractionAssertion:
    """两目标交互断言:谁对谁做了什么(subject -> object),用开放谓词
    predicate 加规范化标签 canonical_label 双重表示,同样带时间段和
    证据帧,可用于交互指标(role/direction/F1)的评测。
    """

    subject_id: int  # 交互发起方的目标编号
    object_id: int  # 交互承受方的目标编号
    predicate: str  # MLLM 原始输出的开放谓词(未规范化)
    canonical_label: str  # 映射到规范谓词表之后的标签
    time_span: tuple[int, int]  # 交互成立的帧号区间(闭区间)
    evidence_frames: tuple[int, ...]  # 生成该断言时实际喂给 MLLM 的关键帧号
    direction: str = "subj->obj"  # 方向标记,目前固定为 subject -> object
    confidence: float = 1.0  # 置信度,预留字段(当前恒为 1.0)
    type: str = "interaction"  # 输出 JSON 里的判别字段,固定值

    def to_json_dict(self) -> dict:
        """序列化为可直接 json.dumps 的 dict(tuple 转 list)。"""
        return _asdict_json(self)


@dataclass(frozen=True)
class VideoAssertion:
    """整段视频级别的概括断言:一句话总结 + 涉及到的所有 track_id。"""

    summary: str  # MLLM 生成的整段视频概括
    involved_ids: tuple[int, ...]  # 概括中涉及到的全部目标编号
    type: str = "video"  # 输出 JSON 里的判别字段,固定值

    def to_json_dict(self) -> dict:
        """序列化为可直接 json.dumps 的 dict(tuple 转 list)。"""
        return _asdict_json(self)
