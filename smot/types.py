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

    t: int
    box: Box
    mask: Optional[str] = None  # RLE 字符串;没有 mask 时为 None
    conf: float = 1.0


@dataclass(frozen=True)
class Trajectory:
    """冻结 Tracker 的输出:一个目标(track_id)在整段视频里的出现轨迹。

    present 是该目标出现的起止帧号区间 (t_in, t_out)(闭区间);
    per_frame 是它在这个区间内每一帧的具体观测(可能因为遮挡等原因不是
    每一帧都有观测,所以是稀疏列表而不是定长数组)。
    """

    track_id: int
    present: tuple[int, int]  # (t_in, t_out),闭区间
    per_frame: tuple[FramePresence, ...]

    def frame_at(self, t: int) -> Optional[FramePresence]:
        """按帧号 t 查找该轨迹在这一帧的观测;找不到返回 None(表示该帧
        该目标缺失,例如被遮挡)。
        """
        for fp in self.per_frame:
            if fp.t == t:
                return fp
        return None

    def frames_in_span(self, t_a: int, t_b: int) -> tuple[FramePresence, ...]:
        """返回落在 [t_a, t_b] 闭区间内的所有帧观测,按原有顺序。"""
        return tuple(fp for fp in self.per_frame if t_a <= fp.t <= t_b)


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
    norm_value / t_span 在本脚手架阶段暂时就是原始数值(还没有做
    跨数据集的归一化,等真正训练 Fact Selector 时再补)。
    """

    type: FactType
    scope: str  # "instance:<i>"(单目标事实) 或 "pair:<i>,<j>"(pair 事实)
    value: object
    t_span: tuple[int, int]
    embed: tuple[float, ...]


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

    edge: tuple[int, int]
    t: int
    vis_i: tuple[float, ...]
    vis_j: tuple[float, ...]
    rel_geom: RelGeom


@dataclass(frozen=True)
class InstanceAssertion:
    """单目标行为描述断言:track_id + 一句话 caption + 时间段 + 证据帧。
    每一条描述都能回指到具体的目标、具体的时间段和具体的证据帧,
    这就是"可归因"的落地形式。
    """

    track_id: int
    caption: str
    time_span: tuple[int, int]
    evidence_frames: tuple[int, ...]
    type: str = "instance"

    def to_json_dict(self) -> dict:
        return _asdict_json(self)


@dataclass(frozen=True)
class InteractionAssertion:
    """两目标交互断言:谁对谁做了什么(subject -> object),用开放谓词
    predicate 加规范化标签 canonical_label 双重表示,同样带时间段和
    证据帧,可用于交互指标(role/direction/F1)的评测。
    """

    subject_id: int
    object_id: int
    predicate: str  # MLLM 原始输出的开放谓词(未规范化)
    canonical_label: str  # 映射到规范谓词表之后的标签
    time_span: tuple[int, int]
    evidence_frames: tuple[int, ...]
    direction: str = "subj->obj"
    confidence: float = 1.0
    type: str = "interaction"

    def to_json_dict(self) -> dict:
        return _asdict_json(self)


@dataclass(frozen=True)
class VideoAssertion:
    """整段视频级别的概括断言:一句话总结 + 涉及到的所有 track_id。"""

    summary: str
    involved_ids: tuple[int, ...]
    type: str = "video"

    def to_json_dict(self) -> dict:
        return _asdict_json(self)
