"""Core data schemas shared across all SMOT modules.

These mirror the JSON schemas from the design doc's data-structures section
verbatim: Trajectory, Fact, PairFeature, and the three assertion types
(instance / interaction / video). All types are plain, frozen dataclasses so
they can be passed safely across module boundaries and serialized to JSON.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def _asdict_json(obj) -> dict:
    """Convert a (possibly nested) frozen dataclass to a JSON-safe dict.

    dataclasses.asdict() already recurses into nested dataclasses, but it
    leaves tuples as tuples; json.dumps handles tuples fine (as arrays), so
    this is mostly a documented single choke point in case that ever needs
    to change (e.g. enum values -> their .value).
    """
    raw = dataclasses.asdict(obj)
    return _tuples_to_lists(raw)


def _tuples_to_lists(value):
    if isinstance(value, dict):
        return {k: _tuples_to_lists(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tuples_to_lists(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True)
class FramePresence:
    t: int
    box: Box
    mask: Optional[str] = None  # RLE string; None when no mask available
    conf: float = 1.0


@dataclass(frozen=True)
class Trajectory:
    """Frozen Tracker output: one object's presence across frames."""

    track_id: int
    present: tuple[int, int]  # (t_in, t_out), inclusive
    per_frame: tuple[FramePresence, ...]

    def frame_at(self, t: int) -> Optional[FramePresence]:
        for fp in self.per_frame:
            if fp.t == t:
                return fp
        return None

    def frames_in_span(self, t_a: int, t_b: int) -> tuple[FramePresence, ...]:
        return tuple(fp for fp in self.per_frame if t_a <= fp.t <= t_b)


class FactType(str, Enum):
    PRESENCE = "presence"
    NET_MOTION = "net_motion"
    SPEED = "speed"
    PROXIMITY = "proximity"
    APPROACH = "approach"


# Fixed priority order used by DeterministicFactSelector (smot/fact_selector.py)
# and by Fact.embed's type_index component below.
FACT_TYPE_ORDER: tuple[FactType, ...] = (
    FactType.PROXIMITY,
    FactType.APPROACH,
    FactType.NET_MOTION,
    FactType.SPEED,
    FactType.PRESENCE,
)


@dataclass(frozen=True)
class Fact:
    """Motion Fact Extractor output. Value is deterministic (not learned);
    embed is provided only so a future learnable Fact Selector has something
    to score.

    embed convention (fixed, 4 floats): (type_index, norm_value,
    t_span_start_norm, t_span_end_norm). type_index is the fact's position
    in FACT_TYPE_ORDER; norm_value/t_span are left as raw floats in this
    scaffold (no dataset-wide normalization exists yet).
    """

    type: FactType
    scope: str  # "instance:<i>" or "pair:<i>,<j>"
    value: object
    t_span: tuple[int, int]
    embed: tuple[float, ...]


@dataclass(frozen=True)
class RelGeom:
    rel_pos: tuple[float, float]
    dist: float
    rel_vel: tuple[float, float]
    orient: float
    overlap: float


@dataclass(frozen=True)
class PairFeature:
    edge: tuple[int, int]
    t: int
    vis_i: tuple[float, ...]
    vis_j: tuple[float, ...]
    rel_geom: RelGeom


@dataclass(frozen=True)
class InstanceAssertion:
    track_id: int
    caption: str
    time_span: tuple[int, int]
    evidence_frames: tuple[int, ...]
    type: str = "instance"

    def to_json_dict(self) -> dict:
        return _asdict_json(self)


@dataclass(frozen=True)
class InteractionAssertion:
    subject_id: int
    object_id: int
    predicate: str
    canonical_label: str
    time_span: tuple[int, int]
    evidence_frames: tuple[int, ...]
    direction: str = "subj->obj"
    confidence: float = 1.0
    type: str = "interaction"

    def to_json_dict(self) -> dict:
        return _asdict_json(self)


@dataclass(frozen=True)
class VideoAssertion:
    summary: str
    involved_ids: tuple[int, ...]
    type: str = "video"

    def to_json_dict(self) -> dict:
        return _asdict_json(self)
