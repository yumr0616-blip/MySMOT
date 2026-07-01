"""Open-vocabulary predicate -> canonical label mapping.

Deterministic, per §4's Output Assembler responsibility. Falls back to the
raw (lowercased, stripped) predicate when unmapped; eval (§7) depends on
canonical labels being stable, so this fallback is intentional and explicit
rather than raising.
"""
from __future__ import annotations

CANONICAL_MAP: dict[str, str] = {
    "approaches": "approach",
    "is approaching": "approach",
    "moves toward": "approach",
    "moves away from": "recede",
    "recedes from": "recede",
    "touches": "contact",
    "is in contact with": "contact",
    "follows": "follow",
    "is present and moving": "present",
}


def map_predicate(raw_predicate: str) -> str:
    normalized = raw_predicate.strip().lower()
    return CANONICAL_MAP.get(normalized, normalized)
