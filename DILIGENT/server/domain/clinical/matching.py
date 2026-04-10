from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class AliasCacheEntry:
    entries: list[tuple[str, bool]]
    seen: set[str]


@dataclass(slots=True)
class MonographRecord:
    nbk_id: str
    drug_name: str
    normalized_name: str
    excerpt: str | None
    synonyms: dict[str, str]
    tokens: set[str]


@dataclass(slots=True)
class LiverToxMatch:
    status: Literal["matched", "missing", "ambiguous"]
    query_name: str
    canonical_query: str
    normalized_query: str
    nbk_id: str | None
    matched_name: str | None
    confidence: float | None
    reason: str
    notes: list[str]
    candidate_names: list[str]
    rejected_candidate_names: list[str]
    record: MonographRecord | None = None
