from __future__ import annotations

from collections import OrderedDict
from typing import Any, Generic, TypeVar

from domain.clinical.matching import (
    LiverToxMatch,
    MonographRecord,
)

KT = TypeVar("KT")
VT = TypeVar("VT")
CACHE_MISS = object()


###############################################################################
class BoundedCache(Generic[KT, VT]):
    __slots__ = ("limit", "store")

    def __init__(self, limit: int) -> None:
        self.limit = max(int(limit), 1)
        self.store: OrderedDict[KT, VT] = OrderedDict()

    # ------------------------------------------------------------------
    def get(self, key: KT, default: Any = CACHE_MISS) -> Any:
        if key not in self.store:
            return default
        value = self.store.pop(key)
        self.store[key] = value
        return value

    # ------------------------------------------------------------------
    def put(self, key: KT, value: VT) -> None:
        if self.limit <= 0:
            return
        if key in self.store:
            self.store.pop(key)
        elif len(self.store) >= self.limit:
            self.store.popitem(last=False)
        self.store[key] = value

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.store.clear()


# Extracted from the facade module; functions intentionally accept the facade instance.


def create_matched_result(
    self,
    *,
    raw_name: str,
    canonical_query: str,
    normalized_query: str,
    record: MonographRecord,
    confidence: float,
    reason: str,
    notes: list[str],
    rejected_candidate_names: list[str] | None = None,
) -> LiverToxMatch:
    normalized_confidence = round(min(max(confidence, self.MIN_CONFIDENCE), 1.0), 2)
    cleaned_notes = list(dict.fromkeys(note for note in notes if note))
    return LiverToxMatch(
        status="matched",
        query_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=normalized_query,
        nbk_id=record.nbk_id,
        matched_name=record.drug_name,
        confidence=normalized_confidence,
        reason=reason,
        notes=cleaned_notes,
        candidate_names=[record.drug_name],
        rejected_candidate_names=list(dict.fromkeys(rejected_candidate_names or [])),
        record=record,
    )


def create_missing_result(
    self,
    *,
    raw_name: str,
    canonical_query: str,
    normalized_query: str,
    reason: str,
    notes: list[str],
) -> LiverToxMatch:
    return LiverToxMatch(
        status="missing",
        query_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=normalized_query,
        nbk_id=None,
        matched_name=None,
        confidence=None,
        reason=reason,
        notes=list(dict.fromkeys(note for note in notes if note)),
        candidate_names=[],
        rejected_candidate_names=[],
        record=None,
    )


def create_ambiguous_result(
    self,
    *,
    raw_name: str,
    canonical_query: str,
    normalized_query: str,
    reason: str,
    stage_matches: list[tuple[MonographRecord, float, list[str]]],
) -> LiverToxMatch:
    candidate_names = sorted(
        dict.fromkeys(record.drug_name for record, _, _ in stage_matches),
        key=str.casefold,
    )
    notes: list[str] = []
    for _, _, entry_notes in stage_matches:
        notes.extend(entry_notes)
    return LiverToxMatch(
        status="ambiguous",
        query_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=normalized_query,
        nbk_id=None,
        matched_name=None,
        confidence=None,
        reason=reason,
        notes=list(dict.fromkeys(note for note in notes if note)),
        candidate_names=candidate_names,
        rejected_candidate_names=[],
        record=None,
    )


def result_sort_key(
    self,
    record: MonographRecord,
    confidence: float,
) -> tuple[float, str, str, str]:
    return (
        -float(confidence),
        record.drug_name.casefold(),
        record.monograph_key or "",
        record.stable_key,
    )


def create_match(
    self,
    record: MonographRecord,
    confidence: float,
    reason: str,
    notes: list[str] | None,
) -> LiverToxMatch:
    return self.create_matched_result(
        raw_name=record.drug_name,
        canonical_query=self.canonicalize_query(record.drug_name),
        normalized_query=self.normalize_name(record.drug_name),
        record=record,
        confidence=confidence,
        reason=reason,
        notes=list(notes or []),
    )
