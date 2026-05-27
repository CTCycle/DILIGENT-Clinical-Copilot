from __future__ import annotations

from collections import OrderedDict
from typing import Any, Generic, TypeVar

from common.utils.logger import logger
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

def match_drug_names(self, patient_drugs: list[str]) -> list[LiverToxMatch]:
    results: list[LiverToxMatch] = []
    for raw_name in patient_drugs:
        canonical_query = self.canonicalize_query(raw_name)
        normalized_query = self.normalize_name(canonical_query or raw_name)
        if not normalized_query:
            results.append(
                self.create_missing_result(
                    raw_name=raw_name,
                    canonical_query=canonical_query,
                    normalized_query=normalized_query,
                    reason="invalid_query",
                    notes=["Unable to normalize query."],
                )
            )
            continue

        cached = self.match_cache.get(normalized_query, CACHE_MISS)
        if cached is not CACHE_MISS:
            results.append(
                self.clone_cached_match(cached, raw_name, canonical_query)
            )
            continue

        alias_entries = self.resolve_alias_candidates(
            raw_name,
            normalized_query,
            include_catalog=False,
        )
        match = self.match_query(
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            alias_entries=alias_entries,
        )

        if match.status == "missing":
            alias_entries = self.resolve_alias_candidates(
                raw_name,
                normalized_query,
                include_catalog=True,
            )
            match = self.match_query(
                raw_name=raw_name,
                canonical_query=canonical_query,
                normalized_query=normalized_query,
                alias_entries=alias_entries,
            )

        self.match_cache.put(normalized_query, match)
        results.append(match)
        if match.status == "matched":
            logger.info(
                "Matched '%s' to '%s' via %s (confidence=%s)",
                raw_name,
                match.matched_name,
                match.reason,
                f"{match.confidence:.2f}" if match.confidence is not None else "NA",
            )
        elif match.status == "ambiguous":
            logger.warning(
                "Ambiguous match for '%s': %s",
                raw_name,
                ", ".join(match.candidate_names),
            )
        else:
            logger.warning("No LiverTox match for '%s'", raw_name)
    return results

def match_query(
    self,
    *,
    raw_name: str,
    canonical_query: str,
    normalized_query: str,
    alias_entries: list[tuple[str, bool]],
) -> LiverToxMatch:
    if not alias_entries:
        return self.create_missing_result(
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            reason="no_alias_candidates",
            notes=["No alias candidates available."],
        )

    source_backed_aliases = self.resolve_source_backed_query_variants(
        normalized_query
    )
    local_aliases = [
        alias for alias, from_catalog in alias_entries if not from_catalog
    ]
    stage1_keys = self.build_unique_keys(
        [canonical_query, *source_backed_aliases, *local_aliases],
        self.canonicalize_query,
    )
    stage1 = self.resolve_stage_matches(stage1_keys, self.match_primary_all)
    stage1_result = self.finalize_stage_result(
        stage_name="exact_canonical",
        raw_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=normalized_query,
        stage_matches=stage1,
    )
    if stage1_result is not None:
        return stage1_result

    stage2_keys = self.build_unique_keys(
        [*source_backed_aliases, *(alias for alias, _ in alias_entries)],
        self.canonicalize_query,
    )
    stage2 = self.resolve_stage_matches(stage2_keys, self.match_alias_exact_all)
    stage2_result = self.finalize_stage_result(
        stage_name="exact_alias",
        raw_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=normalized_query,
        stage_matches=stage2,
    )
    if stage2_result is not None:
        return stage2_result

    stage3_keys = self.build_unique_keys(
        [
            normalized_query,
            *source_backed_aliases,
            *(alias for alias, _ in alias_entries),
        ],
        self.normalize_name,
    )
    stage3 = self.resolve_stage_matches(stage3_keys, self.match_normalized_all)
    stage3_result = self.finalize_stage_result(
        stage_name="normalized_exact",
        raw_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=normalized_query,
        stage_matches=stage3,
    )
    if stage3_result is not None:
        return stage3_result

    spelling = self.match_authoritative_spelling_candidates(normalized_query)
    if len(spelling) == 1:
        record, confidence, notes = spelling[0]
        return self.create_matched_result(
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            record=record,
            confidence=confidence,
            reason="spelling_correction",
            notes=notes,
        )
    if len(spelling) > 1:
        return self.create_ambiguous_result(
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            reason="ambiguous_spelling_correction",
            stage_matches=spelling,
        )

    return self.create_missing_result(
        raw_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=normalized_query,
        reason="no_match",
        notes=["No exact, alias, normalized, or unique spelling-correction match."],
    )

def clone_cached_match(
    self,
    match: LiverToxMatch,
    raw_name: str,
    canonical_query: str,
) -> LiverToxMatch:
    return LiverToxMatch(
        status=match.status,
        query_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=match.normalized_query,
        nbk_id=match.nbk_id,
        matched_name=match.matched_name,
        confidence=match.confidence,
        reason=match.reason,
        notes=list(match.notes),
        candidate_names=list(match.candidate_names),
        rejected_candidate_names=list(match.rejected_candidate_names),
        record=match.record,
    )

def resolve_stage_matches(
    self,
    keys: list[str],
    resolver: Any,
) -> list[tuple[MonographRecord, float, list[str]]]:
    merged: dict[str, tuple[MonographRecord, float, list[str]]] = {}
    for key in keys:
        for record, confidence, notes in resolver(key):
            record_key = self.record_identity_key(record)
            existing = merged.get(record_key)
            if existing is None or confidence > existing[1]:
                merged[record_key] = (
                    record,
                    confidence,
                    list(dict.fromkeys(notes)),
                )
                continue
            if existing is not None:
                combined = list(dict.fromkeys(existing[2] + notes))
                merged[record_key] = (existing[0], existing[1], combined)
    ordered = list(merged.values())
    ordered.sort(key=lambda item: self.result_sort_key(item[0], item[1]))
    return ordered

def finalize_stage_result(
    self,
    *,
    stage_name: str,
    raw_name: str,
    canonical_query: str,
    normalized_query: str,
    stage_matches: list[tuple[MonographRecord, float, list[str]]],
) -> LiverToxMatch | None:
    if not stage_matches:
        return None
    preferred_combo = self.preferred_combo_name(
        raw_name, canonical_query, normalized_query
    )
    ranked = self.rank_stage_matches(
        stage_matches=stage_matches,
        raw_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=normalized_query,
    )
    if len(ranked) == 1:
        record, confidence, notes = ranked[0]
        return self.create_matched_result(
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            record=record,
            confidence=confidence,
            reason=stage_name,
            notes=notes,
        )
    if self.has_strict_rank_winner(
        stage_matches=ranked,
        normalized_query=normalized_query,
        preferred_combo=preferred_combo,
    ):
        best_record, best_confidence, best_notes = ranked[0]
        rejected = [record.drug_name for record, _, _ in ranked[1:]]
        combined_notes = list(
            dict.fromkeys([*best_notes, "deterministic_disambiguation_applied"])
        )
        return self.create_matched_result(
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            record=best_record,
            confidence=best_confidence,
            reason=f"{stage_name}_ranked",
            notes=combined_notes,
            rejected_candidate_names=rejected,
        )
    return self.create_ambiguous_result(
        raw_name=raw_name,
        canonical_query=canonical_query,
        normalized_query=normalized_query,
        reason=f"ambiguous_{stage_name}",
        stage_matches=ranked,
    )

def diagnose_missing_drug(self, drug_name: str) -> dict[str, Any]:
    normalized = self.normalize_name(drug_name)
    data = self.require_data()
    diagnosis = {
        "original_name": drug_name,
        "normalized_name": normalized,
        "in_primary_index": normalized in data.primary_index,
        "in_synonym_index": normalized in data.synonym_index,
        "in_catalog_index": normalized in self.catalog_global_index,
        "catalog_entries": [],
        "alias_candidates": [],
        "token_matches": [],
    }
    if normalized in self.catalog_global_index:
        entry, is_synonym, original = self.catalog_global_index[normalized]
        diagnosis["catalog_entries"].append(
            {
                "is_synonym": is_synonym,
                "original": original,
                "base_name": entry.get("base_name"),
            }
        )
    alias_entries = self.resolve_alias_candidates(drug_name, normalized)
    diagnosis["alias_candidates"] = [
        {"alias": alias, "from_catalog": from_catalog}
        for alias, from_catalog in alias_entries[:10]
    ]
    for token in self.tokenize(normalized):
        if token in data.token_index:
            diagnosis["token_matches"].append(
                {"token": token, "record_count": len(data.token_index[token])}
            )
    return diagnosis
