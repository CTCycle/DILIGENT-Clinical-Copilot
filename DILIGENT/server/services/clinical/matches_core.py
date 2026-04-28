from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Generic, Iterable, TypeVar

import pandas as pd

from DILIGENT.server.configurations.startup import server_settings
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.domain.clinical.matching import (
    AliasCacheEntry,
    LiverToxMatch,
    MonographRecord,
)
from DILIGENT.server.services.text.normalization import (
    canonicalize_drug_query,
    coerce_text,
    normalize_drug_query_name,
    normalize_whitespace,
)
from DILIGENT.server.services.text.vocabulary import get_text_normalization_snapshot
from DILIGENT.server.services.clinical.livertox import LiverToxData
from DILIGENT.server.services.text.synonyms import (
    extract_synonym_strings,
    parse_synonym_list,
    split_synonym_variants,
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


class DrugsLookup:
    DIRECT_CONFIDENCE = server_settings.drugs_matcher.direct_confidence
    MASTER_CONFIDENCE = server_settings.drugs_matcher.master_confidence
    SYNONYM_CONFIDENCE = server_settings.drugs_matcher.synonym_confidence
    MIN_CONFIDENCE = server_settings.drugs_matcher.min_confidence
    NORMALIZATION_CACHE_LIMIT = (
        server_settings.drugs_matcher.normalization_cache_limit
    )
    MATCH_CACHE_LIMIT = server_settings.drugs_matcher.match_cache_limit
    ALIAS_CACHE_LIMIT = server_settings.drugs_matcher.alias_cache_limit
    TOKEN_MIN_LENGTH = server_settings.drugs_matcher.token_min_length
    CATALOG_EXCLUDED_TERM_SUFFIXES = server_settings.drugs_matcher.catalog_excluded_term_suffixes
    CATALOG_INDEX_LIMIT = server_settings.drugs_matcher.catalog_index_limit
    SPELLING_CONFIDENCE = server_settings.drugs_matcher.spelling_confidence
    SPELLING_MIN_QUERY_LENGTH = server_settings.drugs_matcher.spelling_min_query_length
    SPELLING_SHORT_NAME_LENGTH = server_settings.drugs_matcher.spelling_short_name_length
    SPELLING_SHORT_MAX_DISTANCE = server_settings.drugs_matcher.spelling_short_max_distance
    SPELLING_LONG_MAX_DISTANCE = server_settings.drugs_matcher.spelling_long_max_distance
    REGIMEN_SPLIT_RE = re.compile(r"(?:\s*\+\s*|\s*/\s*|\s+\bplus\b\s+)", re.IGNORECASE)
    BRAND_COMBO_PREFERENCES: dict[str, str] = {
        "bactrim": "trimethoprim sulfamethoxazole",
        "co amoxi": "amoxicillin clavulanate",
        "coamoxi": "amoxicillin clavulanate",
    }

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.data: LiverToxData | None = None
        self.match_cache: BoundedCache[str, LiverToxMatch] = BoundedCache(
            self.MATCH_CACHE_LIMIT
        )
        self.alias_cache: BoundedCache[str, AliasCacheEntry] = BoundedCache(
            self.ALIAS_CACHE_LIMIT
        )
        self.catalog_global_index: dict[str, tuple[dict[str, Any], bool, str]] = {}
        self.normalization_cache: BoundedCache[str, str] = BoundedCache(
            self.NORMALIZATION_CACHE_LIMIT
        )

    # -------------------------------------------------------------------------
    def attach_data(self, data: LiverToxData) -> None:
        self.data = data
        self.match_cache.clear()
        self.alias_cache.clear()
        self.normalization_cache.clear()
        self.catalog_global_index = {}
        self.prepare_catalog_synonyms()

    # -------------------------------------------------------------------------
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
                results.append(self.clone_cached_match(cached, raw_name, canonical_query))
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

    # -------------------------------------------------------------------------
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
        local_aliases = [alias for alias, from_catalog in alias_entries if not from_catalog]
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
            [normalized_query, *source_backed_aliases, *(alias for alias, _ in alias_entries)],
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

    # -------------------------------------------------------------------------
    def canonicalize_query(self, value: str | None) -> str:
        return canonicalize_drug_query(value)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def build_unique_keys(
        self,
        values: list[str],
        normalize_fn: Any,
    ) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = normalize_fn(value)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    # -------------------------------------------------------------------------
    def resolve_source_backed_query_variants(self, normalized_query: str) -> list[str]:
        return []

    # -------------------------------------------------------------------------
    def has_trusted_exact_key(self, normalized_key: str, data: LiverToxData) -> bool:
        return (
            normalized_key in data.primary_index
            or normalized_key in data.synonym_index
            or normalized_key in data.brand_index
            or normalized_key in data.ingredient_index
            or normalized_key in self.catalog_global_index
        )

    # -------------------------------------------------------------------------
    def match_authoritative_spelling_candidates(
        self,
        normalized_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        if len(normalized_query) < self.SPELLING_MIN_QUERY_LENGTH:
            return []
        data = self.require_data()
        candidate_keys: set[str] = set()
        for candidate, _record, _original, _is_primary in data.variant_catalog:
            candidate_keys.add(candidate)
        candidate_keys.update(self.catalog_global_index.keys())

        close_keys = [
            candidate
            for candidate in candidate_keys
            if self.is_small_spelling_difference(normalized_query, candidate)
        ]
        if not close_keys:
            return []

        stage_matches: list[tuple[MonographRecord, float, list[str]]] = []
        for key in sorted(close_keys):
            matches = self.match_normalized_all(key)
            if not matches and key in self.catalog_global_index:
                entry, _is_synonym, original = self.catalog_global_index[key]
                expansion_values = [
                    original,
                    entry.get("name"),
                    entry.get("raw_name"),
                    *entry.get("synonyms", []),
                    *entry.get("fallback_aliases", []),
                ]
                expanded_keys = self.build_unique_keys(
                    [str(value) for value in expansion_values if value],
                    self.normalize_name,
                )
                matches = self.resolve_stage_matches(expanded_keys, self.match_normalized_all)
            for record, _confidence, notes in matches:
                stage_matches.append(
                    (
                        record,
                        self.SPELLING_CONFIDENCE,
                        [
                            *notes,
                            f"corrected_query='{normalized_query}'",
                            f"matched_authoritative_key='{key}'",
                        ],
                    )
                )
        return self.dedupe_stage_matches(stage_matches)

    # -------------------------------------------------------------------------
    def is_small_spelling_difference(self, query: str, candidate: str) -> bool:
        if query == candidate:
            return False
        if not query or not candidate or query[0] != candidate[0]:
            return False
        if abs(len(query) - len(candidate)) > 2:
            return False
        query_parts = query.split()
        candidate_parts = candidate.split()
        if len(query_parts) != len(candidate_parts):
            return False
        total_distance = 0
        for query_part, candidate_part in zip(query_parts, candidate_parts, strict=True):
            if abs(len(query_part) - len(candidate_part)) > 2:
                return False
            distance_limit = max(
                self.SPELLING_SHORT_MAX_DISTANCE,
                self.SPELLING_LONG_MAX_DISTANCE,
            )
            distance = self.bounded_edit_distance(
                query_part,
                candidate_part,
                limit=distance_limit,
            )
            if distance > distance_limit:
                return False
            total_distance += distance
        allowed_distance = (
            self.SPELLING_SHORT_MAX_DISTANCE
            if max(len(query), len(candidate)) < self.SPELLING_SHORT_NAME_LENGTH
            else self.SPELLING_LONG_MAX_DISTANCE
        )
        return 0 < total_distance <= allowed_distance

    # -------------------------------------------------------------------------
    @staticmethod
    def bounded_edit_distance(left: str, right: str, *, limit: int) -> int:
        if abs(len(left) - len(right)) > limit:
            return limit + 1
        previous = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current = [left_index]
            row_min = current[0]
            for right_index, right_char in enumerate(right, start=1):
                cost = 0 if left_char == right_char else 1
                value = min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
                current.append(value)
                row_min = min(row_min, value)
            if row_min > limit:
                return limit + 1
            previous = current
        return previous[-1]

    # -------------------------------------------------------------------------
    def dedupe_stage_matches(
        self,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        merged: dict[str, tuple[MonographRecord, float, list[str]]] = {}
        for record, confidence, notes in stage_matches:
            record_key = self.record_identity_key(record)
            existing = merged.get(record_key)
            if existing is None or confidence > existing[1]:
                merged[record_key] = (record, confidence, list(dict.fromkeys(notes)))
                continue
            merged[record_key] = (
                existing[0],
                existing[1],
                list(dict.fromkeys(existing[2] + notes)),
            )
        ordered = list(merged.values())
        ordered.sort(key=lambda item: (item[0].drug_name.casefold(), item[0].nbk_id.casefold()))
        return ordered

    # -------------------------------------------------------------------------
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
        ordered.sort(key=lambda item: (item[0].drug_name.casefold(), item[0].nbk_id.casefold()))
        return ordered

    # -------------------------------------------------------------------------
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
        preferred_combo = self.preferred_combo_name(raw_name, canonical_query, normalized_query)
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

    # -------------------------------------------------------------------------
    def record_identity_key(self, record: MonographRecord) -> str:
        normalized_name = self.normalize_name(record.drug_name)
        return f"{normalized_name}|{record.nbk_id}|{record.drug_name.casefold()}"

    # -------------------------------------------------------------------------
    def rank_stage_matches(
        self,
        *,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        preferred_combo = self.preferred_combo_name(raw_name, canonical_query, normalized_query)
        ranked = sorted(
            stage_matches,
            key=lambda item: self.stage_match_score(
                item=item,
                normalized_query=normalized_query,
                preferred_combo=preferred_combo,
            ),
            reverse=True,
        )
        return ranked

    # -------------------------------------------------------------------------
    def has_strict_rank_winner(
        self,
        *,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
        normalized_query: str,
        preferred_combo: str | None,
    ) -> bool:
        if len(stage_matches) <= 1:
            return True
        top_score = self.stage_match_score(
            item=stage_matches[0],
            normalized_query=normalized_query,
            preferred_combo=preferred_combo,
        )
        next_score = self.stage_match_score(
            item=stage_matches[1],
            normalized_query=normalized_query,
            preferred_combo=preferred_combo,
        )
        return top_score > next_score

    # -------------------------------------------------------------------------
    def stage_match_score(
        self,
        *,
        item: tuple[MonographRecord, float, list[str]],
        normalized_query: str,
        preferred_combo: str | None,
    ) -> tuple[int, int, int, int, float, int]:
        record, confidence, notes = item
        normalized_record_name = self.normalize_name(record.drug_name)
        has_excerpt = int(bool(coerce_text(record.excerpt)))
        is_combo = int(len(normalized_record_name.split()) > 1)
        is_preferred_combo = int(
            preferred_combo is not None and normalized_record_name == preferred_combo
        )
        normalized_notes = [note.casefold() for note in notes]
        alias_priority = 0
        if any(note.startswith("synonym=") for note in normalized_notes):
            alias_priority = 2
        if any(note.startswith("brand=") for note in normalized_notes):
            alias_priority = 3
        if any(note.startswith("ingredient=") for note in normalized_notes):
            alias_priority = max(alias_priority, 1)
        exact_name = int(bool(normalized_query) and normalized_record_name == normalized_query)
        return (
            is_preferred_combo,
            exact_name,
            has_excerpt,
            is_combo,
            float(confidence),
            alias_priority,
        )

    # -------------------------------------------------------------------------
    def preferred_combo_name(
        self,
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
    ) -> str | None:
        normalized_raw = self.normalize_name(raw_name)
        for candidate in (normalized_raw, normalized_query, self.normalize_name(canonical_query)):
            preferred = self.BRAND_COMBO_PREFERENCES.get(candidate)
            if preferred is None:
                continue
            normalized_preferred = self.normalize_name(preferred)
            if normalized_preferred:
                return normalized_preferred
        if self.REGIMEN_SPLIT_RE.search(raw_name) and len(normalized_query.split()) > 1:
            return normalized_query
        return None

    # -------------------------------------------------------------------------
    def match_primary_all(
        self,
        canonical_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        data = self.require_data()
        matches: list[tuple[MonographRecord, float, list[str]]] = []
        for record in data.records:
            if self.canonicalize_query(record.drug_name) != canonical_query:
                continue
            matches.append((record, self.DIRECT_CONFIDENCE, []))
        return matches

    # -------------------------------------------------------------------------
    def match_alias_exact_all(
        self,
        canonical_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        data = self.require_data()
        matches: dict[str, tuple[MonographRecord, float, list[str]]] = {}

        for record in data.records:
            for synonym_original in record.synonyms.values():
                if self.canonicalize_query(synonym_original) != canonical_query:
                    continue
                matches[self.record_identity_key(record)] = (
                    record,
                    self.SYNONYM_CONFIDENCE,
                    [f"synonym='{synonym_original}'"],
                )

        alias_sources: tuple[tuple[str, dict[str, list[tuple[str, str]]]], ...] = (
            ("brand", data.brand_index),
            ("ingredient", data.ingredient_index),
        )
        for alias_type, alias_index in alias_sources:
            for entries in alias_index.values():
                for alias_value, primary_name in entries:
                    if self.canonicalize_query(alias_value) != canonical_query:
                        continue
                    primary_matches = self.match_primary_all(
                        self.canonicalize_query(primary_name)
                    )
                    for record, _, _ in primary_matches:
                        matches[self.record_identity_key(record)] = (
                            record,
                            self.MASTER_CONFIDENCE,
                            [f"{alias_type}='{alias_value}'", f"drug='{primary_name}'"],
                        )
        ordered = list(matches.values())
        ordered.sort(key=lambda item: (item[0].drug_name.casefold(), item[0].nbk_id.casefold()))
        return ordered

    # -------------------------------------------------------------------------
    def match_normalized_all(
        self,
        normalized_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        data = self.require_data()
        matches: dict[str, tuple[MonographRecord, float, list[str]]] = {}

        direct = data.primary_index.get(normalized_query, [])
        for record in direct:
            matches[self.record_identity_key(record)] = (record, self.DIRECT_CONFIDENCE, [])

        for record, original in data.synonym_index.get(normalized_query, []):
            matches[self.record_identity_key(record)] = (
                record,
                self.SYNONYM_CONFIDENCE,
                [f"synonym='{original}'"],
            )

        for alias_type, alias_index in (("brand", data.brand_index), ("ingredient", data.ingredient_index)):
            for alias_value, primary_name in alias_index.get(normalized_query, []):
                primary_matches = self.match_primary_all(self.canonicalize_query(primary_name))
                for record, _, _ in primary_matches:
                    matches[self.record_identity_key(record)] = (
                        record,
                        self.MASTER_CONFIDENCE,
                        [f"{alias_type}='{alias_value}'", f"drug='{primary_name}'"],
                    )

        ordered = list(matches.values())
        ordered.sort(key=lambda item: (item[0].drug_name.casefold(), item[0].nbk_id.casefold()))
        return ordered

    # -------------------------------------------------------------------------
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
            rejected_candidate_names=list(
                dict.fromkeys(rejected_candidate_names or [])
            ),
            record=record,
        )

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def resolve_alias_candidates(
        self, original_name: str, normalized_query: str, *, include_catalog: bool = True
    ) -> list[tuple[str, bool]]:
        alias_entries: list[tuple[str, bool]] = []
        seen: set[str] = set()
        if include_catalog:
            cache_entry = self.alias_cache.get(normalized_query, CACHE_MISS)
            if cache_entry is not CACHE_MISS:
                alias_entries = list(cache_entry.entries)
                seen = set(cache_entry.seen)
            else:
                catalog_match: tuple[dict[str, Any], bool, str] | None = None
                if normalized_query:
                    catalog_match = self.find_catalog_synonym_match(normalized_query)

                if catalog_match is not None:
                    entry, matched_is_synonym, matched_value = catalog_match
                    values_to_expand: set[str] = set()

                    if matched_is_synonym:
                        values_to_expand.update(
                            value for value in entry["synonyms"] if value
                        )
                    else:
                        if matched_value:
                            values_to_expand.add(matched_value)
                        base_name = entry.get("base_name") or entry.get("name")
                        if base_name:
                            values_to_expand.add(base_name)
                        raw_name = entry.get("raw_name")
                        if raw_name:
                            values_to_expand.add(raw_name)
                        values_to_expand.update(
                            value for value in entry["synonyms"] if value
                        )

                    for fallback_alias in entry.get("fallback_aliases", []):
                        if fallback_alias:
                            values_to_expand.add(fallback_alias)

                    for value in sorted(values_to_expand, key=str.casefold):
                        self.add_alias_entry(alias_entries, seen, value, True)
                        for variant in self.expand_variant(value):
                            self.add_alias_entry(alias_entries, seen, variant, True)

                if normalized_query:
                    self.alias_cache.put(
                        normalized_query,
                        AliasCacheEntry(list(alias_entries), set(seen)),
                    )

        self.add_alias_entry(alias_entries, seen, original_name, False)
        for variant in self.expand_variant(original_name):
            self.add_alias_entry(alias_entries, seen, variant, False)
        return alias_entries

    # -------------------------------------------------------------------------
    def add_alias_entry(
        self,
        alias_entries: list[tuple[str, bool]],
        seen: set[str],
        value: str,
        from_catalog: bool,
    ) -> None:
        normalized_value = self.normalize_name(value)
        if not normalized_value or normalized_value in seen:
            return
        seen.add(normalized_value)
        alias_entries.append((value, from_catalog))

    # -------------------------------------------------------------------------
    def find_catalog_synonym_match(
        self, normalized_query: str
    ) -> tuple[dict[str, Any], bool, str] | None:
        if not normalized_query:
            return None
        return self.catalog_global_index.get(normalized_query)

    # -------------------------------------------------------------------------
    def annotate_catalog_match(
        self,
        result: tuple[MonographRecord, float, str, list[str]],
        from_catalog: bool,
        alias_value: str,
    ) -> tuple[MonographRecord, float, str, list[str]]:
        record, confidence, reason, notes = result
        updated_notes = list(notes)
        if from_catalog:
            alias_note = coerce_text(alias_value)
            if alias_note:
                updated_notes.insert(0, f"catalog_alias='{alias_note}'")
        return record, confidence, reason, updated_notes

    # -------------------------------------------------------------------------
    def match_primary(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        data = self.require_data()
        records = data.primary_index.get(normalized_query, [])
        if not records:
            return None
        record = records[0]
        return record, self.DIRECT_CONFIDENCE, "monograph_name", []

    # -------------------------------------------------------------------------
    def match_master_list(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        data = self.require_data()
        alias_sources = (
            ("brand", data.brand_index),
            ("ingredient", data.ingredient_index),
        )
        for alias_type, index in alias_sources:
            entries = index.get(normalized_query)
            if not entries:
                continue
            for alias_value, primary_name in entries:
                resolved = self.match_primary_name(primary_name)
                if resolved is None:
                    continue
                record, base_confidence, primary_reason, primary_notes = resolved
                notes = [
                    f"{alias_type}='{alias_value}'",
                    f"drug='{primary_name}'",
                ]
                notes.extend(primary_notes)
                reason = f"{alias_type}_{primary_reason}"
                confidence = min(self.MASTER_CONFIDENCE, base_confidence)
                return record, confidence, reason, notes
        return None

    # -------------------------------------------------------------------------
    def match_synonym(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        data = self.require_data()
        aliases = data.synonym_index.get(normalized_query, [])
        if not aliases:
            return None
        record, original = aliases[0]
        notes = [f"synonym='{original}'"]
        return record, self.SYNONYM_CONFIDENCE, "synonym_match", notes

    # -------------------------------------------------------------------------
    def match_primary_name(
        self, drug_name: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        normalized_name = self.normalize_name(drug_name)
        if not normalized_name:
            return None
        direct = self.match_primary(normalized_name)
        if direct is not None:
            record, confidence, _, _ = direct
            return record, confidence, "drug_name", []
        data = self.require_data()
        aliases = data.synonym_index.get(normalized_name, [])
        if not aliases:
            return None
        record, original = aliases[0]
        notes = [f"synonym='{original}'"]
        return record, self.SYNONYM_CONFIDENCE, "drug_synonym", notes

    # -------------------------------------------------------------------------
    def prepare_catalog_synonyms(self) -> None:
        data = self.data
        self.catalog_global_index = {}
        if data is None:
            return
        catalog_source = data.drugs_catalog_df
        if catalog_source is None:
            return
        for row in data.iter_catalog_rows():
            term_type = coerce_text(getattr(row, "term_type", None))
            if not self.catalog_term_type_allowed(term_type):
                continue
            raw_name_value = coerce_text(getattr(row, "raw_name", None))
            base_name_value = coerce_text(getattr(row, "name", None))
            raw_synonyms = self.parse_catalog_synonyms(getattr(row, "synonyms", None))
            if not raw_synonyms:
                continue
            unique_synonyms: list[str] = []
            seen_synonyms: set[str] = set()
            for synonym in raw_synonyms:
                if synonym in seen_synonyms:
                    continue
                unique_synonyms.append(synonym)
                seen_synonyms.add(synonym)
            normalized_map: dict[str, str] = {}
            for synonym in unique_synonyms:
                base_normalized = self.normalize_name(synonym)
                if base_normalized and base_normalized not in normalized_map:
                    normalized_map[base_normalized] = synonym
                for variant in self.expand_variant(synonym):
                    normalized_variant = self.normalize_name(variant)
                    if not normalized_variant:
                        continue
                    if normalized_variant not in normalized_map:
                        normalized_map[normalized_variant] = synonym
            if not normalized_map:
                continue
            fallback_aliases: list[str] = []
            fallback_seen: set[str] = set()
            for alias_value in (raw_name_value, base_name_value):
                if alias_value is None:
                    continue
                if alias_value in fallback_seen:
                    continue
                fallback_aliases.append(alias_value)
                fallback_seen.add(alias_value)
            for brand in self.parse_catalog_brand_names(
                getattr(row, "brand_names", None)
            ):
                if brand in fallback_seen:
                    continue
                fallback_aliases.append(brand)
                fallback_seen.add(brand)
            entry = {
                "rxcui": getattr(row, "rxcui", ""),
                "term_type": term_type,
                "raw_name": raw_name_value,
                "name": base_name_value,
                "brand_names": fallback_aliases[:],
                "synonyms": unique_synonyms,
                "fallback_aliases": fallback_aliases,
            }
            self.register_catalog_entry(entry, normalized_map, fallback_aliases)

    # -------------------------------------------------------------------------
    def register_catalog_entry(
        self,
        entry: dict[str, Any],
        normalized_map: dict[str, str],
        fallback_aliases: list[str],
    ) -> None:
        for normalized_synonym, original in normalized_map.items():
            self.add_catalog_index_entry(normalized_synonym, entry, True, original)
        for alias in fallback_aliases:
            normalized_alias = self.normalize_name(alias)
            if not normalized_alias:
                continue
            self.add_catalog_index_entry(normalized_alias, entry, False, alias)

    # -------------------------------------------------------------------------
    def add_catalog_index_entry(
        self,
        normalized_value: str,
        entry: dict[str, Any],
        is_synonym: bool,
        original: str,
    ) -> None:
        if normalized_value in self.catalog_global_index:
            self.catalog_global_index[normalized_value] = (
                entry,
                is_synonym,
                original,
            )
            return
        if len(self.catalog_global_index) >= self.CATALOG_INDEX_LIMIT:
            return
        self.catalog_global_index[normalized_value] = (
            entry,
            is_synonym,
            original,
        )

    # -------------------------------------------------------------------------
    def catalog_term_type_allowed(self, term_type: str | None) -> bool:
        if term_type is None:
            return True
        normalized = term_type.strip().upper()
        if not normalized:
            return True
        return not normalized.endswith(self.CATALOG_EXCLUDED_TERM_SUFFIXES)

    # -------------------------------------------------------------------------
    def parse_catalog_brand_names(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            segments = split_synonym_variants(value)
        elif isinstance(value, (list, tuple, set)):
            segments = []
            for entry in value:
                segments.extend(split_synonym_variants(str(entry)))
        else:
            segments = split_synonym_variants(str(value))
        names: list[str] = []
        for segment in segments:
            text = coerce_text(segment)
            if text:
                names.append(text)
        return sorted(dict.fromkeys(names), key=str.casefold)

    # -------------------------------------------------------------------------
    def parse_catalog_synonyms(self, value: Any) -> list[str]:
        return sorted(dict.fromkeys(parse_synonym_list(value)), key=str.casefold)

    # -------------------------------------------------------------------------
    def iter_alias_variants(self, value: str) -> list[str]:
        normalized_value = normalize_whitespace(value)
        if not normalized_value:
            return []
        variants: set[str] = {normalized_value}
        for segment in re.split(r"[;,/\n]+", value):
            candidate = normalize_whitespace(segment)
            if candidate:
                variants.add(candidate)
        return sorted(variants, key=str.casefold)

    # -------------------------------------------------------------------------
    def parse_synonyms(self, value: Any) -> dict[str, str]:
        synonyms: dict[str, str] = {}
        raw_values = extract_synonym_strings(value)
        if not raw_values:
            text = coerce_text(value)
            if text is None:
                return {}
            raw_values = [text]
        for raw in raw_values:
            text = coerce_text(raw)
            if text is None:
                continue
            for candidate in split_synonym_variants(text):
                for variant in self.expand_variant(candidate):
                    normalized = self.normalize_name(variant)
                    if not normalized:
                        continue
                    if normalized in get_text_normalization_snapshot().matching_stopwords:
                        continue
                    if len(normalized) < self.TOKEN_MIN_LENGTH and " " not in normalized:
                        continue
                    if normalized not in synonyms:
                        synonyms[normalized] = variant
        return synonyms

    # -------------------------------------------------------------------------
    def expand_variant(self, value: str) -> list[str]:
        normalized = normalize_whitespace(value)
        if not normalized:
            return []
        variants = {normalized}
        for segment in re.split(r"[()\[\]]", normalized):
            candidate = segment.strip(" -")
            if candidate:
                variants.add(candidate)
        result = sorted(variants, key=str.casefold)
        return result

    # -------------------------------------------------------------------------
    def collect_tokens(self, primary: str, synonyms: list[str]) -> set[str]:
        tokens: set[str] = set()
        for source in [primary, *synonyms]:
            tokens.update(self.tokenize(source))
        return tokens

    # -------------------------------------------------------------------------
    def tokenize(self, value: str) -> set[str]:
        normalized = self.normalize_name(value)
        if not normalized:
            return set()
        return {token for token in normalized.split() if self.is_token_valid(token)}

    # -------------------------------------------------------------------------
    def is_token_valid(self, token: str) -> bool:
        if len(token) < self.TOKEN_MIN_LENGTH:
            return False
        if token in get_text_normalization_snapshot().matching_stopwords:
            return False
        return not token.isdigit()

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def normalize_name(self, name: str) -> str:
        cached = self.normalization_cache.get(name, CACHE_MISS)
        if cached is not CACHE_MISS:
            return cached
        normalized = normalize_drug_query_name(name)
        self.normalization_cache.put(name, normalized)
        return normalized

    # -------------------------------------------------------------------------
    def require_data(self) -> LiverToxData:
        if self.data is None:
            raise RuntimeError("DrugsLookup requires LiverToxData to be attached")
        return self.data


###############################################################################
class LiverToxMatcher:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        livertox_df: pd.DataFrame,
        master_list_df: pd.DataFrame | None = None,
        *,
        drugs_catalog_df: pd.DataFrame | Iterable[pd.DataFrame] | None = None,
    ) -> None:
        if isinstance(drugs_catalog_df, pd.DataFrame) and drugs_catalog_df.empty:
            catalog_df: pd.DataFrame | Iterable[pd.DataFrame] | None = None
        else:
            catalog_df = drugs_catalog_df
        self.lookup = DrugsLookup()
        self.data = LiverToxData(
            lookup=self.lookup,
            livertox_df=livertox_df,
            master_list_df=master_list_df,
            drugs_catalog_df=catalog_df,
            record_factory=MonographRecord,
        )
        self.lookup.attach_data(self.data)

    # -------------------------------------------------------------------------
    def match_drug_names(self, patient_drugs: list[str]) -> list[LiverToxMatch]:
        return self.lookup.match_drug_names(patient_drugs)

    # -------------------------------------------------------------------------
    def build_drugs_to_excerpt_mapping(
        self,
        patient_drugs: list[str],
        matches: list[LiverToxMatch],
    ) -> list[dict[str, Any]]:
        return self.data.build_drugs_to_excerpt_mapping(patient_drugs, matches)



