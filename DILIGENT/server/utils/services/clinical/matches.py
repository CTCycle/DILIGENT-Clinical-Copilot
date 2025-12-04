from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, TypeVar, Generic

import pandas as pd
from rapidfuzz import fuzz

from DILIGENT.server.utils.configurations import server_settings
from DILIGENT.server.utils.constants import MATCHING_STOPWORDS
from DILIGENT.server.utils.logger import logger
from DILIGENT.server.utils.services.text.normalization import (
    coerce_text,
    normalize_drug_name,
    normalize_whitespace,
)
from DILIGENT.server.utils.services.clinical.livertox import LiverToxData
from DILIGENT.server.utils.services.text.synonyms import (
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


###############################################################################
@dataclass(slots=True)
class AliasCacheEntry:
    entries: list[tuple[str, bool]]
    seen: set[str]


###############################################################################
@dataclass(slots=True)
class MonographRecord:
    nbk_id: str
    drug_name: str
    normalized_name: str
    excerpt: str | None
    synonyms: dict[str, str]
    tokens: set[str]


###############################################################################
@dataclass(slots=True)
class LiverToxMatch:
    nbk_id: str
    matched_name: str
    confidence: float
    reason: str
    notes: list[str]
    record: MonographRecord | None = None


###############################################################################
class DrugsLookup:
    DIRECT_CONFIDENCE = server_settings.drugs_matcher.direct_confidence
    MASTER_CONFIDENCE = server_settings.drugs_matcher.master_confidence
    SYNONYM_CONFIDENCE = server_settings.drugs_matcher.synonym_confidence
    PARTIAL_CONFIDENCE = server_settings.drugs_matcher.partial_confidence
    FUZZY_CONFIDENCE = server_settings.drugs_matcher.fuzzy_confidence
    FUZZY_THRESHOLD = server_settings.drugs_matcher.fuzzy_threshold
    TOKEN_MAX_FREQUENCY = server_settings.drugs_matcher.token_max_frequency
    TOKEN_MIN_LENGTH = server_settings.drugs_matcher.token_min_length
    MIN_CONFIDENCE = server_settings.drugs_matcher.min_confidence
    FUZZY_EARLY_EXIT_RATIO = server_settings.drugs_matcher.fuzzy_early_exit_ratio
    NORMALIZATION_CACHE_LIMIT = (
        server_settings.drugs_matcher.normalization_cache_limit
    )
    VARIANT_CACHE_LIMIT = server_settings.drugs_matcher.variant_cache_limit
    MATCH_CACHE_LIMIT = server_settings.drugs_matcher.match_cache_limit
    ALIAS_CACHE_LIMIT = server_settings.drugs_matcher.alias_cache_limit
    CATALOG_EXCLUDED_TERM_SUFFIXES = server_settings.drugs_matcher.catalog_excluded_term_suffixes
    CATALOG_TOKEN_RATIO_THRESHOLD = (
        server_settings.drugs_matcher.catalog_token_ratio_threshold
    )
    CATALOG_OVERALL_RATIO_THRESHOLD = (
        server_settings.drugs_matcher.catalog_overall_ratio_threshold
    )
    CATALOG_INDEX_LIMIT = server_settings.drugs_matcher.catalog_index_limit
    CATALOG_CANDIDATE_LIMIT = server_settings.drugs_matcher.catalog_candidate_limit

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.data: LiverToxData | None = None
        self.match_cache: BoundedCache[str, LiverToxMatch | None] = BoundedCache(
            self.MATCH_CACHE_LIMIT
        )
        self.alias_cache: BoundedCache[str, AliasCacheEntry] = BoundedCache(
            self.ALIAS_CACHE_LIMIT
        )
        self.catalog_global_index: dict[str, tuple[dict[str, Any], bool, str]] = {}
        self.catalog_token_index: dict[str, set[str]] = {}
        self.normalization_cache: BoundedCache[str, str] = BoundedCache(
            self.NORMALIZATION_CACHE_LIMIT
        )
        self.variant_cache: BoundedCache[str, list[str]] = BoundedCache(
            self.VARIANT_CACHE_LIMIT
        )

    # -------------------------------------------------------------------------
    def attach_data(self, data: LiverToxData) -> None:
        self.data = data
        self.match_cache.clear()
        self.alias_cache.clear()
        self.normalization_cache.clear()
        self.variant_cache.clear()
        self.catalog_global_index = {}
        self.catalog_token_index = {}
        self.prepare_catalog_synonyms()

    # -------------------------------------------------------------------------
    def match_drug_names(self, patient_drugs: list[str]) -> list[LiverToxMatch | None]:
        total = len(patient_drugs)
        results: list[LiverToxMatch | None] = [None] * total
        for idx, name in enumerate(patient_drugs):
            normalized = self.normalize_name(name)
            if not normalized:
                continue
            logger.info("Finding matches for drug '%s'", name)
            cached = self.match_cache.get(normalized, CACHE_MISS)
            if cached is not CACHE_MISS:
                if cached is not None:
                    logger.info(
                        "Cache hit for '%s': '%s' via %s (confidence=%.2f)",
                        name,
                        cached.matched_name,
                        cached.reason,
                        cached.confidence,
                    )
                else:
                    logger.info("Cache recorded no match for '%s'", name)
                results[idx] = cached
                continue
            alias_start = time.perf_counter()
            alias_entries = self.resolve_alias_candidates(patient_drugs[idx], normalized)
            alias_elapsed_s = time.perf_counter() - alias_start
            logger.info(
                "Fetched %d candidate names for '%s' in %.3f s",
                len(alias_entries),
                name,
                alias_elapsed_s,
            )
            if not alias_entries:
                self.match_cache.put(normalized, None)
                logger.warning(
                    "No alias candidates found for '%s' (normalized: '%s'). "
                    "Check catalog availability and term type filters.",
                    name,
                    normalized,
                )
                continue
            match_start = time.perf_counter()
            lookup = self.match_query(alias_entries)
            match_elapsed_s = time.perf_counter() - match_start
            if lookup is None:
                self.match_cache.put(normalized, None)
                logger.warning(
                    "No match found for '%s' after %d aliases in %.3f s. "
                    "Checks: primary=checked, synonym=checked, master=checked, "
                    "partial=checked, fuzzy=checked",
                    name,
                    len(alias_entries),
                    match_elapsed_s,
                )
                continue
            record, confidence, reason, notes = lookup
            match = self.create_match(record, confidence, reason, notes)
            self.match_cache.put(normalized, match)
            results[idx] = match
            summary_notes = "; ".join(match.notes) if match.notes else ""
            logger.info(
                "Best candidate for '%s': '%s' via %s (confidence=%.2f)%s in %.3f s",
                name,
                match.matched_name,
                match.reason,
                match.confidence,
                f" [{summary_notes}]" if summary_notes else "",
                match_elapsed_s,
            )
        return results

    # -------------------------------------------------------------------------
    def match_query(
        self, alias_entries: list[tuple[str, bool]]
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        if not alias_entries:
            return None
        candidates: list[tuple[str, str, bool]] = []
        seen: set[str] = set()
        for alias_value, from_catalog in alias_entries:
            normalized_value = self.normalize_name(alias_value)
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            candidates.append((normalized_value, alias_value, from_catalog))
        if not candidates:
            return None

        for normalized_value, alias_value, from_catalog in candidates:
            direct = self.match_primary(normalized_value)
            if direct is not None:
                return self.annotate_catalog_match(direct, from_catalog, alias_value)

        synonym_matches: list[
            tuple[tuple[MonographRecord, float, str, list[str]], bool, str]
        ] = [
            (synonym, from_catalog, alias_value)
            for normalized_value, alias_value, from_catalog in candidates
            if (synonym := self.match_synonym(normalized_value)) is not None
        ]

        master_matches: list[
            tuple[tuple[MonographRecord, float, str, list[str]], bool, str]
        ] = [
            (master, from_catalog, alias_value)
            for normalized_value, alias_value, from_catalog in candidates
            if (master := self.match_master_list(normalized_value)) is not None
        ]

        for master_match in master_matches:
            master_record, _, _, _ = master_match[0]
            for synonym_match in synonym_matches:
                synonym_record, _, _, _ = synonym_match[0]
                if synonym_record.nbk_id == master_record.nbk_id:
                    return self.annotate_catalog_match(
                        master_match[0], master_match[1], master_match[2]
                    )

        if synonym_matches:
            match_result = synonym_matches[0]
            return self.annotate_catalog_match(
                match_result[0], match_result[1], match_result[2]
            )

        if master_matches:
            match_result = master_matches[0]
            return self.annotate_catalog_match(
                match_result[0], match_result[1], match_result[2]
            )

        for normalized_value, alias_value, from_catalog in candidates:
            partial = self.match_partial(normalized_value)
            if partial is not None:
                return self.annotate_catalog_match(partial, from_catalog, alias_value)

        for normalized_value, alias_value, from_catalog in candidates:
            fuzzy = self.match_fuzzy(normalized_value)
            if fuzzy is not None:
                return self.annotate_catalog_match(fuzzy, from_catalog, alias_value)

        return None

    # -------------------------------------------------------------------------
    def resolve_alias_candidates(
        self, original_name: str, normalized_query: str
    ) -> list[tuple[str, bool]]:
        alias_entries: list[tuple[str, bool]] = []
        seen: set[str] = set()
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
                    base_name = entry.get("base_name")
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

                for value in values_to_expand:
                    self.add_alias_entry(alias_entries, seen, value, True)
                    for variant in self.expand_variant(value):
                        self.add_alias_entry(alias_entries, seen, variant, True)

            if normalized_query:
                self.alias_cache.put(
                    normalized_query,
                    AliasCacheEntry(list(alias_entries), set(seen)),
                )

        self.add_alias_entry(alias_entries, seen, original_name, False)
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
        direct_match = self.catalog_global_index.get(normalized_query)
        if direct_match is not None:
            return direct_match

        if not self.catalog_global_index:
            return None

        significant_query_tokens = self.catalog_significant_tokens(normalized_query)
        candidate_keys: set[str] = set()
        if significant_query_tokens:
            for token in significant_query_tokens:
                matches = self.catalog_token_index.get(token)
                if matches:
                    candidate_keys.update(matches)
                if len(candidate_keys) >= self.CATALOG_CANDIDATE_LIMIT:
                    break
        if not candidate_keys:
            candidate_keys = set(self.catalog_global_index.keys())
        candidate_list = sorted(candidate_keys)
        if len(candidate_list) > self.CATALOG_CANDIDATE_LIMIT:
            candidate_list = candidate_list[: self.CATALOG_CANDIDATE_LIMIT]

        best_candidate: tuple[
            tuple[int, int, float, float, float, int], dict[str, Any], bool, str
        ] | None = None
        for candidate_normalized in candidate_list:
            payload = self.catalog_global_index.get(candidate_normalized)
            if payload is None:
                continue
            entry, is_synonym, original = payload
            accepted, score = self.evaluate_catalog_candidate(
                normalized_query,
                candidate_normalized,
                significant_query_tokens,
            )
            if not accepted:
                continue
            candidate = (score, entry, is_synonym, original)
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate[1], best_candidate[2], best_candidate[3]

        return None

    # -------------------------------------------------------------------------
    def catalog_significant_tokens(self, value: str) -> list[str]:
        tokens = value.split()
        return [
            token
            for token in tokens
            if len(token) >= self.TOKEN_MIN_LENGTH and token not in MATCHING_STOPWORDS
        ]

    # -------------------------------------------------------------------------
    def evaluate_catalog_candidate(
        self,
        normalized_query: str,
        candidate: str,
        significant_query_tokens: list[str],
    ) -> tuple[bool, tuple[int, int, float, float, float, int]]:
        candidate_tokens = self.catalog_significant_tokens(candidate)
        shared_tokens = set(significant_query_tokens) & set(candidate_tokens)
        token_ratio = fuzz.token_set_ratio(normalized_query, candidate) / 100.0
        partial_ratio = fuzz.partial_ratio(normalized_query, candidate) / 100.0
        overall_ratio = fuzz.ratio(normalized_query, candidate) / 100.0
        substring_length = 0
        if candidate in normalized_query:
            substring_length = len(candidate)
        elif normalized_query in candidate:
            substring_length = len(normalized_query)
        accepted = bool(shared_tokens)
        if not accepted:
            accepted = bool(
                token_ratio >= self.CATALOG_TOKEN_RATIO_THRESHOLD
                and partial_ratio >= self.CATALOG_OVERALL_RATIO_THRESHOLD
            )
        score = (
            len(shared_tokens),
            substring_length,
            token_ratio,
            partial_ratio,
            overall_ratio,
            -abs(len(candidate) - len(normalized_query)),
        )
        return accepted, score

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
        record = data.primary_index.get(normalized_query)
        if record is None:
            return None
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
        alias = data.synonym_index.get(normalized_query)
        if alias is None:
            return None
        record, original = alias
        notes = [f"synonym='{original}'"]
        return record, self.SYNONYM_CONFIDENCE, "synonym_match", notes

    # -------------------------------------------------------------------------
    def match_partial(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        data = self.require_data()
        tokens = [
            token for token in normalized_query.split() if self.is_token_valid(token)
        ]
        if not tokens:
            return None
        candidate_scores: dict[str, int] = {}
        record_lookup: dict[str, MonographRecord] = {}
        matched_tokens: dict[str, set[str]] = {}
        for token in tokens:
            for record in data.token_index.get(token, []):
                key = record.normalized_name or record.drug_name.lower()
                record_lookup[key] = record
                bucket = matched_tokens.setdefault(key, set())
                bucket.add(token)
                candidate_scores[key] = len(bucket)
        if not candidate_scores:
            return None
        best_score = max(candidate_scores.values())
        tied = [key for key, score in candidate_scores.items() if score == best_score]
        if len(tied) != 1:
            best_record = self.select_best_partial_record(
                normalized_query,
                tied,
                record_lookup,
                matched_tokens,
            )
            if best_record is None:
                return None
            best_key = best_record.normalized_name or best_record.drug_name.lower()
        else:
            best_key = tied[0]
            best_record = record_lookup[best_key]
        note_tokens = sorted(matched_tokens.get(best_key, set()))
        notes = [f"token='{token}'" for token in note_tokens]
        return best_record, self.PARTIAL_CONFIDENCE, "partial_synonym", notes

    # -------------------------------------------------------------------------
    def select_best_partial_record(
        self,
        normalized_query: str,
        candidates: list[str],
        record_lookup: dict[str, MonographRecord],
        matched_tokens: dict[str, set[str]],
    ) -> MonographRecord | None:
        best_record: MonographRecord | None = None
        best_score: tuple[int, float, int] | None = None
        for key in candidates:
            record = record_lookup.get(key)
            if record is None:
                continue
            normalized_target = record.normalized_name or self.normalize_name(
                record.drug_name
            )
            token_overlap = len(matched_tokens.get(key, set()))
            ratio = (
                fuzz.ratio(normalized_query, normalized_target) / 100.0
                if normalized_target
                else 0.0
            )
            length_bias = (
                -abs(len(normalized_target) - len(normalized_query))
                if normalized_target
                else -len(normalized_query)
            )
            score = (token_overlap, ratio, length_bias)
            if best_score is None or score > best_score:
                best_score = score
                best_record = record
        return best_record

    # -------------------------------------------------------------------------
    def match_fuzzy(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        if len(normalized_query) < self.TOKEN_MIN_LENGTH:
            return None
        variant = self.find_best_variant(normalized_query)
        if variant is None:
            return None
        record, original, is_primary, score = variant
        reason = "fuzzy_primary" if is_primary else "fuzzy_synonym"
        notes: list[str] = [f"score={score:.2f}"]
        if not is_primary:
            notes.insert(0, f"variant='{original}'")
        return record, max(self.FUZZY_CONFIDENCE, score), reason, notes

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
        alias = data.synonym_index.get(normalized_name)
        if alias is None:
            return None
        record, original = alias
        notes = [f"synonym='{original}'"]
        return record, self.SYNONYM_CONFIDENCE, "drug_synonym", notes

    # -------------------------------------------------------------------------
    def find_best_variant(
        self, normalized_query: str
    ) -> tuple[MonographRecord, str, bool, float] | None:
        data = self.require_data()
        best: tuple[MonographRecord, str, bool, float] | None = None
        best_ratio = 0.0
        for candidate, record, original, is_primary in data.variant_catalog:
            if candidate == normalized_query:
                return record, original, is_primary, 1.0
            ratio = fuzz.ratio(normalized_query, candidate) / 100.0
            if ratio > best_ratio:
                best_ratio = ratio
                best = (record, original, is_primary, ratio)
                if best_ratio >= self.FUZZY_EARLY_EXIT_RATIO:
                    break
        if best is None or best_ratio < self.FUZZY_THRESHOLD:
            return None
        return best

    # -------------------------------------------------------------------------
    def prepare_catalog_synonyms(self) -> None:
        data = self.data
        self.catalog_global_index = {}
        self.catalog_token_index = {}
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
            self._register_catalog_tokens(normalized_value)
            return
        if len(self.catalog_global_index) >= self.CATALOG_INDEX_LIMIT:
            return
        self.catalog_global_index[normalized_value] = (
            entry,
            is_synonym,
            original,
        )
        self._register_catalog_tokens(normalized_value)

    # -------------------------------------------------------------------------
    def _register_catalog_tokens(self, normalized_value: str) -> None:
        tokens = self.catalog_significant_tokens(normalized_value)
        if not tokens:
            return
        for token in tokens:
            bucket = self.catalog_token_index.setdefault(token, set())
            bucket.add(normalized_value)
            if len(bucket) > self.CATALOG_CANDIDATE_LIMIT * 2:
                while len(bucket) > self.CATALOG_CANDIDATE_LIMIT:
                    try:
                        bucket.pop()
                    except KeyError:  # pragma: no cover - defensive
                        break

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
        return names

    # -------------------------------------------------------------------------
    def parse_catalog_synonyms(self, value: Any) -> list[str]:
        return parse_synonym_list(value)

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
        return list(variants)

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
                    if normalized in MATCHING_STOPWORDS:
                        continue
                    if len(normalized) < self.TOKEN_MIN_LENGTH and " " not in normalized:
                        continue
                    if normalized not in synonyms:
                        synonyms[normalized] = variant
        return synonyms

    # -------------------------------------------------------------------------
    def expand_variant(self, value: str) -> list[str]:
        cached = self.variant_cache.get(value, CACHE_MISS)
        if cached is not CACHE_MISS:
            return list(cached)
        normalized = normalize_whitespace(value)
        if not normalized:
            return []
        variants = {normalized}
        for segment in re.split(r"[()\[\]]", normalized):
            candidate = segment.strip(" -")
            if candidate:
                variants.add(candidate)
        result = list(variants)
        self.variant_cache.put(value, result)
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
        if token in MATCHING_STOPWORDS:
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
        normalized_confidence = round(min(max(confidence, self.MIN_CONFIDENCE), 1.0), 2)
        cleaned_notes = list(dict.fromkeys(note for note in (notes or []) if note))
        return LiverToxMatch(
            nbk_id=record.nbk_id,
            matched_name=record.drug_name,
            confidence=normalized_confidence,
            reason=reason,
            notes=cleaned_notes,
            record=record,
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
        normalized = normalize_drug_name(name)
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
    def match_drug_names(self, patient_drugs: list[str]) -> list[LiverToxMatch | None]:
        return self.lookup.match_drug_names(patient_drugs)

    # -------------------------------------------------------------------------
    def build_drugs_to_excerpt_mapping(
        self,
        patient_drugs: list[str],
        matches: list[LiverToxMatch | None],
    ) -> list[dict[str, Any]]:
        return self.data.build_drugs_to_excerpt_mapping(patient_drugs, matches)
