from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from DILIGENT.app.constants import MATCHING_STOPWORDS as BASE_MATCHING_STOPWORDS
from DILIGENT.app.utils.updater.livertox import LiverToxUpdater

__all__ = [
    "LiverToxUpdater",
    "MonographRecord",
    "LiverToxMatch",
    "LiverToxMatcher",
]


MATCHING_STOPWORDS = BASE_MATCHING_STOPWORDS | {
    "combo",
    "combination",
    "of",
    "or",
    "patch",
    "po",
}


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
class LiverToxMatcher:
    DIRECT_CONFIDENCE = 1.0
    MASTER_CONFIDENCE = 0.92
    SYNONYM_CONFIDENCE = 0.90
    PARTIAL_CONFIDENCE = 0.86
    FUZZY_CONFIDENCE = 0.84
    FUZZY_THRESHOLD = 0.85
    TOKEN_MAX_FREQUENCY = 3
    MIN_CONFIDENCE = 0.40

    # -------------------------------------------------------------------------
    def __init__(
        self,
        livertox_df: pd.DataFrame,
        master_list_df: pd.DataFrame | None = None,
        *,
        drugs_catalog_df: pd.DataFrame | None = None,
    ) -> None:
        # Build lookup tables from the LiverTox dataset so queries can be
        # resolved quickly without repeatedly walking the source data.
        # Persist the raw dataframes so the matcher can reuse them for
        # additional lookups (for example, preparing response payloads).
        self.livertox_df = livertox_df
        if master_list_df is not None and not master_list_df.empty:
            self.master_list_df = master_list_df.copy()
        else:
            self.master_list_df = self.derive_master_alias_source(livertox_df)
        if drugs_catalog_df is not None and not drugs_catalog_df.empty:
            self.drugs_catalog_df = drugs_catalog_df.copy()
        else:
            self.drugs_catalog_df = None
        # The matcher relies on several derived indexes; populate them once
        # during initialization so later queries remain efficient.
        self.match_cache: dict[str, LiverToxMatch | None] = {}
        self.alias_cache: dict[str, tuple[list[tuple[str, bool]], set[str]]] = {}
        self.records: list[MonographRecord] = []
        self.primary_index: dict[str, MonographRecord] = {}
        self.synonym_index: dict[str, tuple[MonographRecord, str]] = {}
        self.variant_catalog: list[tuple[str, MonographRecord, str, bool]] = []
        self.token_occurrences: dict[str, list[MonographRecord]] = {}
        self.token_index: dict[str, list[MonographRecord]] = {}
        self.brand_index: dict[str, list[tuple[str, str]]] = {}
        self.ingredient_index: dict[str, list[tuple[str, str]]] = {}
        self.rows_by_name: dict[str, dict[str, Any]] = {}
        self.catalog_synonym_records: list[dict[str, Any]] = []
        self.build_records()
        self.build_master_list_aliases()
        self.prepare_catalog_synonyms()
        self.finalize_token_index()

    # -------------------------------------------------------------------------
    def match_drug_names(self, patient_drugs: list[str]) -> list[LiverToxMatch | None]:
        # Normalize and resolve each patient supplied drug using cached results
        # whenever possible to reduce redundant lookups.
        total = len(patient_drugs)
        results: list[LiverToxMatch | None] = [None] * total
        for idx, name in enumerate(patient_drugs):
            normalized = self.normalize_name(name)
            if not normalized:
                continue
            cached = self.match_cache.get(normalized)
            if cached is not None or normalized in self.match_cache:
                results[idx] = cached
                continue
            alias_entries = self.resolve_alias_candidates(
                patient_drugs[idx], normalized
            )
            if not alias_entries:
                self.match_cache[normalized] = None
                continue
            lookup = self.match_query(alias_entries)
            if lookup is None:
                self.match_cache[normalized] = None
                continue
            record, confidence, reason, notes = lookup
            match = self.create_match(record, confidence, reason, notes)
            self.match_cache[normalized] = match
            results[idx] = match
        return results

    # -------------------------------------------------------------------------
    def build_drugs_to_excerpt_mapping(
        self,
        patient_drugs: list[str],
        matches: list[LiverToxMatch | None],
    ) -> list[dict[str, Any]]:
        # Combine patient drug names with matched monographs and excerpts to
        # create a payload suitable for downstream presentation.
        entries: list[dict[str, Any]] = []
        row_index = self.ensure_row_index()
        for original, match in zip(patient_drugs, matches, strict=False):
            row_data: dict[str, Any] | None = None
            excerpts: list[str] = []
            if match is not None:
                normalized_key: str | None = None
                if match.record is not None:
                    normalized_key = (
                        match.record.normalized_name
                        or self.normalize_name(match.record.drug_name)
                    )
                if not normalized_key:
                    normalized_key = self.normalize_name(match.matched_name)
                if normalized_key:
                    row = row_index.get(normalized_key)
                    if row:
                        row_data = dict(row)
                excerpt_value = row_data.get("excerpt") if row_data else None
                if match.record and match.record.excerpt:
                    excerpts.append(match.record.excerpt)
                if isinstance(excerpt_value, str) and excerpt_value:
                    excerpts.append(excerpt_value)
                unique_excerpts = list(dict.fromkeys(excerpts))

            else:
                row_data = None
                unique_excerpts = []

            entries.append(
                {
                    "drug_name": original,
                    "matched_livertox_row": row_data,
                    "extracted_excerpts": unique_excerpts,
                }
            )

        return entries

    # -------------------------------------------------------------------------
    def match_query(
        self, alias_entries: list[tuple[str, bool]]
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        # Execute the layered matching strategy using progressively fuzzier
        # heuristics until a suitable monograph record is found.
        if not alias_entries:
            return None
        candidates: list[tuple[str, str, bool]] = []
        seen: set[str] = set()
        # Normalize each alias once so we can reuse the lowered value throughout
        # the layered matching pipeline without reprocessing strings.
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
        # Enumerate the most promising aliases for a patient provided name by
        # blending catalog entries with the raw patient input.
        alias_entries: list[tuple[str, bool]] = []
        seen: set[str] = set()
        cache_entry = self.alias_cache.get(normalized_query)
        if cache_entry is not None:
            cached_entries, cached_seen = cache_entry
            alias_entries = list(cached_entries)
            seen = set(cached_seen)
        else:
            alias_entries = []
            seen = set()

        catalog_match: tuple[dict[str, Any], bool, str] | None = None
        if normalized_query:
            catalog_match = self.find_catalog_synonym_match(normalized_query)

        if catalog_match is not None:
            entry, matched_is_synonym, matched_value = catalog_match
            # Reorder the synonym list so the value that triggered the match is
            # evaluated first, retaining catalog intent.
            prioritized_synonyms: list[str] = []
            if matched_is_synonym:
                prioritized_synonyms.append(matched_value)
            prioritized_synonyms.extend(
                synonym
                for synonym in entry["synonyms"]
                if synonym not in prioritized_synonyms
            )
            for synonym in prioritized_synonyms:
                self.add_alias_entry(alias_entries, seen, synonym, True)
                for variant in self.expand_variant(synonym):
                    self.add_alias_entry(alias_entries, seen, variant, True)
            if not matched_is_synonym:
                self.add_alias_entry(alias_entries, seen, matched_value, True)
                for variant in self.expand_variant(matched_value):
                    self.add_alias_entry(alias_entries, seen, variant, True)

        if normalized_query and normalized_query not in self.alias_cache:
            self.alias_cache[normalized_query] = (
                list(alias_entries),
                set(seen),
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
        # Insert a new alias candidate while preventing duplicate normalized
        # values from reaching later stages.
        normalized_value = self.normalize_name(value)
        if not normalized_value or normalized_value in seen:
            return
        seen.add(normalized_value)
        alias_entries.append((value, from_catalog))

    # -------------------------------------------------------------------------
    def find_catalog_synonym_match(
        self, normalized_query: str
    ) -> tuple[dict[str, Any], bool, str] | None:
        # Look up catalog provided synonyms using exact, partial, and fuzzy
        # comparisons to seed the alias expansion phase.
        if not self.catalog_synonym_records:
            return None

        significant_query_tokens = self.catalog_significant_tokens(normalized_query)
        best_synonym: tuple[
            tuple[int, int, float, float, int],
            dict[str, Any],
            str,
        ] | None = None
        # First, inspect curated catalog synonyms since they represent the most
        # trustworthy aliases available.
        for entry in self.catalog_synonym_records:
            normalized_map: dict[str, str] = entry["normalized_map"]
            matched = normalized_map.get(normalized_query)
            if matched:
                return entry, True, matched
            for normalized_synonym, original in normalized_map.items():
                accepted, score = self.evaluate_catalog_candidate(
                    normalized_query,
                    normalized_synonym,
                    significant_query_tokens,
                )
                if not accepted:
                    continue
                candidate = (score, entry, original)
                if best_synonym is None or candidate[0] > best_synonym[0]:
                    best_synonym = candidate

        if best_synonym is not None:
            return best_synonym[1], True, best_synonym[2]

        best_alias: tuple[
            tuple[int, int, float, float, int],
            dict[str, Any],
            str,
        ] | None = None
        # If no synonym qualifies, widen the search to fallback aliases such as
        # catalog names or brand spellings.
        for entry in self.catalog_synonym_records:
            fallback_aliases: list[str] = entry.get("fallback_aliases", [])
            for alias in fallback_aliases:
                normalized_alias = self.normalize_name(alias)
                if not normalized_alias:
                    continue
                if normalized_alias == normalized_query:
                    return entry, False, alias
                accepted, score = self.evaluate_catalog_candidate(
                    normalized_query,
                    normalized_alias,
                    significant_query_tokens,
                )
                if not accepted:
                    continue
                candidate = (score, entry, alias)
                if best_alias is None or candidate[0] > best_alias[0]:
                    best_alias = candidate

        if best_alias is not None:
            return best_alias[1], False, best_alias[2]

        return None

    # -------------------------------------------------------------------------
    def catalog_significant_tokens(self, value: str) -> list[str]:
        tokens = value.split()
        return [
            token
            for token in tokens
            if len(token) >= 4 and token not in MATCHING_STOPWORDS
        ]

    # -------------------------------------------------------------------------
    def evaluate_catalog_candidate(
        self,
        normalized_query: str,
        candidate: str,
        significant_query_tokens: list[str],
    ) -> tuple[bool, tuple[int, int, float, float, int]]:
        # Significant tokens for the candidate allow quick rejection when the
        # query and candidate share no meaningful language.
        candidate_tokens = self.catalog_significant_tokens(candidate)
        shared_tokens = set(significant_query_tokens) & set(candidate_tokens)
        best_token_ratio = 0.0
        if significant_query_tokens and candidate_tokens:
            for query_token in significant_query_tokens:
                for candidate_token in candidate_tokens:
                    ratio = SequenceMatcher(None, query_token, candidate_token).ratio()
                    if ratio > best_token_ratio:
                        best_token_ratio = ratio
        overall_ratio = SequenceMatcher(None, normalized_query, candidate).ratio()
        # Substring matches indicate one value is fully contained in the other,
        # which provides a strong signal even when token overlap is low.
        substring_length = 0
        if candidate in normalized_query:
            substring_length = len(candidate)
        elif normalized_query in candidate:
            substring_length = len(normalized_query)
        accepted = bool(shared_tokens)
        if not accepted:
            if substring_length >= 5:
                accepted = True
            elif best_token_ratio >= 0.85:
                accepted = True
            elif overall_ratio >= 0.90:
                accepted = True
        score = (
            len(shared_tokens),
            substring_length,
            best_token_ratio,
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
        # Label matches originating from the catalog so consumers can trace how
        # an alias contributed to the final result.
        record, confidence, reason, notes = result
        updated_notes = list(notes)
        if from_catalog:
            alias_note = self.coerce_text(alias_value)
            if alias_note:
                updated_notes.insert(0, f"catalog_alias='{alias_note}'")
        return record, confidence, reason, updated_notes

    # -------------------------------------------------------------------------
    def match_primary(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        # Attempt to resolve the query directly against monograph primary names
        # for the highest confidence match.
        record = self.primary_index.get(normalized_query)
        if record is None:
            return None
        return record, self.DIRECT_CONFIDENCE, "monograph_name", []

    # -------------------------------------------------------------------------
    def match_master_list(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        # Use the expanded master alias list (brand and ingredient names) to
        # route the query back to a monograph entry.
        alias_sources = (
            ("brand", self.brand_index),
            ("ingredient", self.ingredient_index),
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
        # Resolve the query through known synonyms extracted from monographs.
        alias = self.synonym_index.get(normalized_query)
        if alias is None:
            return None
        record, original = alias
        notes = [f"synonym='{original}'"]
        return record, self.SYNONYM_CONFIDENCE, "synonym_match", notes

    # -------------------------------------------------------------------------
    def match_partial(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        # Perform token based matching for partial overlaps when full aliases
        # fail, requiring a unique best scoring candidate.
        tokens = [
            token for token in normalized_query.split() if self.is_token_valid(token)
        ]
        if not tokens:
            return None
        candidate_scores: dict[str, int] = {}
        record_lookup: dict[str, MonographRecord] = {}
        for token in tokens:
            for record in self.token_index.get(token, []):
                key = record.normalized_name or record.drug_name.lower()
                record_lookup[key] = record
                candidate_scores[key] = candidate_scores.get(key, 0) + 1
        if not candidate_scores:
            return None
        best_key = max(candidate_scores, key=lambda candidate: candidate_scores[candidate])
        best_score = candidate_scores[best_key]
        tied = [key for key, score in candidate_scores.items() if score == best_score]
        if len(tied) != 1:
            return None
        best_record = record_lookup[best_key]
        matched_tokens: list[str] = []
        for token in tokens:
            for candidate in self.token_index.get(token, []):
                key = candidate.normalized_name or candidate.drug_name.lower()
                if key == best_key:
                    matched_tokens.append(token)
                    break
        notes = [f"token='{token}'" for token in sorted(set(matched_tokens))]
        return best_record, self.PARTIAL_CONFIDENCE, "partial_synonym", notes

    # -------------------------------------------------------------------------
    def match_fuzzy(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        # Fall back to fuzzy matching against all recorded variants when other
        # strategies do not yield a result.
        if len(normalized_query) < 4:
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
        # Reconcile a raw drug name with the monograph set using primary names
        # first and synonyms as a secondary option.
        normalized_name = self.normalize_name(drug_name)
        if not normalized_name:
            return None
        direct = self.match_primary(normalized_name)
        if direct is not None:
            record, confidence, _, _ = direct
            return record, confidence, "drug_name", []
        alias = self.synonym_index.get(normalized_name)
        if alias is None:
            return None
        record, original = alias
        notes = [f"synonym='{original}'"]
        return record, self.SYNONYM_CONFIDENCE, "drug_synonym", notes

    # -------------------------------------------------------------------------
    def find_best_variant(
        self, normalized_query: str
    ) -> tuple[MonographRecord, str, bool, float] | None:
        # Identify the closest normalized variant across primary and synonym
        # spellings using SequenceMatcher ratios.
        best: tuple[MonographRecord, str, bool, float] | None = None
        best_ratio = 0.0
        for candidate, record, original, is_primary in self.variant_catalog:
            ratio = SequenceMatcher(None, normalized_query, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = (record, original, is_primary, ratio)
        if best is None or best_ratio < self.FUZZY_THRESHOLD:
            return None
        return best

    # -------------------------------------------------------------------------
    def build_records(self) -> None:
        # Preprocess the LiverTox dataset into monograph records that expose
        # normalized names, synonyms, and token indexes.
        if self.livertox_df is None or self.livertox_df.empty:
            return
        token_occurrences: dict[str, list[MonographRecord]] = {}
        for row in self.livertox_df.itertuples(index=False):
            raw_name = self.coerce_text(getattr(row, "drug_name", None))
            if raw_name is None:
                continue
            normalized_name = self.normalize_name(raw_name)
            if not normalized_name:
                continue
            nbk_raw = getattr(row, "nbk_id", None)
            nbk_id = str(nbk_raw).strip() if nbk_raw not in (None, "") else ""
            if not nbk_id:
                continue
            excerpt = self.coerce_text(getattr(row, "excerpt", None))
            synonyms_value = getattr(row, "synonyms", None)
            synonyms = self.parse_synonyms(synonyms_value)
            tokens = self.collect_tokens(raw_name, list(synonyms.values()))
            record = MonographRecord(
                nbk_id=nbk_id,
                drug_name=raw_name,
                normalized_name=normalized_name,
                excerpt=excerpt,
                synonyms=synonyms,
                tokens=tokens,
            )
            self.records.append(record)
            if normalized_name not in self.primary_index:
                self.primary_index[normalized_name] = record
            self.variant_catalog.append(
                (normalized_name, record, record.drug_name, True)
            )
            for normalized_synonym, original in synonyms.items():
                if normalized_synonym not in self.synonym_index:
                    self.synonym_index[normalized_synonym] = (record, original)
                # Record every synonym variant so fuzzy searches can inspect a
                # uniform list without repeatedly normalizing strings.
                self.variant_catalog.append(
                    (normalized_synonym, record, original, False)
                )
            for token in tokens:
                bucket = token_occurrences.setdefault(token, [])
                if record not in bucket:
                    bucket.append(record)
        self.records.sort(key=lambda record: record.drug_name.lower())
        self.variant_catalog.sort(key=lambda item: item[0])
        self.token_occurrences = token_occurrences

    # -------------------------------------------------------------------------
    def build_master_list_aliases(self) -> None:
        self.brand_index = {}
        self.ingredient_index = {}
        if self.master_list_df is None or self.master_list_df.empty:
            return
        for row in self.master_list_df.itertuples(index=False):
            drug_name = self.coerce_text(getattr(row, "drug_name", None))
            if drug_name is None:
                continue
            brand = self.coerce_text(getattr(row, "brand_name", None))
            ingredient = self.coerce_text(getattr(row, "ingredient", None))
            for alias_type, value in ("brand", brand), ("ingredient", ingredient):
                if value is None:
                    continue
                if value.lower() == "not available":
                    continue
                for variant in self.iter_alias_variants(value):
                    normalized_variant = self.normalize_name(variant)
                    if not normalized_variant:
                        continue
                    index = (
                        self.brand_index
                        if alias_type == "brand"
                        else self.ingredient_index
                    )
                    bucket = index.setdefault(normalized_variant, [])
                    entry = (variant, drug_name)
                    if entry not in bucket:
                        bucket.append(entry)

    # -------------------------------------------------------------------------
    def prepare_catalog_synonyms(self) -> None:
        self.catalog_synonym_records = []
        if self.drugs_catalog_df is None or self.drugs_catalog_df.empty:
            return
        for row in self.drugs_catalog_df.itertuples(index=False):
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
            # Each catalog entry exposes a map of normalized synonym -> original
            # spelling so we can reference the canonical synonym when a match
            # succeeds.
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
            # Fallback aliases (brand names or raw catalog names) are only used
            # when no synonym qualifies; they expand coverage for less curated
            # catalog entries while remaining lower priority than true synonyms.
            fallback_aliases: list[str] = []
            fallback_seen: set[str] = set()
            for field_name in ("raw_name", "name"):
                alias_value = self.coerce_text(getattr(row, field_name, None))
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
            self.catalog_synonym_records.append(
                {
                    "synonyms": unique_synonyms,
                    "normalized_map": normalized_map,
                    "fallback_aliases": fallback_aliases,
                }
            )

    # -------------------------------------------------------------------------
    def parse_catalog_brand_names(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            segments = re.split(r"[,;/\n]+", value)
        elif isinstance(value, (list, tuple, set)):
            segments = []
            for entry in value:
                segments.extend(re.split(r"[,;/\n]+", str(entry)))
        else:
            segments = [value]
        names: list[str] = []
        for segment in segments:
            text = self.coerce_text(segment)
            if text:
                names.append(text)
        return names

    # -------------------------------------------------------------------------
    def parse_catalog_synonyms(self, value: Any) -> list[str]:
        raw_values = self.extract_synonym_strings(value)
        synonyms: list[str] = []
        for raw in raw_values:
            text = self.coerce_text(raw)
            if text:
                synonyms.append(text)
        return synonyms

    # -------------------------------------------------------------------------
    def finalize_token_index(self) -> None:
        if not self.token_occurrences:
            self.token_index = {}
            return
        filtered: dict[str, list[MonographRecord]] = {}
        for token, records in self.token_occurrences.items():
            if len(records) > self.TOKEN_MAX_FREQUENCY:
                continue
            filtered[token] = sorted(
                records, key=lambda record: record.drug_name.lower()
            )
        self.token_index = filtered

    # -------------------------------------------------------------------------
    def coerce_text(self, value: Any) -> str | None:
        # Standardize scalar values into trimmed strings while respecting NA
        # semantics from pandas inputs.
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        if isinstance(value, str):
            text = value.strip()
            return text or None
        text = str(value).strip()
        return text or None

    # -------------------------------------------------------------------------
    def iter_alias_variants(self, value: str) -> list[str]:
        # Generate text variants for a catalog alias by splitting common
        # delimiters such as commas and slashes.
        normalized_value = self.normalize_whitespace(value)
        if not normalized_value:
            return []
        variants: set[str] = {normalized_value}
        for segment in re.split(r"[;,/\n]+", value):
            candidate = self.normalize_whitespace(segment)
            if candidate:
                variants.add(candidate)
        return list(variants)

    # -------------------------------------------------------------------------
    def derive_master_alias_source(
        self, dataset: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        # Construct a fallback master alias list directly from the LiverTox
        # dataset when a dedicated catalog is unavailable.
        if dataset is None or dataset.empty:
            return None
        required = {"drug_name", "ingredient", "brand_name"}
        if not required.issubset(dataset.columns):
            return None
        alias = dataset[list(required)].copy()
        alias = alias.dropna(subset=["drug_name"])
        alias = alias.replace("Not available", pd.NA)
        alias = alias.dropna(how="all", subset=["ingredient", "brand_name"])
        return alias.reset_index(drop=True)

    # -------------------------------------------------------------------------
    def parse_synonyms(self, value: Any) -> dict[str, str]:
        # Extract normalized synonym mappings from the diverse data shapes
        # present in the source dataset.
        synonyms: dict[str, str] = {}
        raw_values = self.extract_synonym_strings(value)
        if not raw_values:
            text = self.coerce_text(value)
            if text is None:
                return {}
            raw_values = [text]
        for raw in raw_values:
            text = self.coerce_text(raw)
            if text is None:
                continue
            for candidate in re.split(r"[;,/\n]+", text):
                for variant in self.expand_variant(candidate):
                    normalized = self.normalize_name(variant)
                    if not normalized:
                        continue
                    if normalized in MATCHING_STOPWORDS:
                        continue
                    if len(normalized) < 4 and " " not in normalized:
                        continue
                    if normalized not in synonyms:
                        synonyms[normalized] = variant
        return synonyms

    # -------------------------------------------------------------------------
    def extract_synonym_strings(
        self, value: Any, seen_refs: set[int] | None = None
    ) -> list[str]:
        # Walk arbitrarily nested containers to surface raw synonym strings for
        # further normalization.
        if seen_refs is None:
            seen_refs = set()
        if value is None:
            return []
        if isinstance(value, dict):
            marker = id(value)
            if marker in seen_refs:
                return []
            seen_refs.add(marker)
            collected: list[str] = []
            for entry in value.values():
                collected.extend(self.extract_synonym_strings(entry, seen_refs))
            return collected
        if isinstance(value, (list, tuple, set)):
            marker = id(value)
            if marker in seen_refs:
                return []
            seen_refs.add(marker)
            collected: list[str] = []
            for entry in value:
                collected.extend(self.extract_synonym_strings(entry, seen_refs))
            return collected
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
                parsed = self.try_parse_json(stripped)
                if isinstance(parsed, dict) or isinstance(parsed, list):
                    return self.extract_synonym_strings(parsed, seen_refs)
            return [value]
        text = self.coerce_text(value)
        if text is None:
            return []
        return self.extract_synonym_strings(text, seen_refs)

    # -------------------------------------------------------------------------
    def try_parse_json(self, value: str) -> Any:
        # Best effort parsing for serialized synonym blobs stored as strings.
        if not value:
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None

    # -------------------------------------------------------------------------
    def expand_variant(self, value: str) -> list[str]:
        # Produce additional alias candidates by stripping punctuation and
        # parenthetical qualifiers from the provided value.
        normalized = self.normalize_whitespace(value)
        if not normalized:
            return []
        variants = {normalized}
        for segment in re.split(r"[()]", normalized):
            candidate = segment.strip(" -")
            if candidate:
                variants.add(candidate)
        return list(variants)

    # -------------------------------------------------------------------------
    def collect_tokens(self, primary: str, synonyms: list[str]) -> set[str]:
        # Aggregate normalized tokens from both primary names and synonyms to
        # support partial matching.
        tokens: set[str] = set()
        for source in [primary, *synonyms]:
            tokens.update(self.tokenize(source))
        return tokens

    # -------------------------------------------------------------------------
    def tokenize(self, value: str) -> set[str]:
        # Normalize an alias and split it into valid tokens used by the partial
        # matching strategy.
        normalized = self.normalize_name(value)
        if not normalized:
            return set()
        return {token for token in normalized.split() if self.is_token_valid(token)}

    # -------------------------------------------------------------------------
    def is_token_valid(self, token: str) -> bool:
        # Guard the token index against short, stopword, or numeric terms that
        # do not meaningfully improve matching quality.
        if len(token) < 4:
            return False
        if token in MATCHING_STOPWORDS:
            return False
        return not token.isdigit()

    # -------------------------------------------------------------------------
    def normalize_whitespace(self, value: str) -> str:
        # Collapse repeated whitespace to simplify downstream comparisons.
        return re.sub(r"\s+", " ", value).strip()

    # -------------------------------------------------------------------------
    def ensure_row_index(self) -> dict[str, dict[str, Any]]:
        # Cache monograph rows keyed by normalized drug name for quick joins
        # when preparing patient friendly responses.
        if self.rows_by_name:
            return self.rows_by_name
        if self.livertox_df is None or self.livertox_df.empty:
            return {}
        index: dict[str, dict[str, Any]] = {}
        for row in self.livertox_df.to_dict(orient="records"):
            drug_name = self.coerce_text(row.get("drug_name"))
            if drug_name is None:
                continue
            normalized = self.normalize_name(drug_name)
            if not normalized:
                continue
            index[normalized] = row
        self.rows_by_name = index
        return self.rows_by_name

    # -------------------------------------------------------------------------
    def create_match(
        self,
        record: MonographRecord,
        confidence: float,
        reason: str,
        notes: list[str] | None,
    ) -> LiverToxMatch:
        # Normalize the scoring metadata and bundle it with the resolved
        # monograph so downstream consumers receive consistent payloads.
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
    def normalize_name(self, name: str) -> str:
        # Apply Unicode normalization and aggressive punctuation stripping to
        # produce deterministic tokens used throughout matching.
        normalized = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
