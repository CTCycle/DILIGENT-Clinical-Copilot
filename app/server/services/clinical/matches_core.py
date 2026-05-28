from __future__ import annotations
# ruff: noqa: E402

import re
from collections import OrderedDict
from typing import Any, Generic, Iterable, TypeVar

import pandas as pd

from configurations.startup import get_server_settings
from domain.clinical.matching import (
    AliasCacheEntry,
    LiverToxMatch,
    MonographRecord,
)
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.clinical.livertox import LiverToxData

KT = TypeVar("KT")
VT = TypeVar("VT")
CACHE_MISS = object()


def _catalog_excluded_term_suffixes() -> tuple[str, ...]:
    values = get_reference_catalog_snapshot().values(
        "drug_matching",
        "rxnav_excluded_term_suffixes",
        key="default",
    )
    return tuple(value.strip().upper() for value in values if value.strip())


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


from services.clinical import matches_causality, matches_drug, matches_serialization


class DrugsLookup:
    DIRECT_CONFIDENCE = get_server_settings().drugs_matcher.direct_confidence
    MASTER_CONFIDENCE = get_server_settings().drugs_matcher.master_confidence
    SYNONYM_CONFIDENCE = get_server_settings().drugs_matcher.synonym_confidence
    MIN_CONFIDENCE = get_server_settings().drugs_matcher.min_confidence
    NORMALIZATION_CACHE_LIMIT = (
        get_server_settings().drugs_matcher.normalization_cache_limit
    )
    MATCH_CACHE_LIMIT = get_server_settings().drugs_matcher.match_cache_limit
    ALIAS_CACHE_LIMIT = get_server_settings().drugs_matcher.alias_cache_limit
    TOKEN_MIN_LENGTH = get_server_settings().drugs_matcher.token_min_length
    CATALOG_EXCLUDED_TERM_SUFFIXES = _catalog_excluded_term_suffixes()
    CATALOG_INDEX_LIMIT = get_server_settings().drugs_matcher.catalog_index_limit
    SPELLING_CONFIDENCE = get_server_settings().drugs_matcher.spelling_confidence
    SPELLING_MIN_QUERY_LENGTH = (
        get_server_settings().drugs_matcher.spelling_min_query_length
    )
    SPELLING_SHORT_NAME_LENGTH = (
        get_server_settings().drugs_matcher.spelling_short_name_length
    )
    SPELLING_SHORT_MAX_DISTANCE = (
        get_server_settings().drugs_matcher.spelling_short_max_distance
    )
    SPELLING_LONG_MAX_DISTANCE = (
        get_server_settings().drugs_matcher.spelling_long_max_distance
    )
    REGIMEN_SPLIT_RE = re.compile(r"(?:\s*\+\s*|\s*/\s*|\s+\bplus\b\s+)", re.IGNORECASE)
    BRAND_COMBO_PREFERENCES: dict[str, str] = {}

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
        return matches_causality.match_drug_names(self, patient_drugs)

    # -------------------------------------------------------------------------
    def match_query(
        self,
        *,
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
        alias_entries: list[tuple[str, bool]],
    ) -> LiverToxMatch:
        return matches_causality.match_query(
            self,
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            alias_entries=alias_entries,
        )

    # -------------------------------------------------------------------------
    def canonicalize_query(self, value: str | None) -> str:
        return matches_drug.canonicalize_query(self, value)

    # -------------------------------------------------------------------------
    def clone_cached_match(
        self,
        match: LiverToxMatch,
        raw_name: str,
        canonical_query: str,
    ) -> LiverToxMatch:
        return matches_causality.clone_cached_match(
            self, match, raw_name, canonical_query
        )

    # -------------------------------------------------------------------------
    def build_unique_keys(
        self,
        values: list[str],
        normalize_fn: Any,
    ) -> list[str]:
        return matches_drug.build_unique_keys(self, values, normalize_fn)

    # -------------------------------------------------------------------------
    def resolve_source_backed_query_variants(self, normalized_query: str) -> list[str]:
        return matches_drug.resolve_source_backed_query_variants(self, normalized_query)

    # -------------------------------------------------------------------------
    def has_trusted_exact_key(self, normalized_key: str, data: LiverToxData) -> bool:
        return matches_drug.has_trusted_exact_key(self, normalized_key, data)

    # -------------------------------------------------------------------------
    def match_authoritative_spelling_candidates(
        self,
        normalized_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return matches_drug.match_authoritative_spelling_candidates(
            self, normalized_query
        )

    # -------------------------------------------------------------------------
    def is_small_spelling_difference(self, query: str, candidate: str) -> bool:
        return matches_drug.is_small_spelling_difference(self, query, candidate)

    # -------------------------------------------------------------------------
    @staticmethod
    def bounded_edit_distance(left: str, right: str, *, limit: int) -> int:
        return matches_drug.bounded_edit_distance(left, right, limit=limit)

    # -------------------------------------------------------------------------
    def dedupe_stage_matches(
        self,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return matches_drug.dedupe_stage_matches(self, stage_matches)

    # -------------------------------------------------------------------------
    def resolve_stage_matches(
        self,
        keys: list[str],
        resolver: Any,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return matches_causality.resolve_stage_matches(self, keys, resolver)

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
        return matches_causality.finalize_stage_result(
            self,
            stage_name=stage_name,
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            stage_matches=stage_matches,
        )

    # -------------------------------------------------------------------------
    def record_identity_key(self, record: MonographRecord) -> str:
        return matches_drug.record_identity_key(self, record)

    # -------------------------------------------------------------------------
    def rank_stage_matches(
        self,
        *,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return matches_drug.rank_stage_matches(
            self,
            stage_matches=stage_matches,
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
        )

    # -------------------------------------------------------------------------
    def has_strict_rank_winner(
        self,
        *,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
        normalized_query: str,
        preferred_combo: str | None,
    ) -> bool:
        return matches_drug.has_strict_rank_winner(
            self,
            stage_matches=stage_matches,
            normalized_query=normalized_query,
            preferred_combo=preferred_combo,
        )

    # -------------------------------------------------------------------------
    def stage_match_score(
        self,
        *,
        item: tuple[MonographRecord, float, list[str]],
        normalized_query: str,
        preferred_combo: str | None,
    ) -> tuple[int, int, int, int, float, int]:
        return matches_drug.stage_match_score(
            self,
            item=item,
            normalized_query=normalized_query,
            preferred_combo=preferred_combo,
        )

    # -------------------------------------------------------------------------
    def preferred_combo_name(
        self,
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
    ) -> str | None:
        return matches_drug.preferred_combo_name(
            self, raw_name, canonical_query, normalized_query
        )

    # -------------------------------------------------------------------------
    def match_primary_all(
        self,
        canonical_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return matches_drug.match_primary_all(self, canonical_query)

    # -------------------------------------------------------------------------
    def match_alias_exact_all(
        self,
        canonical_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return matches_drug.match_alias_exact_all(self, canonical_query)

    # -------------------------------------------------------------------------
    def match_normalized_all(
        self,
        normalized_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return matches_drug.match_normalized_all(self, normalized_query)

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
        return matches_serialization.create_matched_result(
            self,
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            record=record,
            confidence=confidence,
            reason=reason,
            notes=notes,
            rejected_candidate_names=rejected_candidate_names,
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
        return matches_serialization.create_missing_result(
            self,
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            reason=reason,
            notes=notes,
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
        return matches_serialization.create_ambiguous_result(
            self,
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            reason=reason,
            stage_matches=stage_matches,
        )

    # -------------------------------------------------------------------------
    def resolve_alias_candidates(
        self, original_name: str, normalized_query: str, *, include_catalog: bool = True
    ) -> list[tuple[str, bool]]:
        return matches_drug.resolve_alias_candidates(
            self, original_name, normalized_query, include_catalog=include_catalog
        )

    # -------------------------------------------------------------------------
    def add_alias_entry(
        self,
        alias_entries: list[tuple[str, bool]],
        seen: set[str],
        value: str,
        from_catalog: bool,
    ) -> None:
        return matches_drug.add_alias_entry(
            self, alias_entries, seen, value, from_catalog
        )

    # -------------------------------------------------------------------------
    def find_catalog_synonym_match(
        self, normalized_query: str
    ) -> tuple[dict[str, Any], bool, str] | None:
        return matches_drug.find_catalog_synonym_match(self, normalized_query)

    # -------------------------------------------------------------------------
    def annotate_catalog_match(
        self,
        result: tuple[MonographRecord, float, str, list[str]],
        from_catalog: bool,
        alias_value: str,
    ) -> tuple[MonographRecord, float, str, list[str]]:
        return matches_drug.annotate_catalog_match(
            self, result, from_catalog, alias_value
        )

    # -------------------------------------------------------------------------
    def match_primary(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        return matches_drug.match_primary(self, normalized_query)

    # -------------------------------------------------------------------------
    def match_master_list(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        return matches_drug.match_master_list(self, normalized_query)

    # -------------------------------------------------------------------------
    def match_synonym(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        return matches_drug.match_synonym(self, normalized_query)

    # -------------------------------------------------------------------------
    def match_primary_name(
        self, drug_name: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        return matches_drug.match_primary_name(self, drug_name)

    # -------------------------------------------------------------------------
    def result_sort_key(
        self,
        record: MonographRecord,
        confidence: float,
    ) -> tuple[float, str, str, str]:
        return matches_serialization.result_sort_key(self, record, confidence)

    # -------------------------------------------------------------------------
    def prepare_catalog_synonyms(self) -> None:
        return matches_drug.prepare_catalog_synonyms(self)

    # -------------------------------------------------------------------------
    def register_catalog_entry(
        self,
        entry: dict[str, Any],
        normalized_map: dict[str, str],
        fallback_aliases: list[str],
    ) -> None:
        return matches_drug.register_catalog_entry(
            self, entry, normalized_map, fallback_aliases
        )

    # -------------------------------------------------------------------------
    def add_catalog_index_entry(
        self,
        normalized_value: str,
        entry: dict[str, Any],
        is_synonym: bool,
        original: str,
    ) -> None:
        return matches_drug.add_catalog_index_entry(
            self, normalized_value, entry, is_synonym, original
        )

    # -------------------------------------------------------------------------
    def catalog_term_type_allowed(self, term_type: str | None) -> bool:
        return matches_drug.catalog_term_type_allowed(self, term_type)

    # -------------------------------------------------------------------------
    def parse_catalog_brand_names(self, value: Any) -> list[str]:
        return matches_drug.parse_catalog_brand_names(self, value)

    # -------------------------------------------------------------------------
    def parse_catalog_synonyms(self, value: Any) -> list[str]:
        return matches_drug.parse_catalog_synonyms(self, value)

    # -------------------------------------------------------------------------
    def iter_alias_variants(self, value: str) -> list[str]:
        return matches_drug.iter_alias_variants(self, value)

    # -------------------------------------------------------------------------
    def parse_synonyms(self, value: Any) -> dict[str, str]:
        return matches_drug.parse_synonyms(self, value)

    # -------------------------------------------------------------------------
    def expand_variant(self, value: str) -> list[str]:
        return matches_drug.expand_variant(self, value)

    # -------------------------------------------------------------------------
    def collect_tokens(self, primary: str, synonyms: list[str]) -> set[str]:
        return matches_drug.collect_tokens(self, primary, synonyms)

    # -------------------------------------------------------------------------
    def tokenize(self, value: str) -> set[str]:
        return matches_drug.tokenize(self, value)

    # -------------------------------------------------------------------------
    def is_token_valid(self, token: str) -> bool:
        return matches_drug.is_token_valid(self, token)

    # -------------------------------------------------------------------------
    def create_match(
        self,
        record: MonographRecord,
        confidence: float,
        reason: str,
        notes: list[str] | None,
    ) -> LiverToxMatch:
        return matches_serialization.create_match(
            self, record, confidence, reason, notes
        )

    # -------------------------------------------------------------------------
    def diagnose_missing_drug(self, drug_name: str) -> dict[str, Any]:
        return matches_causality.diagnose_missing_drug(self, drug_name)

    # -------------------------------------------------------------------------
    def normalize_name(self, name: str) -> str:
        return matches_drug.normalize_name(self, name)

    # -------------------------------------------------------------------------
    def require_data(self) -> LiverToxData:
        return matches_drug.require_data(self)


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
