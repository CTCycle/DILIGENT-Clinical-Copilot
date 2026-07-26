from __future__ import annotations
import re
from typing import Any, Iterable

import pandas as pd

from configurations.startup import get_server_settings
from domain.clinical.matching import (
    AliasCacheEntry,
    LiverToxMatch,
    MonographRecord,
)
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.clinical.livertox import LiverToxData
from common.utils.bounded_cache import BoundedCache
from services.clinical.drug_matcher import DrugMatcher
from services.clinical.drug_name_service import DrugNameService


###############################################################################
def _catalog_excluded_term_suffixes() -> tuple[str, ...]:
    values = get_reference_catalog_snapshot().values(
        "drug_matching",
        "rxnav_excluded_term_suffixes",
        key="default",
    )
    return tuple(value.strip().upper() for value in values if value.strip())


###############################################################################
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
        self.drug_matcher = DrugMatcher(self)
        self.drug_name_service = DrugNameService(self)

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
        return self.drug_matcher.match_drug_names(patient_drugs)

    # -------------------------------------------------------------------------
    def match_query(
        self,
        *,
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
        alias_entries: list[tuple[str, bool]],
    ) -> LiverToxMatch:
        return self.drug_matcher.match_query(
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            alias_entries=alias_entries,
        )

    # -------------------------------------------------------------------------
    def canonicalize_query(self, value: str | None) -> str:
        return self.drug_name_service.canonicalize_query(value)

    # -------------------------------------------------------------------------
    def clone_cached_match(
        self,
        match: LiverToxMatch,
        raw_name: str,
        canonical_query: str,
    ) -> LiverToxMatch:
        return self.drug_matcher.clone_cached_match(match, raw_name, canonical_query)

    # -------------------------------------------------------------------------
    def build_unique_keys(
        self,
        values: list[str],
        normalize_fn: Any,
    ) -> list[str]:
        return self.drug_name_service.build_unique_keys(values, normalize_fn)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def has_trusted_exact_key(self, normalized_key: str, data: LiverToxData) -> bool:
        return self.drug_name_service.has_trusted_exact_key(normalized_key, data)

    # -------------------------------------------------------------------------
    def match_authoritative_spelling_candidates(
        self,
        normalized_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return self.drug_name_service.match_authoritative_spelling_candidates(
            normalized_query
        )

    # -------------------------------------------------------------------------
    def is_small_spelling_difference(self, query: str, candidate: str) -> bool:
        return self.drug_name_service.is_small_spelling_difference(query, candidate)

    # -------------------------------------------------------------------------
    @staticmethod
    def bounded_edit_distance(left: str, right: str, *, limit: int) -> int:
        return DrugNameService.bounded_edit_distance(left, right, limit=limit)

    # -------------------------------------------------------------------------
    def dedupe_stage_matches(
        self,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return self.drug_name_service.dedupe_stage_matches(stage_matches)

    # -------------------------------------------------------------------------
    def resolve_stage_matches(
        self,
        keys: list[str],
        resolver: Any,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return self.drug_matcher.resolve_stage_matches(keys, resolver)

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
        return self.drug_matcher.finalize_stage_result(
            stage_name=stage_name,
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            stage_matches=stage_matches,
        )

    # -------------------------------------------------------------------------
    def record_identity_key(self, record: MonographRecord) -> str:
        return self.drug_name_service.record_identity_key(record)

    # -------------------------------------------------------------------------
    def rank_stage_matches(
        self,
        *,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return self.drug_name_service.rank_stage_matches(
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
        return self.drug_name_service.has_strict_rank_winner(
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
        return self.drug_name_service.stage_match_score(
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
        return self.drug_name_service.preferred_combo_name(
            raw_name, canonical_query, normalized_query
        )

    # -------------------------------------------------------------------------
    def match_primary_all(
        self,
        canonical_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return self.drug_name_service.match_primary_all(canonical_query)

    # -------------------------------------------------------------------------
    def match_alias_exact_all(
        self,
        canonical_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return self.drug_name_service.match_alias_exact_all(canonical_query)

    # -------------------------------------------------------------------------
    def match_normalized_all(
        self,
        normalized_query: str,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        return self.drug_name_service.match_normalized_all(normalized_query)

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
        return self.drug_matcher.create_matched_result(
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
        return self.drug_matcher.create_missing_result(
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
        return self.drug_matcher.create_ambiguous_result(
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
        return self.drug_name_service.resolve_alias_candidates(
            original_name, normalized_query, include_catalog=include_catalog
        )

    # -------------------------------------------------------------------------
    def add_alias_entry(
        self,
        alias_entries: list[tuple[str, bool]],
        seen: set[str],
        value: str,
        from_catalog: bool,
    ) -> None:
        return self.drug_name_service.add_alias_entry(
            alias_entries, seen, value, from_catalog
        )

    # -------------------------------------------------------------------------
    def find_catalog_synonym_match(
        self, normalized_query: str
    ) -> tuple[dict[str, Any], bool, str] | None:
        return self.drug_name_service.find_catalog_synonym_match(normalized_query)

    # -------------------------------------------------------------------------
    def annotate_catalog_match(
        self,
        result: tuple[MonographRecord, float, str, list[str]],
        from_catalog: bool,
        alias_value: str,
    ) -> tuple[MonographRecord, float, str, list[str]]:
        return self.drug_name_service.annotate_catalog_match(
            result, from_catalog, alias_value
        )

    # -------------------------------------------------------------------------
    def match_primary(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        return self.drug_name_service.match_primary(normalized_query)

    # -------------------------------------------------------------------------
    def match_master_list(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        return self.drug_name_service.match_master_list(normalized_query)

    # -------------------------------------------------------------------------
    def match_synonym(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        return self.drug_name_service.match_synonym(normalized_query)

    # -------------------------------------------------------------------------
    def match_primary_name(
        self, drug_name: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        return self.drug_name_service.match_primary_name(drug_name)

    # -------------------------------------------------------------------------
    def result_sort_key(
        self,
        record: MonographRecord,
        confidence: float,
    ) -> tuple[float, str, str, str]:
        return self.drug_matcher.result_sort_key(record, confidence)

    # -------------------------------------------------------------------------
    def prepare_catalog_synonyms(self) -> None:
        return self.drug_name_service.prepare_catalog_synonyms()

    # -------------------------------------------------------------------------
    def register_catalog_entry(
        self,
        entry: dict[str, Any],
        normalized_map: dict[str, str],
        fallback_aliases: list[str],
    ) -> None:
        return self.drug_name_service.register_catalog_entry(
            entry, normalized_map, fallback_aliases
        )

    # -------------------------------------------------------------------------
    def add_catalog_index_entry(
        self,
        normalized_value: str,
        entry: dict[str, Any],
        is_synonym: bool,
        original: str,
    ) -> None:
        return self.drug_name_service.add_catalog_index_entry(
            normalized_value, entry, is_synonym, original
        )

    # -------------------------------------------------------------------------
    def catalog_term_type_allowed(self, term_type: str | None) -> bool:
        return self.drug_name_service.catalog_term_type_allowed(term_type)

    # -------------------------------------------------------------------------
    def parse_catalog_brand_names(self, value: Any) -> list[str]:
        return self.drug_name_service.parse_catalog_brand_names(value)

    # -------------------------------------------------------------------------
    def parse_catalog_synonyms(self, value: Any) -> list[str]:
        return self.drug_name_service.parse_catalog_synonyms(value)

    # -------------------------------------------------------------------------
    def iter_alias_variants(self, value: str) -> list[str]:
        return self.drug_name_service.iter_alias_variants(value)

    # -------------------------------------------------------------------------
    def parse_synonyms(self, value: Any) -> dict[str, str]:
        return self.drug_name_service.parse_synonyms(value)

    # -------------------------------------------------------------------------
    def expand_variant(self, value: str) -> list[str]:
        return self.drug_name_service.expand_variant(value)

    # -------------------------------------------------------------------------
    def collect_tokens(self, primary: str, synonyms: list[str]) -> set[str]:
        return self.drug_name_service.collect_tokens(primary, synonyms)

    # -------------------------------------------------------------------------
    def tokenize(self, value: str) -> set[str]:
        return self.drug_name_service.tokenize(value)

    # -------------------------------------------------------------------------
    def is_token_valid(self, token: str) -> bool:
        return self.drug_name_service.is_token_valid(token)

    # -------------------------------------------------------------------------
    def create_match(
        self,
        record: MonographRecord,
        confidence: float,
        reason: str,
        notes: list[str] | None,
    ) -> LiverToxMatch:
        return self.drug_matcher.create_match(record, confidence, reason, notes)

    # -------------------------------------------------------------------------
    def diagnose_missing_drug(self, drug_name: str) -> dict[str, Any]:
        return self.drug_matcher.diagnose_missing_drug(drug_name)

    # -------------------------------------------------------------------------
    def normalize_name(self, name: str) -> str:
        return self.drug_name_service.normalize_name(name)

    # -------------------------------------------------------------------------
    def require_data(self) -> LiverToxData:
        return self.drug_name_service.require_data()


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
