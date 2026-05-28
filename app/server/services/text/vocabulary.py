from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from common.utils.logger import logger
from domain.catalogs import normalize_catalog_value
from domain.text import TextNormalizationSnapshot, empty_text_normalization_snapshot
from repositories.database.session import get_default_repository
from repositories.serialization.catalogs import ReferenceCatalogSerializer
from services.catalogs.runtime import get_reference_catalog_snapshot


@lru_cache(maxsize=1)
def get_text_normalization_snapshot() -> TextNormalizationSnapshot:
    try:
        snapshot = get_reference_catalog_snapshot()
        repository = get_default_repository()
        catalogs = ReferenceCatalogSerializer(repository.session_factory)
        query_aliases: dict[str, str] = {}
        brand_combo_preferences: dict[str, str] = {}
        for entry in catalogs.list_active_entries():
            if (
                entry.domain == "text_normalization"
                and entry.category == "query_aliases"
                and entry.active
            ):
                query_aliases[normalize_catalog_value(entry.value)] = (
                    normalize_catalog_value(entry.key)
                )
            if (
                entry.domain == "drug_matching"
                and entry.category == "catalog_fallback_aliases"
            ):
                brand_combo_preferences[normalize_catalog_value(entry.value)] = (
                    normalize_catalog_value(entry.key)
                )
        manufacturer_tokens = frozenset(
            normalize_catalog_value(value)
            for value in snapshot.values("drug_matching", "company_suffix_terms")
        )
        formulation_stopwords = frozenset(
            set(snapshot.values("text_normalization", "matching_stopwords"))
            | set(snapshot.values("drug_matching", "matching_stopwords"))
        )
        return TextNormalizationSnapshot(
            matching_stopwords=frozenset(
                snapshot.values("text_normalization", "matching_stopwords")
            ),
            clinical_generic_terms=frozenset(
                snapshot.values("text_normalization", "generic_drug_terms")
            ),
            formulation_stopwords=formulation_stopwords,
            manufacturer_tokens=manufacturer_tokens,
            manufacturer_suffixes=tuple(sorted(manufacturer_tokens)),
            rxnav_salt_stopwords=frozenset(),
            rxnav_form_stopwords=frozenset(),
            rxnav_unit_stopwords=frozenset(),
            rxnav_name_stopwords=frozenset(),
            trailing_temporal_tokens=frozenset(
                set(snapshot.values("clinical_extraction", "drug_continuation_markers"))
                | set(snapshot.values("clinical_extraction", "drug_frequency_terms"))
                | set(snapshot.values("text_normalization", "noisy_phrases"))
            ),
            query_aliases=query_aliases,
            noisy_phrases=frozenset(
                snapshot.values("text_normalization", "noisy_phrases")
            ),
            drug_non_mentions=frozenset(
                snapshot.values("text_normalization", "drug_non_mentions")
            ),
            drug_duration_words=frozenset(),
            drug_weekday_words=frozenset(),
            lab_marker_aliases={},
            brand_combo_preferences=brand_combo_preferences,
            knowledge_source_references={},
            section_title_aliases={},
        )
    except Exception:
        logger.warning(
            "Text normalization vocabulary unavailable; using empty snapshot.",
            exc_info=True,
        )
        return empty_text_normalization_snapshot()


def invalidate_text_normalization_snapshot() -> None:
    get_text_normalization_snapshot.cache_clear()


def record_text_normalization_observation(term: str, *, category: str) -> None:
    clean_category = category.strip()
    if not clean_category:
        return
    try:
        repository = get_default_repository()
        session_factory = repository.session_factory
    except Exception:
        logger.debug("Skipping text normalization observation; database unavailable.")
        return
    try:
        catalogs = ReferenceCatalogSerializer(session_factory)
        catalogs.upsert_runtime_observation(
            term=term,
            category=clean_category,
            source="session",
            is_active=True,
        )
        invalidate_text_normalization_snapshot()
    except Exception:
        logger.debug("Failed recording text normalization observation.", exc_info=True)


def list_text_normalization_term_payloads(
    *, category: str | None = None
) -> list[dict[str, Any]]:
    repository = get_default_repository()
    try:
        catalogs = ReferenceCatalogSerializer(repository.session_factory)
        rows = catalogs.list_runtime_observations(category=category)
        payloads: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            replacement: str | None = None
            if row.metadata_json:
                try:
                    metadata = json.loads(row.metadata_json)
                    if isinstance(metadata, dict):
                        replacement = (
                            str(metadata.get("replacement") or "").strip() or None
                        )
                except json.JSONDecodeError:
                    replacement = None
            payloads.append(
                {
                    "id": index,
                    "category": row.category,
                    "term": row.value,
                    "replacement": replacement,
                    "source": "session",
                    "encounter_count": 1,
                    "is_active": bool(row.active),
                }
            )
        return payloads
    except Exception:
        logger.warning("Failed listing text normalization terms.", exc_info=True)
        return []


def upsert_text_normalization_term_payload(
    *,
    category: str,
    term: str,
    replacement: str | None,
    source: str,
    is_active: bool,
) -> dict[str, Any]:
    repository = get_default_repository()
    if not normalize_catalog_value(term):
        raise ValueError("term must not be blank")
    catalogs = ReferenceCatalogSerializer(repository.session_factory)
    row_id = catalogs.upsert_runtime_observation(
        term=term,
        category=category,
        replacement=replacement,
        source=source,
        is_active=is_active,
    )
    invalidate_text_normalization_snapshot()
    return {
        "id": int(row_id or 0),
        "category": category,
        "term": term,
        "replacement": replacement,
        "source": source,
        "encounter_count": 1,
        "is_active": bool(is_active),
    }


def deactivate_text_normalization_term_payload(*, category: str, term: str) -> bool:
    repository = get_default_repository()
    catalogs = ReferenceCatalogSerializer(repository.session_factory)
    changed = catalogs.deactivate_runtime_observation(category=category, term=term)
    if changed:
        invalidate_text_normalization_snapshot()
    return changed


__all__ = [
    "TextNormalizationSnapshot",
    "deactivate_text_normalization_term_payload",
    "get_text_normalization_snapshot",
    "invalidate_text_normalization_snapshot",
    "list_text_normalization_term_payloads",
    "record_text_normalization_observation",
    "upsert_text_normalization_term_payload",
]
