from __future__ import annotations

from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.domain.text import TextNormalizationSnapshot
from DILIGENT.server.repositories.database.session import get_default_repository
from DILIGENT.server.repositories.schemas.models import TextNormalizationTerm
from DILIGENT.server.repositories.serialization.text_normalization import (
    TextNormalizationVocabularySerializer,
    normalize_term,
)


EMPTY_SNAPSHOT = TextNormalizationSnapshot(
    matching_stopwords=frozenset(),
    clinical_generic_terms=frozenset(),
    formulation_stopwords=frozenset(),
    manufacturer_tokens=frozenset(),
    manufacturer_suffixes=(),
    rxnav_salt_stopwords=frozenset(),
    rxnav_form_stopwords=frozenset(),
    rxnav_unit_stopwords=frozenset(),
    rxnav_name_stopwords=frozenset(),
    trailing_temporal_tokens=frozenset(),
    query_aliases={},
)


@lru_cache(maxsize=1)
def get_text_normalization_snapshot() -> TextNormalizationSnapshot:
    try:
        repository = get_default_repository()
        db_session = repository.session_factory()
    except Exception as exc:
        logger.warning("Text normalization vocabulary unavailable: %s", exc)
        return EMPTY_SNAPSHOT

    try:
        rows = (
            db_session.execute(
                select(TextNormalizationTerm).where(
                    TextNormalizationTerm.is_active.is_(True)
                )
            )
            .scalars()
            .all()
        )
    except SQLAlchemyError as exc:
        logger.warning("Failed loading text normalization vocabulary: %s", exc)
        return EMPTY_SNAPSHOT
    finally:
        db_session.close()

    grouped: dict[str, set[str]] = {
        "matching_stopword": set(),
        "clinical_generic_term": set(),
        "formulation_stopword": set(),
        "manufacturer_token": set(),
        "manufacturer_suffix": set(),
        "rxnav_salt_stopword": set(),
        "rxnav_form_stopword": set(),
        "rxnav_unit_stopword": set(),
        "rxnav_name_stopword": set(),
        "trailing_temporal_token": set(),
    }
    query_aliases: dict[str, str] = {}
    for row in rows:
        term_norm = normalize_term(row.term)
        if not term_norm:
            continue
        if row.category == "query_alias":
            replacement = normalize_term(row.replacement)
            if replacement:
                query_aliases[term_norm] = replacement
            continue
        bucket = grouped.get(row.category)
        if bucket is not None:
            bucket.add(term_norm)

    formulation_stopwords = (
        grouped["formulation_stopword"]
        | grouped["matching_stopword"]
        | grouped["clinical_generic_term"]
    )
    rxnav_name_stopwords = (
        grouped["rxnav_name_stopword"]
        | grouped["rxnav_salt_stopword"]
        | grouped["rxnav_form_stopword"]
        | grouped["rxnav_unit_stopword"]
    )
    return TextNormalizationSnapshot(
        matching_stopwords=frozenset(grouped["matching_stopword"]),
        clinical_generic_terms=frozenset(grouped["clinical_generic_term"]),
        formulation_stopwords=frozenset(formulation_stopwords),
        manufacturer_tokens=frozenset(grouped["manufacturer_token"]),
        manufacturer_suffixes=tuple(sorted(grouped["manufacturer_suffix"])),
        rxnav_salt_stopwords=frozenset(grouped["rxnav_salt_stopword"]),
        rxnav_form_stopwords=frozenset(grouped["rxnav_form_stopword"]),
        rxnav_unit_stopwords=frozenset(grouped["rxnav_unit_stopword"]),
        rxnav_name_stopwords=frozenset(rxnav_name_stopwords),
        trailing_temporal_tokens=frozenset(grouped["trailing_temporal_token"]),
        query_aliases=query_aliases,
    )


def invalidate_text_normalization_snapshot() -> None:
    get_text_normalization_snapshot.cache_clear()


def record_text_normalization_observation(term: str, *, category: str) -> None:
    clean_category = category.strip()
    if not clean_category:
        return
    try:
        repository = get_default_repository()
        serializer = TextNormalizationVocabularySerializer(
            engine=repository.engine,
            session_factory=repository.session_factory,
        )
        db_session = repository.session_factory()
    except Exception as exc:
        logger.debug("Skipping text normalization observation; database unavailable: %s", exc)
        return
    try:
        serializer.upsert_term(
            db_session,
            category=clean_category,
            term=term,
            replacement=None,
            source="session",
            increment=True,
        )
        db_session.commit()
        invalidate_text_normalization_snapshot()
    except Exception as exc:
        db_session.rollback()
        logger.debug("Failed recording text normalization observation: %s", exc)
    finally:
        db_session.close()


__all__ = [
    "TextNormalizationSnapshot",
    "get_text_normalization_snapshot",
    "invalidate_text_normalization_snapshot",
    "record_text_normalization_observation",
]
