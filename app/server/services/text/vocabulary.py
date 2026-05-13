from __future__ import annotations

from functools import lru_cache

from common.utils.logger import logger
from domain.text import TextNormalizationSnapshot, empty_text_normalization_snapshot
from repositories.database.session import get_default_repository
from repositories.serialization.text_normalization import (
    TextNormalizationVocabularySerializer,
)


@lru_cache(maxsize=1)
def get_text_normalization_snapshot() -> TextNormalizationSnapshot:
    try:
        repository = get_default_repository()
        serializer = TextNormalizationVocabularySerializer(
            engine=repository.engine,
            session_factory=repository.session_factory,
        )
        return serializer.load_snapshot()
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
        serializer = TextNormalizationVocabularySerializer(
            engine=repository.engine,
            session_factory=repository.session_factory,
        )
    except Exception:
        logger.debug("Skipping text normalization observation; database unavailable.")
        return
    try:
        serializer.record_observation(
            category=clean_category,
            term=term,
        )
        invalidate_text_normalization_snapshot()
    except Exception:
        logger.debug(
            "Failed recording text normalization observation.",
            exc_info=True,
        )


__all__ = [
    "TextNormalizationSnapshot",
    "get_text_normalization_snapshot",
    "invalidate_text_normalization_snapshot",
    "record_text_normalization_observation",
]

