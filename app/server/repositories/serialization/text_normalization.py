from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from common.utils.catalog_loader import CatalogLoader
from common.utils.logger import logger
from repositories.database.session import (
    resolve_engine,
    resolve_session_factory,
)
from repositories.schemas.models import TextNormalizationTerm


SEED_CATALOG = "text_normalization.json"
STRING_LIST_CATEGORIES = {
    "matching_stopwords": "matching_stopword",
    "clinical_generic_terms": "clinical_generic_term",
    "formulation_stopwords": "formulation_stopword",
    "manufacturer_tokens": "manufacturer_token",
    "manufacturer_suffixes": "manufacturer_suffix",
    "rxnav_salt_stopwords": "rxnav_salt_stopword",
    "rxnav_form_stopwords": "rxnav_form_stopword",
    "rxnav_unit_stopwords": "rxnav_unit_stopword",
    "rxnav_name_stopwords": "rxnav_name_stopword",
    "trailing_temporal_tokens": "trailing_temporal_token",
    "drug_non_mentions": "drug_non_mention",
    "drug_duration_words": "drug_duration_word",
    "drug_weekday_words": "drug_weekday_word",
}
MAPPING_CATEGORIES = {
    "query_aliases": "query_alias",
    "lab_marker_aliases": "lab_marker_alias",
    "brand_combo_preferences": "brand_combo_preference",
    "knowledge_source_references": "knowledge_source_reference",
}
SECTION_TITLE_ALIAS_CATEGORIES = {
    "anamnesis": "section_title_alias_anamnesis",
    "drugs": "section_title_alias_drugs",
    "laboratory_analysis": "section_title_alias_laboratory_analysis",
}


def normalize_term(value: str | None) -> str:
    if not value:
        return ""
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


class TextNormalizationVocabularySerializer:
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.engine = resolve_engine(engine)
        self.session_factory = resolve_session_factory(
            engine=self.engine,
            session_factory=session_factory,
        )

    def ensure_seeded(self) -> None:
        db_session = self.session_factory()
        try:
            self.seed_from_catalog(db_session)
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    def seed_from_catalog(self, db_session: Session) -> None:
        payload = CatalogLoader.load_catalog(SEED_CATALOG)
        seeded = 0
        for key, category in STRING_LIST_CATEGORIES.items():
            values = payload.get(key, [])
            if not isinstance(values, list):
                continue
            seeded += self.upsert_terms(
                db_session,
                category=category,
                values=(str(value) for value in values),
                source="seed",
            )
        for key, category in MAPPING_CATEGORIES.items():
            rows = payload.get(key, [])
            if not isinstance(rows, list):
                continue
            seeded += self.upsert_mapping_terms(
                db_session,
                category=category,
                rows=rows,
                source="seed",
            )
        section_aliases = payload.get("section_title_aliases", {})
        if isinstance(section_aliases, dict):
            for section_key, category in SECTION_TITLE_ALIAS_CATEGORIES.items():
                values = section_aliases.get(section_key, [])
                if not isinstance(values, list):
                    continue
                seeded += self.upsert_terms(
                    db_session,
                    category=category,
                    values=(str(value) for value in values),
                    source="seed",
                )
        logger.info("Seeded text normalization vocabulary (%s entries checked)", seeded)

    def upsert_terms(
        self,
        db_session: Session,
        *,
        category: str,
        values: Iterable[str],
        source: str,
    ) -> int:
        count = 0
        for value in values:
            term = self.clean_text(value)
            if term is None:
                continue
            self.upsert_term(
                db_session,
                category=category,
                term=term,
                replacement=None,
                source=source,
            )
            count += 1
        return count

    def upsert_term(
        self,
        db_session: Session,
        *,
        category: str,
        term: str,
        replacement: str | None,
        source: str,
        increment: bool = False,
    ) -> None:
        clean_term = self.clean_text(term)
        if clean_term is None:
            return
        term_norm = normalize_term(clean_term)
        if not term_norm:
            return
        existing = (
            db_session.execute(
                select(TextNormalizationTerm).where(
                    TextNormalizationTerm.category == category,
                    TextNormalizationTerm.term_norm == term_norm,
                )
            )
            .scalars()
            .first()
        )
        if existing is None:
            db_session.add(
                TextNormalizationTerm(
                    category=category,
                    term=clean_term,
                    term_norm=term_norm,
                    replacement=self.clean_text(replacement),
                    source=source,
                    encounter_count=1 if increment else 0,
                    is_active=True,
                )
            )
            return
        if replacement is not None:
            existing.replacement = self.clean_text(replacement)
        if not existing.is_active and source == "seed":
            existing.is_active = True
        if increment:
            existing.encounter_count = int(existing.encounter_count or 0) + 1

    def list_terms(
        self,
        db_session: Session,
        *,
        category: str | None = None,
    ) -> list[TextNormalizationTerm]:
        query = select(TextNormalizationTerm)
        if category:
            query = query.where(TextNormalizationTerm.category == category)
        return list(
            db_session.execute(
                query.order_by(
                    TextNormalizationTerm.category, TextNormalizationTerm.term_norm
                )
            )
            .scalars()
            .all()
        )

    def set_term_active(
        self,
        db_session: Session,
        *,
        category: str,
        term: str,
        is_active: bool,
    ) -> bool:
        term_norm = normalize_term(term)
        if not term_norm:
            return False
        existing = (
            db_session.execute(
                select(TextNormalizationTerm).where(
                    TextNormalizationTerm.category == category,
                    TextNormalizationTerm.term_norm == term_norm,
                )
            )
            .scalars()
            .first()
        )
        if existing is None:
            return False
        existing.is_active = bool(is_active)
        return True

    def upsert_mapping_terms(
        self,
        db_session: Session,
        *,
        category: str,
        rows: Iterable[dict[str, Any]],
        source: str,
    ) -> int:
        count = 0
        for row in rows:
            term = self.clean_text(row.get("term"))
            replacement = self.clean_text(row.get("replacement"))
            if term is None or replacement is None:
                continue
            self.upsert_term(
                db_session,
                category=category,
                term=term,
                replacement=replacement,
                source=source,
            )
            count += 1
        return count

    @staticmethod
    def clean_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["TextNormalizationVocabularySerializer", "normalize_term"]

