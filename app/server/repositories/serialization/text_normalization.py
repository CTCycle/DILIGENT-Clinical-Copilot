from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from common.utils.catalog_loader import CatalogLoader
from common.utils.logger import logger
from domain.text import TextNormalizationSnapshot, empty_text_normalization_snapshot
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
QUERY_ALIAS_CATEGORY = "query_alias"
LAB_MARKER_ALIAS_CATEGORY = "lab_marker_alias"
BRAND_COMBO_PREFERENCE_CATEGORY = "brand_combo_preference"
KNOWLEDGE_SOURCE_REFERENCE_CATEGORY = "knowledge_source_reference"


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
        if engine is None and session_factory is None:
            raise ValueError("engine or session_factory is required")
        self.engine = engine
        self.session_factory = session_factory or sessionmaker(
            bind=engine,
            future=True,
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

    def load_snapshot(self) -> TextNormalizationSnapshot:
        db_session = self.session_factory()
        try:
            rows = self.list_terms(db_session)
            return self._build_snapshot(rows)
        except SQLAlchemyError:
            logger.warning(
                "Text normalization vocabulary unavailable; using empty snapshot.",
                exc_info=True,
            )
            return empty_text_normalization_snapshot()
        finally:
            db_session.close()

    def list_term_payloads(self, *, category: str | None = None) -> list[dict[str, Any]]:
        db_session = self.session_factory()
        try:
            rows = self.list_terms(db_session, category=category)
            return [self.term_to_payload(row) for row in rows]
        finally:
            db_session.close()

    def upsert_term_payload(
        self,
        *,
        category: str,
        term: str,
        replacement: str | None,
        source: str,
        is_active: bool,
    ) -> dict[str, Any]:
        db_session = self.session_factory()
        try:
            self.upsert_term(
                db_session,
                category=category,
                term=term,
                replacement=replacement,
                source=source,
            )
            self.set_term_active(
                db_session,
                category=category,
                term=term,
                is_active=is_active,
            )
            db_session.commit()
            row = self.get_term(db_session, category=category, term=term)
            if row is None:
                raise RuntimeError("Term upsert did not return persisted row")
            return self.term_to_payload(row)
        except SQLAlchemyError:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    def deactivate_term(self, *, category: str, term: str) -> bool:
        db_session = self.session_factory()
        try:
            updated = self.set_term_active(
                db_session,
                category=category,
                term=term,
                is_active=False,
            )
            if updated:
                db_session.commit()
            else:
                db_session.rollback()
            return updated
        except SQLAlchemyError:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    def record_observation(self, *, category: str, term: str) -> None:
        db_session = self.session_factory()
        try:
            self.upsert_term(
                db_session,
                category=category,
                term=term,
                replacement=None,
                source="session",
                increment=True,
            )
            db_session.commit()
        except SQLAlchemyError:
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

    def get_term(
        self,
        db_session: Session,
        *,
        category: str,
        term: str,
    ) -> TextNormalizationTerm | None:
        term_norm = normalize_term(term)
        if not term_norm:
            return None
        return (
            db_session.execute(
                select(TextNormalizationTerm).where(
                    TextNormalizationTerm.category == category,
                    TextNormalizationTerm.term_norm == term_norm,
                )
            )
            .scalars()
            .first()
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

    @staticmethod
    def term_to_payload(row: TextNormalizationTerm) -> dict[str, Any]:
        return {
            "id": row.id,
            "category": row.category,
            "term": row.term,
            "replacement": row.replacement,
            "source": row.source,
            "encounter_count": int(row.encounter_count or 0),
            "is_active": bool(row.is_active),
        }

    @staticmethod
    def _build_snapshot(rows: list[TextNormalizationTerm]) -> TextNormalizationSnapshot:
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
            "drug_non_mention": set(),
            "drug_duration_word": set(),
            "drug_weekday_word": set(),
            "section_title_alias_anamnesis": set(),
            "section_title_alias_drugs": set(),
            "section_title_alias_laboratory_analysis": set(),
        }
        query_aliases: dict[str, str] = {}
        lab_marker_aliases: dict[str, str] = {}
        brand_combo_preferences: dict[str, str] = {}
        knowledge_source_references: dict[str, str] = {}
        for row in rows:
            if not bool(row.is_active):
                continue
            term_norm = normalize_term(row.term)
            if not term_norm:
                continue
            if row.category == QUERY_ALIAS_CATEGORY:
                replacement = normalize_term(row.replacement)
                if replacement:
                    query_aliases[term_norm] = replacement
                continue
            if row.category == LAB_MARKER_ALIAS_CATEGORY:
                replacement = normalize_term(row.replacement)
                if replacement:
                    lab_marker_aliases[term_norm] = replacement
                continue
            if row.category == BRAND_COMBO_PREFERENCE_CATEGORY:
                replacement = normalize_term(row.replacement)
                if replacement:
                    brand_combo_preferences[term_norm] = replacement
                continue
            if row.category == KNOWLEDGE_SOURCE_REFERENCE_CATEGORY:
                replacement = (row.replacement or "").strip()
                if replacement:
                    knowledge_source_references[term_norm] = replacement
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
        section_title_aliases = {
            "anamnesis": frozenset(grouped["section_title_alias_anamnesis"]),
            "drugs": frozenset(grouped["section_title_alias_drugs"]),
            "laboratory_analysis": frozenset(
                grouped["section_title_alias_laboratory_analysis"]
            ),
        }
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
            drug_non_mentions=frozenset(grouped["drug_non_mention"]),
            drug_duration_words=frozenset(grouped["drug_duration_word"]),
            drug_weekday_words=frozenset(grouped["drug_weekday_word"]),
            lab_marker_aliases=lab_marker_aliases,
            brand_combo_preferences=brand_combo_preferences,
            knowledge_source_references=knowledge_source_references,
            section_title_aliases=section_title_aliases,
        )


__all__ = ["TextNormalizationVocabularySerializer", "normalize_term"]

