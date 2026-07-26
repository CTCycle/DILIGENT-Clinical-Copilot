from __future__ import annotations

from typing import Any, Iterator

import pandas as pd
from sqlalchemy.orm import Session

from repositories.context import RepositoryContext
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.schemas.knowledge import DrugAlias, LiverToxMonograph
from repositories.serialization import evidence_aliases, evidence_data, rxnav_data, session_result_data


class KnowledgeRepository:
    def __init__(self, context: RepositoryContext, drug_catalog_repository: DrugCatalogRepository) -> None:
        self.context = context
        self.drug_catalog_repository = drug_catalog_repository
        self.engine = context.engine
        self.session_factory = context.session_factory

    def save_livertox_records(self, records: pd.DataFrame) -> None:
        evidence_data.save_livertox_records(self, records)

    def get_livertox_records(self) -> pd.DataFrame:
        return evidence_data.get_livertox_records(self)

    def get_livertox_master_list(self) -> pd.DataFrame:
        return evidence_data.get_livertox_master_list(self)

    def get_drugs_catalog(self) -> pd.DataFrame:
        return evidence_data.get_drugs_catalog(self)

    def stream_drugs_catalog(self, page_size: int | None = None) -> Iterator[pd.DataFrame]:
        return evidence_data.stream_drugs_catalog(self, page_size)

    def list_livertox_catalog(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        return evidence_data.list_livertox_catalog(self, **kwargs)

    def get_livertox_excerpt(self, drug_id: int) -> dict[str, Any] | None:
        return evidence_data.get_livertox_excerpt(self, drug_id)

    def get_drug_knowledge_bundle(self, drug_id: int) -> dict[str, Any]:
        return evidence_data.get_drug_knowledge_bundle(self, drug_id)

    def resolve_drug_id_from_match_cache(self, db_session: Session, **kwargs: Any) -> int | None:
        return evidence_data.resolve_drug_id_from_match_cache(self, db_session, **kwargs)

    def upsert_high_confidence_kb_match_cache(self, db_session: Session, **kwargs: Any) -> None:
        evidence_data.upsert_high_confidence_kb_match_cache(self, db_session, **kwargs)

    def load_livertox_match_from_db_cache(self, **kwargs: Any) -> dict[str, Any] | None:
        return evidence_data.load_livertox_match_from_db_cache(self, **kwargs)

    def prepare_livertox_rows(self, records: pd.DataFrame) -> list[dict[str, Any]]:
        return evidence_data.prepare_livertox_rows(self, records)

    def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        return rxnav_data.sanitize_livertox_records(self, records)

    def livertox_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return evidence_data.livertox_row_sort_key(self, row)

    def to_sortable_text(self, value: Any) -> str:
        return evidence_data.to_sortable_text(self, value)

    def build_livertox_monograph_key(self, row: dict[str, Any]) -> str:
        return evidence_data.build_livertox_monograph_key(self, row)

    def upsert_livertox_monograph(self, **kwargs: Any) -> None:
        evidence_data.upsert_livertox_monograph(self, **kwargs)

    def try_assign_livertox_nbk_id(self, db_session: Session, **kwargs: Any) -> None:
        evidence_data.try_assign_livertox_nbk_id(self, db_session, **kwargs)

    def get_monograph_by_key(self, db_session: Session, monograph_key: str) -> LiverToxMonograph | None:
        return evidence_aliases.get_monograph_by_key(self, db_session, monograph_key)

    def get_monograph_by_drug_id(self, db_session: Session, drug_id: int) -> LiverToxMonograph | None:
        return evidence_aliases.get_monograph_by_drug_id(self, db_session, drug_id)

    def normalize_string(self, value: Any) -> str | None:
        return session_result_data.normalize_string(self, value)

    def normalize_date(self, value: Any) -> str | None:
        return session_result_data.normalize_date(self, value)

    def normalize_date_value(self, value: Any):
        return session_result_data.normalize_date_value(self, value)

    def normalize_flag(self, value: Any) -> int | None:
        return session_result_data.normalize_flag(self, value)

    def to_int(self, value: Any) -> int | None:
        return session_result_data.to_int(self, value)

    def join_values(self, values: set[str]) -> str | None:
        return session_result_data.join_values(self, values)

    def build_search_pattern(self, search: str | None) -> str | None:
        return evidence_data.build_search_pattern(self, search)

    def get_drug_by_rxcui(self, db_session: Session, rxcui: str | None):
        return self.drug_catalog_repository.get_drug_by_rxcui(db_session, rxcui)

    def upsert_drug_alias(self, db_session: Session, **kwargs: Any) -> None:
        self.drug_catalog_repository.upsert_drug_alias(db_session, **kwargs)

    def extract_text_candidates(self, value: Any) -> list[str]:
        return self.drug_catalog_repository.extract_text_candidates(value)

    def extract_synonym_candidates(self, value: Any) -> list[str]:
        return self.drug_catalog_repository.extract_synonym_candidates(value)

    def alias_model_values_for_kind(self, aliases: list[DrugAlias], alias_kind: str) -> set[str]:
        return evidence_aliases.alias_model_values_for_kind(self, aliases, alias_kind)

    def first_alias_model_value(self, aliases: list[DrugAlias], alias_kind: str) -> str | None:
        return evidence_aliases.first_alias_model_value(self, aliases, alias_kind)

    def first_alias_model_term_type(self, aliases: list[DrugAlias]) -> str | None:
        return evidence_aliases.first_alias_model_term_type(self, aliases)

    def group_aliases_by_kind(self, aliases: list[DrugAlias]) -> dict[str, set[str]]:
        return evidence_aliases.group_aliases_by_kind(self, aliases)
