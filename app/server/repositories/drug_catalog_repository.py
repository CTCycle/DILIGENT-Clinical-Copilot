from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from repositories.context import RepositoryContext
from repositories import values as repository_values
from repositories.schemas.knowledge import Drug, DrugAlias
from repositories.serialization import evidence_aliases, evidence_data, rxnav_data


class DrugCatalogRepository:
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    def upsert_drugs_catalog_records(self, records: pd.DataFrame | list[dict[str, Any]], **kwargs: Any) -> None:
        rxnav_data.upsert_drugs_catalog_records(self, records, **kwargs)

    def list_rxnav_catalog(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        return evidence_data.list_rxnav_catalog(self, **kwargs)

    def get_rxnav_alias_groups(self, drug_id: int) -> dict[str, Any] | None:
        return evidence_data.get_rxnav_alias_groups(self, drug_id)

    def update_rxnav_drug_name(self, drug_id: int, *, drug_name: str) -> dict[str, Any] | None:
        return evidence_data.update_rxnav_drug_name(self, drug_id, drug_name=drug_name)

    def delete_drug_with_cleanup(self, drug_id: int) -> bool:
        return evidence_data.delete_drug_with_cleanup(self, drug_id)

    def resolve_drug_id(self, db_session: Session, **kwargs: Any) -> int | None:
        return evidence_aliases.resolve_drug_id(self, db_session, **kwargs)

    def ensure_drug(self, db_session: Session, **kwargs: Any) -> Drug:
        return evidence_aliases.ensure_drug(self, db_session, **kwargs)

    def assign_identifier_if_consistent(self, **kwargs: Any) -> None:
        evidence_aliases.assign_identifier_if_consistent(self, **kwargs)

    def upsert_drug_rxcui(self, db_session: Session, **kwargs: Any) -> None:
        evidence_aliases.upsert_drug_rxcui(self, db_session, **kwargs)

    def get_drug_by_rxcui(self, db_session: Session, rxcui: str | None) -> Drug | None:
        return evidence_aliases.get_drug_by_rxcui(self, db_session, rxcui)

    def get_drug_by_canonical_name_norm(self, db_session: Session, name: str | None) -> Drug | None:
        return evidence_aliases.get_drug_by_canonical_name_norm(self, db_session, name)

    def get_drug_alias_by_norm(self, db_session: Session, alias_norm: str | None) -> list[DrugAlias]:
        return evidence_aliases.get_drug_alias_by_norm(self, db_session, alias_norm)

    def upsert_drug_alias(self, db_session: Session, **kwargs: Any) -> None:
        evidence_aliases.upsert_drug_alias(self, db_session, **kwargs)

    def persist_livertox_aliases(self, db_session: Session, drug_id: int, row: dict[str, Any]) -> None:
        evidence_aliases.persist_livertox_aliases(self, db_session, drug_id, row)

    def extract_text_candidates(self, value: Any) -> list[str]:
        return evidence_aliases.extract_text_candidates(self, value)

    def extract_synonym_candidates(self, value: Any) -> list[str]:
        return evidence_aliases.extract_synonym_candidates(self, value)

    def unique_text(self, values: list[str]) -> list[str]:
        return evidence_aliases.unique_text(self, values)

    def normalize_string(self, value: Any) -> str | None:
        return repository_values.normalize_string(value)

    def normalize_date(self, value: Any) -> str | None:
        return repository_values.normalize_date(value)

    def normalize_date_value(self, value: Any):
        return repository_values.normalize_date_value(value)

    def normalize_flag(self, value: Any) -> int | None:
        return repository_values.normalize_flag(value)

    def to_int(self, value: Any) -> int | None:
        return repository_values.to_int(value)

    def join_values(self, values: set[str]) -> str | None:
        return repository_values.join_values(values)

    def build_search_pattern(self, search: str | None) -> str | None:
        return evidence_data.build_search_pattern(self, search)

    def try_assign_livertox_nbk_id(self, db_session: Session, **kwargs: Any) -> None:
        evidence_data.try_assign_livertox_nbk_id(self, db_session, **kwargs)

    def is_valid_drug_name(self, value: str) -> bool:
        return rxnav_data.is_valid_drug_name(self, value)

    def prepare_rxnav_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        return rxnav_data.prepare_rxnav_row(self, row)

    def prepare_rxnav_rows(self, records: pd.DataFrame | list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rxnav_data.prepare_rxnav_rows(self, records)

    def rxnav_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return rxnav_data.rxnav_row_sort_key(self, row)

    def to_sortable_text(self, value: Any) -> str:
        return evidence_data.to_sortable_text(self, value)
