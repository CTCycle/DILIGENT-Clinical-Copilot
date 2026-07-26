from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from repositories.context import RepositoryContext
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository
from repositories.serialization import evidence_data, session_result_data, session_revision_data


class ClinicalSessionRepository:
    def __init__(self, context: RepositoryContext, drug_catalog_repository: DrugCatalogRepository, knowledge_repository: KnowledgeRepository) -> None:
        self.context = context
        self.drug_catalog_repository = drug_catalog_repository
        self.knowledge_repository = knowledge_repository
        self.engine = context.engine
        self.session_factory = context.session_factory
        self._vocabulary_changed = False

    def save_clinical_session(self, session_data: dict[str, Any]) -> int | None:
        return session_result_data.save_clinical_session(self, session_data)

    def list_sessions(self, **kwargs: Any):
        return session_result_data.list_sessions(self, **kwargs)

    def get_session_detail(self, session_id: int):
        return session_result_data.get_session_detail(self, session_id)

    def get_session_result_payload(self, session_id: int):
        return session_result_data.get_session_result_payload(self, session_id)

    def upsert_session_result_payload(self, session_id: int, payload: dict[str, Any]) -> bool:
        return session_result_data.upsert_session_result_payload(self, session_id, payload)

    def update_session_text_and_metadata(self, session_id: int, **kwargs: Any):
        return session_result_data.update_session_text_and_metadata(self, session_id, **kwargs)

    def delete_session(self, session_id: int) -> bool:
        return session_result_data.delete_session(self, session_id)

    def get_next_session_version(self, root_session_id: int) -> int:
        return session_result_data.get_next_session_version(self, root_session_id)

    def update_session_metadata(self, session_id: int, **kwargs: Any):
        return session_revision_data.update_session_metadata(self, session_id, **kwargs)

    def persist_session_sections(self, db_session: Session, session_id: int, data: dict[str, Any]) -> None:
        session_result_data.persist_session_sections(self, db_session, session_id, data)

    def persist_session_labs(self, db_session: Session, session_id: int, data: dict[str, Any]) -> None:
        session_result_data.persist_session_labs(self, db_session, session_id, data)

    def persist_session_drugs(self, db_session: Session, session_id: int, data: dict[str, Any]) -> bool:
        return session_result_data.persist_session_drugs(self, db_session, session_id, data)

    def persist_session_result_payload(self, db_session: Session, session_id: int, data: dict[str, Any]) -> None:
        session_result_data.persist_session_result_payload(self, db_session, session_id, data)

    def consume_vocabulary_change_signal(self) -> bool:
        changed = self._vocabulary_changed
        self._vocabulary_changed = False
        return changed

    def resolve_drug_id(self, db_session: Session, **kwargs: Any):
        return self.drug_catalog_repository.resolve_drug_id(db_session, **kwargs)

    def resolve_drug_id_from_match_cache(self, db_session: Session, **kwargs: Any):
        return self.knowledge_repository.resolve_drug_id_from_match_cache(db_session, **kwargs)

    def upsert_high_confidence_kb_match_cache(self, db_session: Session, **kwargs: Any) -> None:
        self.knowledge_repository.upsert_high_confidence_kb_match_cache(db_session, **kwargs)

    def upsert_drug_alias(self, db_session: Session, **kwargs: Any) -> None:
        self.drug_catalog_repository.upsert_drug_alias(db_session, **kwargs)

    def normalize_session_status(self, value: Any) -> str:
        return session_result_data.normalize_session_status(self, value)

    def decode_patient_image(self, value: Any) -> bytes | None:
        return session_result_data.decode_patient_image(self, value)

    def normalize_string(self, value: Any) -> str | None:
        return session_result_data.normalize_string(self, value)

    def normalize_date_value(self, value: Any) -> date | None:
        return session_result_data.normalize_date_value(self, value)

    def normalize_date(self, value: Any) -> str | None:
        return session_result_data.normalize_date(self, value)

    def normalize_flag(self, value: Any) -> int | None:
        return session_result_data.normalize_flag(self, value)

    def join_values(self, values: set[str]) -> str | None:
        return session_result_data.join_values(self, values)

    def to_int(self, value: Any) -> int | None:
        return session_result_data.to_int(self, value)

    def to_float(self, value: Any) -> float | None:
        return session_result_data.to_float(self, value)

    def parse_datetime(self, value: Any) -> Any:
        return session_result_data.parse_datetime(self, value)

    def parse_session_result_payload(self, value: str | None):
        return session_result_data.parse_session_result_payload(self, value)

    def serialize_json_payload(self, value: Any) -> str | None:
        return session_result_data.serialize_json_payload(self, value)

    def build_search_pattern(self, value: str | None) -> str | None:
        return evidence_data.build_search_pattern(self, value)
