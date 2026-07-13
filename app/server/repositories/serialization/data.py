from __future__ import annotations

from datetime import date
from typing import Any, Iterator

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from repositories.database.session import (
    resolve_engine,
    resolve_session_factory,
)
from repositories.schemas.models import (
    Drug,
    DrugAlias,
    LiverToxMonograph,
    Patient,
)
from repositories.serialization import (
    evidence_aliases,
    evidence_data,
    rxnav_data,
    session_revision_artifacts,
    session_result_data,
    session_timelines,
    session_revision_data,
    session_revision_steps,
)

###############################################################################
class DataSerializer:

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def save_clinical_session(self, session_data: dict[str, Any]) -> int | None:
        return session_result_data.save_clinical_session(self, session_data)

    # -------------------------------------------------------------------------
    def normalize_session_status(self, value: Any) -> str:
        return session_result_data.normalize_session_status(self, value)

    # -------------------------------------------------------------------------
    def persist_patient(
        self, db_session: Session, session_data: dict[str, Any]
    ) -> Patient:
        return session_result_data.persist_patient(self, db_session, session_data)

    # -------------------------------------------------------------------------
    def decode_patient_image(self, value: Any) -> bytes | None:
        return session_result_data.decode_patient_image(self, value)

    # -------------------------------------------------------------------------
    def save_livertox_records(self, records: pd.DataFrame) -> None:
        return evidence_data.save_livertox_records(self, records)

    # -------------------------------------------------------------------------
    def prepare_livertox_rows(self, records: pd.DataFrame) -> list[dict[str, Any]]:
        return evidence_data.prepare_livertox_rows(self, records)

    # -------------------------------------------------------------------------
    def livertox_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return evidence_data.livertox_row_sort_key(self, row)

    # -------------------------------------------------------------------------
    def to_sortable_text(self, value: Any) -> str:
        return evidence_data.to_sortable_text(self, value)

    # -------------------------------------------------------------------------
    def upsert_livertox_monograph(
        self,
        *,
        db_session: Session,
        drug_id: int,
        row: dict[str, Any],
    ) -> None:
        return evidence_data.upsert_livertox_monograph(
            self, db_session=db_session, drug_id=drug_id, row=row
        )

    # -------------------------------------------------------------------------
    def try_assign_livertox_nbk_id(
        self,
        db_session: Session,
        *,
        drug: Drug,
        livertox_nbk_id: str,
    ) -> None:
        return evidence_data.try_assign_livertox_nbk_id(
            self, db_session, drug=drug, livertox_nbk_id=livertox_nbk_id
        )

    # -------------------------------------------------------------------------
    def build_livertox_monograph_key(self, row: dict[str, Any]) -> str:
        return evidence_data.build_livertox_monograph_key(self, row)

    # -------------------------------------------------------------------------
    def upsert_drugs_catalog_records(
        self,
        records: pd.DataFrame | list[dict[str, Any]],
        *,
        commit_interval: int | None = None,
        curated_aliases_by_canonical: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        return rxnav_data.upsert_drugs_catalog_records(
            self,
            records,
            commit_interval=commit_interval,
            curated_aliases_by_canonical=curated_aliases_by_canonical,
        )

    # -------------------------------------------------------------------------
    def resolve_commit_interval(self, override: int | None) -> int:
        return rxnav_data.resolve_commit_interval(self, override)

    # -------------------------------------------------------------------------
    def prepare_rxnav_rows(
        self,
        records: pd.DataFrame | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return rxnav_data.prepare_rxnav_rows(self, records)

    # -------------------------------------------------------------------------
    def prepare_rxnav_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        return rxnav_data.prepare_rxnav_row(self, row)

    # -------------------------------------------------------------------------
    def rxnav_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return rxnav_data.rxnav_row_sort_key(self, row)

    # -------------------------------------------------------------------------
    def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        return rxnav_data.sanitize_livertox_records(self, records)

    # -------------------------------------------------------------------------
    def is_valid_drug_name(self, value: str) -> bool:
        return rxnav_data.is_valid_drug_name(self, value)

    # -------------------------------------------------------------------------
    def get_livertox_records(self) -> pd.DataFrame:
        return evidence_data.get_livertox_records(self)

    # -------------------------------------------------------------------------
    def get_livertox_master_list(self) -> pd.DataFrame:
        return evidence_data.get_livertox_master_list(self)

    # -------------------------------------------------------------------------
    def get_drugs_catalog(self) -> pd.DataFrame:
        return evidence_data.get_drugs_catalog(self)

    # -------------------------------------------------------------------------
    def stream_drugs_catalog(
        self, page_size: int | None = None
    ) -> Iterator[pd.DataFrame]:
        return evidence_data.stream_drugs_catalog(self, page_size)

    # -------------------------------------------------------------------------
    def build_search_pattern(self, search: str | None) -> str | None:
        return evidence_data.build_search_pattern(self, search)

    # -------------------------------------------------------------------------
    def list_sessions(
        self,
        *,
        search: str | None,
        status_filter: str | None,
        date_mode: str | None,
        filter_date: date | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        return session_result_data.list_sessions(
            self,
            search=search,
            status_filter=status_filter,
            date_mode=date_mode,
            filter_date=filter_date,
            offset=offset,
            limit=limit,
        )

    # -------------------------------------------------------------------------
    def parse_session_result_payload(
        self, payload_json: str | None
    ) -> dict[str, Any] | None:
        return session_result_data.parse_session_result_payload(self, payload_json)

    # -------------------------------------------------------------------------
    def get_session_result_payload(self, session_id: int) -> dict[str, Any] | None:
        return session_result_data.get_session_result_payload(self, session_id)

    # -------------------------------------------------------------------------
    def list_session_timelines(self, session_id: int) -> list[dict[str, Any]]:
        return session_timelines.list_session_timelines(self, session_id)

    # -------------------------------------------------------------------------
    def get_session_timeline_record(
        self,
        session_id: int,
        timeline_id: int,
    ) -> dict[str, Any] | None:
        return session_timelines.get_session_timeline_record(
            self,
            session_id,
            timeline_id,
        )

    # -------------------------------------------------------------------------
    def get_latest_session_timeline_record(
        self, session_id: int
    ) -> dict[str, Any] | None:
        return session_timelines.get_latest_session_timeline_record(self, session_id)

    # -------------------------------------------------------------------------
    def create_session_timeline_record(
        self, session_id: int, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        return session_timelines.create_session_timeline_record(
            self, session_id, payload
        )

    # -------------------------------------------------------------------------
    def get_session_detail(self, session_id: int) -> dict[str, Any] | None:
        return session_result_data.get_session_detail(self, session_id)

    # -------------------------------------------------------------------------
    def upsert_session_result_payload(
        self, session_id: int, payload: dict[str, Any]
    ) -> bool:
        return session_result_data.upsert_session_result_payload(
            self, session_id, payload
        )

    # -------------------------------------------------------------------------
    def update_session_text_and_metadata(
        self,
        session_id: int,
        *,
        session_text: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return session_result_data.update_session_text_and_metadata(
            self,
            session_id,
            session_text=session_text,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    def update_current_report_text_with_manual_audit(
        self,
        session_id: int,
        *,
        report_text: str,
        edited_fields: list[str] | None,
        reviewer_note: str | None,
        edited_by: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return session_revision_data.update_current_report_text_with_manual_audit(
            self,
            session_id,
            report_text=report_text,
            edited_fields=edited_fields,
            reviewer_note=reviewer_note,
            edited_by=edited_by,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    def list_manual_report_edits(self, session_id: int) -> list[dict[str, Any]]:
        return session_revision_data.list_manual_report_edits(self, session_id)

    # -------------------------------------------------------------------------
    def update_session_metadata(
        self,
        session_id: int,
        *,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return session_revision_data.update_session_metadata(
            self,
            session_id,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    def list_session_versions(self, session_id: int) -> list[dict[str, Any]]:
        return session_revision_data.list_session_versions(self, session_id)

    # -------------------------------------------------------------------------
    def get_session_version_detail(
        self,
        session_id: int,
        *,
        version_id: int,
    ) -> dict[str, Any] | None:
        return session_revision_data.get_session_version_detail(
            self,
            session_id,
            version_id=version_id,
        )

    # -------------------------------------------------------------------------
    def get_latest_version_record_for_session(
        self,
        session_id: int,
    ) -> dict[str, Any] | None:
        return session_revision_data.get_latest_version_record_for_session(
            self, session_id
        )

    # -------------------------------------------------------------------------
    def get_version_record_for_session(
        self,
        session_id: int,
    ) -> dict[str, Any] | None:
        return session_revision_data.get_version_record_for_session(self, session_id)

    # -------------------------------------------------------------------------
    def create_revision_version_shell(
        self,
        session_id: int,
        *,
        reviewer_note: str | None,
        configuration: dict[str, Any],
        pipeline_run_id: str | None = None,
        initiated_by: str | None = None,
    ) -> dict[str, Any] | None:
        return session_revision_data.create_revision_version_shell(
            self,
            session_id,
            reviewer_note=reviewer_note,
            configuration=configuration,
            pipeline_run_id=pipeline_run_id,
            initiated_by=initiated_by,
        )

    # -------------------------------------------------------------------------
    def create_or_update_revision_run(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return session_revision_data.create_or_update_revision_run(self, **kwargs)

    # -------------------------------------------------------------------------
    def get_revision_run(self, pipeline_run_id: str) -> dict[str, Any] | None:
        return session_revision_data.get_revision_run(self, pipeline_run_id)

    # -------------------------------------------------------------------------
    def get_revision_run_by_job_id(self, job_id: str) -> dict[str, Any] | None:
        return session_revision_data.get_revision_run_by_job_id(self, job_id)

    # -------------------------------------------------------------------------
    def list_revision_steps(self, pipeline_run_id: str) -> list[dict[str, Any]]:
        return session_revision_data.list_revision_steps(self, pipeline_run_id)

    # -------------------------------------------------------------------------
    def fail_revision_run(
        self,
        *,
        pipeline_run_id: str,
        error: dict[str, Any] | None = None,
    ) -> None:
        return session_revision_data.fail_revision_run(
            self,
            pipeline_run_id=pipeline_run_id,
            error=error,
        )

    # -------------------------------------------------------------------------
    def start_revision_step(self, **kwargs: Any) -> dict[str, Any]:
        return session_revision_steps.start_revision_step(self, **kwargs)

    # -------------------------------------------------------------------------
    def complete_revision_step(self, **kwargs: Any) -> dict[str, Any] | None:
        return session_revision_steps.complete_revision_step(self, **kwargs)

    # -------------------------------------------------------------------------
    def fail_revision_step(self, **kwargs: Any) -> dict[str, Any] | None:
        return session_revision_steps.fail_revision_step(self, **kwargs)

    # -------------------------------------------------------------------------
    def persist_revision_agent_issue_scan(
        self,
        *,
        pipeline_run_id: str,
        revision_version_id: int,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return session_revision_artifacts.persist_revision_agent_issue_scan(
            self,
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            payload=payload,
        )

    # -------------------------------------------------------------------------
    def list_revision_artifacts_for_version(
        self,
        *,
        revision_version_id: int,
    ) -> list[dict[str, Any]]:
        return session_revision_artifacts.list_revision_artifacts_for_version(
            self,
            revision_version_id=revision_version_id,
        )

    # -------------------------------------------------------------------------
    def list_revision_entities_for_version(
        self,
        *,
        revision_version_id: int,
    ) -> list[dict[str, Any]]:
        return session_revision_artifacts.list_revision_entities_for_version(
            self,
            revision_version_id=revision_version_id,
        )

    # -------------------------------------------------------------------------
    def list_revision_reviews_for_version(
        self,
        *,
        revision_version_id: int,
    ) -> list[dict[str, Any]]:
        return session_revision_steps.list_revision_reviews_for_version(
            self,
            revision_version_id=revision_version_id,
        )

    # -------------------------------------------------------------------------
    def record_revision_review_action(self, **kwargs: Any) -> dict[str, Any] | None:
        return session_revision_steps.record_revision_review_action(self, **kwargs)

    # -------------------------------------------------------------------------
    def get_next_session_version(self, original_session_id: int) -> int:
        return session_result_data.get_next_session_version(self, original_session_id)

    # -------------------------------------------------------------------------
    def get_session_timeline_source(self, session_id: int) -> dict[str, Any] | None:
        return session_timelines.get_session_timeline_source(self, session_id)

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> bool:
        return session_result_data.delete_session(self, session_id)

    # -------------------------------------------------------------------------
    def list_rxnav_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        return evidence_data.list_rxnav_catalog(
            self, search=search, offset=offset, limit=limit
        )

    # -------------------------------------------------------------------------
    def get_rxnav_alias_groups(self, drug_id: int) -> dict[str, Any] | None:
        return evidence_data.get_rxnav_alias_groups(self, drug_id)

    # -------------------------------------------------------------------------
    def update_rxnav_drug_name(
        self,
        drug_id: int,
        *,
        drug_name: str,
    ) -> dict[str, Any] | None:
        return evidence_data.update_rxnav_drug_name(
            self,
            drug_id,
            drug_name=drug_name,
        )

    # -------------------------------------------------------------------------
    def list_livertox_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        return evidence_data.list_livertox_catalog(
            self, search=search, offset=offset, limit=limit
        )

    # -------------------------------------------------------------------------
    def get_livertox_excerpt(self, drug_id: int) -> dict[str, Any] | None:
        return evidence_data.get_livertox_excerpt(self, drug_id)

    # -------------------------------------------------------------------------
    def get_drug_knowledge_bundle(self, drug_id: int) -> dict[str, Any]:
        return evidence_data.get_drug_knowledge_bundle(self, drug_id)

    # -------------------------------------------------------------------------
    def delete_drug_with_cleanup(self, drug_id: int) -> bool:
        return evidence_data.delete_drug_with_cleanup(self, drug_id)

    # -------------------------------------------------------------------------
    def normalize_string(self, value: Any) -> str | None:
        return session_result_data.normalize_string(self, value)

    # -------------------------------------------------------------------------
    def normalize_flag(self, value: Any) -> int | None:
        return session_result_data.normalize_flag(self, value)

    # -------------------------------------------------------------------------
    def normalize_date(self, value: Any) -> str | None:
        return session_result_data.normalize_date(self, value)

    # -------------------------------------------------------------------------
    def normalize_date_value(self, value: Any) -> date | None:
        return session_result_data.normalize_date_value(self, value)

    # -------------------------------------------------------------------------
    def join_values(self, values: set[str]) -> str | None:
        return session_result_data.join_values(self, values)

    # -------------------------------------------------------------------------
    def to_int(self, value: Any) -> int | None:
        return session_result_data.to_int(self, value)

    # -------------------------------------------------------------------------
    def to_float(self, value: Any) -> float | None:
        return session_result_data.to_float(self, value)

    # -------------------------------------------------------------------------
    def parse_datetime(self, value: Any) -> Any:
        return session_result_data.parse_datetime(self, value)

    # -------------------------------------------------------------------------
    def persist_session_sections(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        return session_result_data.persist_session_sections(
            self, db_session, session_id, session_data
        )

    # -------------------------------------------------------------------------
    def persist_session_labs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        return session_result_data.persist_session_labs(
            self, db_session, session_id, session_data
        )

    # -------------------------------------------------------------------------
    def persist_session_drugs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        return session_result_data.persist_session_drugs(
            self, db_session, session_id, session_data
        )

    # -------------------------------------------------------------------------
    def resolve_drug_id_from_match_cache(
        self,
        db_session: Session,
        *,
        normalized_drug_key: str,
    ) -> int | None:
        return evidence_data.resolve_drug_id_from_match_cache(
            self, db_session, normalized_drug_key=normalized_drug_key
        )

    # -------------------------------------------------------------------------
    def upsert_high_confidence_kb_match_cache(
        self,
        db_session: Session,
        *,
        raw_drug_name: str,
        raw_drug_name_norm: str,
        normalized_drug_key: str,
        drug_id: int | None,
        rxnorm_rxcui: str | None,
        livertox_nbk_id: str | None,
        source: str,
        confidence: float | None,
        evidence: dict[str, Any],
        ambiguous: bool,
    ) -> None:
        return evidence_data.upsert_high_confidence_kb_match_cache(
            self,
            db_session,
            raw_drug_name=raw_drug_name,
            raw_drug_name_norm=raw_drug_name_norm,
            normalized_drug_key=normalized_drug_key,
            drug_id=drug_id,
            rxnorm_rxcui=rxnorm_rxcui,
            livertox_nbk_id=livertox_nbk_id,
            source=source,
            confidence=confidence,
            evidence=evidence,
            ambiguous=ambiguous,
        )

    # -------------------------------------------------------------------------
    def load_livertox_match_from_db_cache(
        self,
        *,
        normalized_drug_key: str,
    ) -> dict[str, Any] | None:
        return evidence_data.load_livertox_match_from_db_cache(
            self,
            normalized_drug_key=normalized_drug_key,
        )

    # -------------------------------------------------------------------------
    def persist_session_result_payload(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        return session_result_data.persist_session_result_payload(
            self, db_session, session_id, session_data
        )

    # -------------------------------------------------------------------------
    def serialize_json_payload(self, payload: Any) -> str | None:
        return session_result_data.serialize_json_payload(self, payload)

    # -------------------------------------------------------------------------
    def resolve_drug_id(
        self,
        db_session: Session,
        *,
        matched_drug_name: str | None,
        rxcui: str | None,
        nbk_id: str | None,
    ) -> int | None:
        return evidence_aliases.resolve_drug_id(
            self,
            db_session,
            matched_drug_name=matched_drug_name,
            rxcui=rxcui,
            nbk_id=nbk_id,
        )

    # -------------------------------------------------------------------------
    def ensure_drug(
        self,
        db_session: Session,
        *,
        canonical_name: str,
        canonical_name_norm: str,
        rxnorm_rxcui: str | None,
        livertox_nbk_id: str | None,
        rxnav_last_update: str | None = None,
        use_livertox_nbk_lookup: bool = True,
    ) -> Drug:
        return evidence_aliases.ensure_drug(
            self,
            db_session,
            canonical_name=canonical_name,
            canonical_name_norm=canonical_name_norm,
            rxnorm_rxcui=rxnorm_rxcui,
            livertox_nbk_id=livertox_nbk_id,
            rxnav_last_update=rxnav_last_update,
            use_livertox_nbk_lookup=use_livertox_nbk_lookup,
        )

    # -------------------------------------------------------------------------
    def assign_identifier_if_consistent(
        self,
        *,
        drug: Drug,
        field_name: str,
        incoming_value: str | None,
    ) -> None:
        return evidence_aliases.assign_identifier_if_consistent(
            self, drug=drug, field_name=field_name, incoming_value=incoming_value
        )

    # -------------------------------------------------------------------------
    def upsert_drug_rxcui(
        self,
        db_session: Session,
        *,
        drug_id: int,
        rxcui: str | None,
    ) -> None:
        return evidence_aliases.upsert_drug_rxcui(
            self, db_session, drug_id=drug_id, rxcui=rxcui
        )

    # -------------------------------------------------------------------------
    def get_drug_by_rxcui(
        self,
        db_session: Session,
        rxcui: str | None,
    ) -> Drug | None:
        return evidence_aliases.get_drug_by_rxcui(self, db_session, rxcui)

    # -------------------------------------------------------------------------
    def get_drug_by_canonical_name_norm(
        self,
        db_session: Session,
        canonical_name_norm: str | None,
    ) -> Drug | None:
        return evidence_aliases.get_drug_by_canonical_name_norm(
            self, db_session, canonical_name_norm
        )

    # -------------------------------------------------------------------------
    def get_drug_alias_by_norm(
        self,
        db_session: Session,
        alias_norm: str | None,
    ) -> DrugAlias | None:
        return evidence_aliases.get_drug_alias_by_norm(self, db_session, alias_norm)

    # -------------------------------------------------------------------------
    def get_monograph_by_drug_id(
        self,
        db_session: Session,
        drug_id: int,
    ) -> LiverToxMonograph | None:
        return evidence_aliases.get_monograph_by_drug_id(self, db_session, drug_id)

    # -------------------------------------------------------------------------
    def get_monograph_by_key(
        self,
        db_session: Session,
        monograph_key: str,
    ) -> LiverToxMonograph | None:
        return evidence_aliases.get_monograph_by_key(self, db_session, monograph_key)

    # -------------------------------------------------------------------------
    def upsert_drug_alias(
        self,
        db_session: Session,
        *,
        drug_id: int,
        alias: str,
        alias_kind: str,
        source: str,
        term_type: str | None,
    ) -> None:
        return evidence_aliases.upsert_drug_alias(
            self,
            db_session,
            drug_id=drug_id,
            alias=alias,
            alias_kind=alias_kind,
            source=source,
            term_type=term_type,
        )

    # -------------------------------------------------------------------------
    def persist_livertox_aliases(
        self, db_session: Session, drug_id: int, row: dict[str, Any]
    ) -> None:
        return evidence_aliases.persist_livertox_aliases(self, db_session, drug_id, row)

    # -------------------------------------------------------------------------
    def extract_text_candidates(self, value: Any) -> list[str]:
        return evidence_aliases.extract_text_candidates(self, value)

    # -------------------------------------------------------------------------
    def extract_synonym_candidates(self, value: Any) -> list[str]:
        return evidence_aliases.extract_synonym_candidates(self, value)

    # -------------------------------------------------------------------------
    def unique_text(self, values: list[str]) -> list[str]:
        return evidence_aliases.unique_text(self, values)

    # -------------------------------------------------------------------------
    def build_alias_lookup_by_kind(
        self, aliases_frame: pd.DataFrame
    ) -> dict[int, dict[str, set[str]]]:
        return evidence_aliases.build_alias_lookup_by_kind(self, aliases_frame)

    # -------------------------------------------------------------------------
    def group_aliases_by_kind(self, aliases: list[DrugAlias]) -> dict[str, set[str]]:
        return evidence_aliases.group_aliases_by_kind(self, aliases)

    # -------------------------------------------------------------------------
    def alias_values_for_kind(self, aliases: pd.DataFrame, alias_kind: str) -> set[str]:
        return evidence_aliases.alias_values_for_kind(self, aliases, alias_kind)

    # -------------------------------------------------------------------------
    def alias_model_values_for_kind(
        self,
        aliases: list[DrugAlias],
        alias_kind: str,
    ) -> set[str]:
        return evidence_aliases.alias_model_values_for_kind(self, aliases, alias_kind)

    # -------------------------------------------------------------------------
    def first_alias_value(self, aliases: pd.DataFrame, alias_kind: str) -> str | None:
        return evidence_aliases.first_alias_value(self, aliases, alias_kind)

    # -------------------------------------------------------------------------
    def first_alias_term_type(self, aliases: pd.DataFrame) -> str | None:
        return evidence_aliases.first_alias_term_type(self, aliases)

    # -------------------------------------------------------------------------
    def first_alias_model_value(
        self,
        aliases: list[DrugAlias],
        alias_kind: str,
    ) -> str | None:
        return evidence_aliases.first_alias_model_value(self, aliases, alias_kind)

    # -------------------------------------------------------------------------
    def first_alias_model_term_type(self, aliases: list[DrugAlias]) -> str | None:
        return evidence_aliases.first_alias_model_term_type(self, aliases)

    # -------------------------------------------------------------------------
    def persist_revision_artifact(self, **kwargs: Any) -> list[dict[str, Any]]:
        return session_revision_artifacts.persist_revision_artifact(self, **kwargs)

    # -------------------------------------------------------------------------
    def finalize_revision_version(self, **kwargs: Any) -> dict[str, Any] | None:
        return session_revision_data.finalize_revision_version(self, **kwargs)

    # -------------------------------------------------------------------------
    def persist_revision_entities(self, **kwargs: Any) -> list[dict[str, Any]]:
        return session_revision_artifacts.persist_revision_entities(self, **kwargs)
