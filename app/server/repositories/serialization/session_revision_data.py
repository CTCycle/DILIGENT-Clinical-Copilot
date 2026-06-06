from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from domain.clinical.revision import (
    RevisedDiliAssessment,
    RevisedDiseasePayload,
    RevisedDrugPayload,
    RevisedLabPayload,
    RevisionLiverToxDecision,
)
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from repositories.schemas.models import (
    ClinicalSession,
    ClinicalSessionRevisionArtifact,
    ClinicalSessionRevisionEntity,
    ClinicalSessionRevisionReview,
    ClinicalSessionManualEdit,
    ClinicalSessionRevisionRun,
    ClinicalSessionRevisionStep,
    ClinicalSessionResult,
    ClinicalSessionVersion,
    Patient,
)


def build_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_payload_hash(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_text_key(value: str | None) -> str | None:
    cleaned = str(value or "").strip().casefold()
    return cleaned or None


def default_version_status(*, is_latest: bool) -> str:
    return "current" if is_latest else "superseded"


def sync_preserved_version_status(
    existing_status: str | None,
    *,
    is_latest_completed: bool,
) -> str:
    normalized = str(existing_status or "").strip()
    if normalized in {"", "current", "superseded"}:
        return default_version_status(is_latest=is_latest_completed)
    return normalized


def derive_revision_kind(session_row: ClinicalSession, root_session_id: int) -> str:
    if int(session_row.id) == int(root_session_id) and int(session_row.version or 1) == 1:
        return "original"
    return "llm_assisted_revision"


REVISION_DRUG_SCHEMA_NAME = "revised_drug_entry"
REVISION_DISEASE_SCHEMA_NAME = "revised_disease_entry"
REVISION_LAB_SCHEMA_NAME = "revised_lab_entry"
REVISION_LIVERTOX_DECISION_SCHEMA_NAME = "revision_livertox_decision"
REVISION_DILI_ASSESSMENT_SCHEMA_NAME = "revised_dili_assessment"
REVISION_ENTITY_SCHEMA_VERSION = "1"


def validate_revised_drug_payload(payload: Any) -> RevisedDrugPayload:
    return RevisedDrugPayload.model_validate(payload)


def validate_revised_disease_payload(payload: Any) -> RevisedDiseasePayload:
    return RevisedDiseasePayload.model_validate(payload)


def validate_revised_lab_payload(payload: Any) -> RevisedLabPayload:
    return RevisedLabPayload.model_validate(payload)


def validate_revision_livertox_decision(payload: Any) -> RevisionLiverToxDecision:
    return RevisionLiverToxDecision.model_validate(payload)


def validate_revised_dili_assessment(payload: Any) -> RevisedDiliAssessment:
    return RevisedDiliAssessment.model_validate(payload)


def serialize_version_row(
    self,
    row: ClinicalSessionVersion,
) -> dict[str, Any]:
    model_configuration = self.parse_session_result_payload(row.model_configuration_json)
    return {
        "version_id": int(row.id),
        "session_id": int(row.session_id) if row.session_id is not None else None,
        "root_session_id": int(row.root_session_id),
        "source_version_id": (
            int(row.source_version_id) if row.source_version_id is not None else None
        ),
        "revision_version_id": int(row.id),
        "version_number": int(row.version_number),
        "version_status": row.version_status,
        "revision_kind": row.revision_kind,
        "llm_qa_status": row.llm_qa_status,
        "clinical_review_status": row.clinical_review_status,
        "pipeline_run_id": self.normalize_string(row.pipeline_run_id),
        "model_configuration": (
            model_configuration if isinstance(model_configuration, dict) else {}
        ),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "completed_at": row.completed_at,
    }


def serialize_revision_run_row(
    self,
    row: ClinicalSessionRevisionRun,
) -> dict[str, Any]:
    return {
        "pipeline_run_id": row.pipeline_run_id,
        "session_id": int(row.session_id),
        "root_session_id": int(row.root_session_id),
        "source_version_id": int(row.source_version_id),
        "target_revision_version_id": (
            int(row.target_revision_version_id)
            if row.target_revision_version_id is not None
            else None
        ),
        "revision_mode": row.revision_mode,
        "revision_kind": row.revision_kind,
        "configuration": self.parse_session_result_payload(row.configuration_json) or {},
        "reviewer_note": self.normalize_string(row.reviewer_note),
        "initiated_by": self.normalize_string(row.initiated_by),
        "actor_id": self.normalize_string(row.actor_id),
        "actor_display_name": self.normalize_string(row.actor_display_name),
        "actor_source": row.actor_source,
        "actor_confidence": row.actor_confidence,
        "started_at": row.started_at,
        "completed_at": row.completed_at,
        "status": row.status,
        "error": self.parse_session_result_payload(row.error_json),
        "token_usage": self.parse_session_result_payload(row.token_usage_json),
        "latency_ms": int(row.latency_ms) if row.latency_ms is not None else None,
        "cost_estimate": float(row.cost_estimate)
        if row.cost_estimate is not None
        else None,
        "trace_id": self.normalize_string(row.trace_id),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def serialize_revision_step_row(
    self,
    row: ClinicalSessionRevisionStep,
) -> dict[str, Any]:
    return {
        "pipeline_run_id": row.pipeline_run_id,
        "step_name": row.step_name,
        "step_index": int(row.step_index),
        "step_count": int(row.step_count),
        "attempt_number": int(row.attempt_number),
        "status": row.status,
        "input_hash": self.normalize_string(row.input_hash),
        "output_hash": self.normalize_string(row.output_hash),
        "input_summary": self.parse_session_result_payload(row.input_summary_json),
        "output_summary": self.parse_session_result_payload(row.output_summary_json),
        "output_payload": self.parse_session_result_payload(row.output_payload_json),
        "schema_name": self.normalize_string(row.schema_name),
        "schema_version": self.normalize_string(row.schema_version),
        "prompt_version": self.normalize_string(row.prompt_version),
        "parser_version": self.normalize_string(row.parser_version),
        "model_provider": self.normalize_string(row.model_provider),
        "model_name": self.normalize_string(row.model_name),
        "token_usage": self.parse_session_result_payload(row.token_usage_json),
        "latency_ms": int(row.latency_ms) if row.latency_ms is not None else None,
        "retry_count": int(row.retry_count),
        "error": self.parse_session_result_payload(row.error_json),
        "started_at": row.started_at,
        "completed_at": row.completed_at,
        "superseded_at": row.superseded_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def serialize_revision_artifact_row(
    self,
    row: ClinicalSessionRevisionArtifact,
) -> dict[str, Any]:
    return {
        "revision_version_id": int(row.revision_version_id),
        "pipeline_run_id": row.pipeline_run_id,
        "artifact_kind": row.artifact_kind,
        "artifact_key": self.normalize_string(row.artifact_key),
        "entity_type": self.normalize_string(row.entity_type),
        "entity_name": self.normalize_string(row.entity_name),
        "status": self.normalize_string(row.status),
        "schema_version": self.normalize_string(row.schema_version),
        "payload": self.parse_session_result_payload(row.payload_json),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def serialize_revision_review_row(
    self,
    row: ClinicalSessionRevisionReview,
) -> dict[str, Any]:
    return {
        "revision_version_id": int(row.revision_version_id),
        "session_id": int(row.session_id) if row.session_id is not None else None,
        "clinical_review_status": row.clinical_review_status,
        "reviewer_note": self.normalize_string(row.reviewer_note),
        "reviewed_by": self.normalize_string(row.reviewed_by),
        "actor_id": self.normalize_string(row.actor_id),
        "actor_display_name": self.normalize_string(row.actor_display_name),
        "actor_source": row.actor_source,
        "actor_confidence": row.actor_confidence,
        "metadata": self.parse_session_result_payload(row.metadata_json) or {},
        "reviewed_at": row.reviewed_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _create_revision_artifact_row(
    self,
    *,
    revision_version_id: int,
    pipeline_run_id: str,
    artifact_kind: str,
    artifact_key: str | None = None,
    entity_type: str | None = None,
    entity_name: str | None = None,
    status: str | None = None,
    schema_version: str | None = "1",
    payload: dict[str, Any] | None = None,
) -> ClinicalSessionRevisionArtifact:
    return ClinicalSessionRevisionArtifact(
        revision_version_id=revision_version_id,
        pipeline_run_id=pipeline_run_id,
        artifact_kind=artifact_kind,
        artifact_key=self.normalize_string(artifact_key),
        entity_type=self.normalize_string(entity_type),
        entity_name=self.normalize_string(entity_name),
        status=self.normalize_string(status),
        schema_version=self.normalize_string(schema_version),
        payload_json=self.serialize_json_payload(payload),
    )


def _create_revision_entity_row(
    self,
    *,
    revision_version_id: int,
    source_version_id: int | None,
    pipeline_run_id: str,
    step_name: str,
    entity_type: str,
    entity_revision_status: str = "active",
    source_section: str,
    original_entity_id: str | None,
    original_name: str | None,
    revised_name: str | None,
    normalized_name: str | None,
    requires_human_review: bool,
    payload: dict[str, Any],
    schema_name: str,
) -> ClinicalSessionRevisionEntity:
    return ClinicalSessionRevisionEntity(
        revision_version_id=revision_version_id,
        source_version_id=source_version_id,
        pipeline_run_id=pipeline_run_id,
        step_name=step_name,
        entity_type=entity_type,
        entity_revision_status=entity_revision_status,
        source_section=self.normalize_string(source_section),
        original_entity_id=self.normalize_string(original_entity_id),
        original_name=self.normalize_string(original_name),
        revised_name=self.normalize_string(revised_name),
        normalized_name=self.normalize_string(normalized_name),
        requires_human_review=bool(requires_human_review),
        human_review_status=("required" if requires_human_review else "not_required"),
        payload_json=self.serialize_json_payload(payload),
        schema_name=schema_name,
        schema_version=REVISION_ENTITY_SCHEMA_VERSION,
        prompt_version=None,
        parser_version=None,
        model_provider=None,
        model_name=None,
        input_hash=None,
        output_hash=build_payload_hash(payload),
        superseded_at=None,
    )


def serialize_revision_entity_row(
    self,
    row: ClinicalSessionRevisionEntity,
) -> dict[str, Any]:
    return {
        "revision_version_id": int(row.revision_version_id),
        "source_version_id": (
            int(row.source_version_id) if row.source_version_id is not None else None
        ),
        "pipeline_run_id": row.pipeline_run_id,
        "step_name": row.step_name,
        "entity_type": row.entity_type,
        "entity_revision_status": row.entity_revision_status,
        "source_section": self.normalize_string(row.source_section),
        "original_entity_id": self.normalize_string(row.original_entity_id),
        "original_name": self.normalize_string(row.original_name),
        "revised_name": self.normalize_string(row.revised_name),
        "normalized_name": self.normalize_string(row.normalized_name),
        "requires_human_review": bool(row.requires_human_review),
        "human_review_status": self.normalize_string(row.human_review_status),
        "payload": self.parse_session_result_payload(row.payload_json),
        "schema_name": self.normalize_string(row.schema_name),
        "schema_version": self.normalize_string(row.schema_version),
        "prompt_version": self.normalize_string(row.prompt_version),
        "parser_version": self.normalize_string(row.parser_version),
        "model_provider": self.normalize_string(row.model_provider),
        "model_name": self.normalize_string(row.model_name),
        "input_hash": self.normalize_string(row.input_hash),
        "output_hash": self.normalize_string(row.output_hash),
        "created_at": row.created_at,
        "superseded_at": row.superseded_at,
    }


def get_root_session_id_for_session(
    db_session: Session,
    session_id: int,
) -> int | None:
    session_row = db_session.get(ClinicalSession, int(session_id))
    if session_row is None:
        return None
    return int(session_row.original_session_id or session_row.id)


def ensure_version_record_for_session(
    self,
    db_session: Session,
    *,
    session_row: ClinicalSession,
    root_session_id: int,
    source_version_id: int | None,
    is_latest_completed: bool,
) -> ClinicalSessionVersion:
    existing = db_session.execute(
        select(ClinicalSessionVersion).where(
            ClinicalSessionVersion.session_id == int(session_row.id)
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.root_session_id = int(root_session_id)
        existing.source_version_id = source_version_id
        existing.version_number = int(session_row.version or 1)
        existing.version_status = sync_preserved_version_status(
            existing.version_status,
            is_latest_completed=is_latest_completed,
        )
        existing.revision_kind = derive_revision_kind(session_row, root_session_id)
        existing.llm_qa_status = existing.llm_qa_status or "not_run"
        existing.clinical_review_status = existing.clinical_review_status or "not_reviewed"
        return existing

    model_configuration = {
        "text_extraction_model": self.normalize_string(
            session_row.text_extraction_model
        ),
        "clinical_model": self.normalize_string(session_row.clinical_model),
    }
    version_row = ClinicalSessionVersion(
        session_id=int(session_row.id),
        root_session_id=int(root_session_id),
        source_version_id=source_version_id,
        version_number=int(session_row.version or 1),
        version_status=default_version_status(is_latest=is_latest_completed),
        revision_kind=derive_revision_kind(session_row, root_session_id),
        llm_qa_status="not_run",
        clinical_review_status="not_reviewed",
        pipeline_run_id=None,
        model_configuration_json=self.serialize_json_payload(model_configuration),
        completed_at=session_row.session_timestamp,
    )
    db_session.add(version_row)
    db_session.flush()
    return version_row


def sync_version_records_for_root(
    self,
    db_session: Session,
    *,
    root_session_id: int,
) -> list[ClinicalSessionVersion]:
    session_rows = list(
        db_session.execute(
            select(ClinicalSession)
            .where(
                (ClinicalSession.id == int(root_session_id))
                | (ClinicalSession.original_session_id == int(root_session_id))
            )
            .order_by(ClinicalSession.version.asc(), ClinicalSession.id.asc())
        ).scalars()
    )
    latest_completed_session_id = int(session_rows[-1].id) if session_rows else None
    previous_version_id: int | None = None
    synced: list[ClinicalSessionVersion] = []
    for session_row in session_rows:
        version_row = ensure_version_record_for_session(
            self,
            db_session,
            session_row=session_row,
            root_session_id=root_session_id,
            source_version_id=previous_version_id,
            is_latest_completed=latest_completed_session_id == int(session_row.id),
        )
        db_session.flush()
        previous_version_id = int(version_row.id)
        synced.append(version_row)
    return synced


def list_session_versions(self, session_id: int) -> list[dict[str, Any]]:
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        root_session_id = get_root_session_id_for_session(db_session, safe_session_id)
        if root_session_id is None:
            return []
        sync_version_records_for_root(self, db_session, root_session_id=root_session_id)
        db_session.commit()
        rows = list(
            db_session.execute(
                select(ClinicalSessionVersion)
                .where(ClinicalSessionVersion.root_session_id == int(root_session_id))
                .order_by(
                    ClinicalSessionVersion.version_number.asc(),
                    ClinicalSessionVersion.id.asc(),
                )
            ).scalars()
        )
        return [serialize_version_row(self, row) for row in rows]
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def get_session_version_detail(
    self,
    session_id: int,
    *,
    version_id: int,
) -> dict[str, Any] | None:
    safe_session_id = int(session_id)
    safe_version_id = int(version_id)
    db_session = self.session_factory()
    try:
        root_session_id = get_root_session_id_for_session(db_session, safe_session_id)
        if root_session_id is None:
            return None
        sync_version_records_for_root(self, db_session, root_session_id=root_session_id)
        db_session.commit()
        row = db_session.get(ClinicalSessionVersion, safe_version_id)
        if row is None or int(row.root_session_id) != int(root_session_id):
            return None
        return {
            "version": serialize_version_row(self, row),
            "session": (
                self.get_session_detail(int(row.session_id))
                if row.session_id is not None
                else None
            ),
        }
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def get_latest_version_record_for_session(
    self,
    session_id: int,
) -> dict[str, Any] | None:
    versions = list_session_versions(self, session_id)
    if not versions:
        return None
    return versions[-1]


def get_version_record_for_session(
    self,
    session_id: int,
) -> dict[str, Any] | None:
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        root_session_id = get_root_session_id_for_session(db_session, safe_session_id)
        if root_session_id is None:
            return None
        sync_version_records_for_root(self, db_session, root_session_id=root_session_id)
        db_session.commit()
        row = db_session.execute(
            select(ClinicalSessionVersion).where(
                ClinicalSessionVersion.session_id == safe_session_id
            )
        ).scalar_one_or_none()
        return None if row is None else serialize_version_row(self, row)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def create_revision_version_shell(
    self,
    session_id: int,
    *,
    reviewer_note: str | None,
    configuration: dict[str, Any],
    pipeline_run_id: str | None = None,
    initiated_by: str | None = None,
) -> dict[str, Any] | None:
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        root_session_id = get_root_session_id_for_session(db_session, safe_session_id)
        if root_session_id is None:
            return None
        synced = sync_version_records_for_root(
            self, db_session, root_session_id=root_session_id
        )
        source_version = next(
            (row for row in synced if row.session_id == safe_session_id),
            None,
        )
        if source_version is None:
            return None
        next_version_number = max(int(row.version_number) for row in synced) + 1
        run_id = pipeline_run_id or uuid.uuid4().hex
        existing = db_session.execute(
            select(ClinicalSessionVersion).where(
                ClinicalSessionVersion.pipeline_run_id == run_id
            )
        ).scalar_one_or_none()
        if existing is not None:
            return serialize_version_row(self, existing)
        shell = ClinicalSessionVersion(
            session_id=None,
            root_session_id=int(root_session_id),
            source_version_id=int(source_version.id),
            version_number=next_version_number,
            version_status="draft_revision",
            revision_kind="llm_assisted_revision",
            llm_qa_status="pending",
            clinical_review_status="not_reviewed",
            pipeline_run_id=run_id,
            model_configuration_json=self.serialize_json_payload(configuration),
            completed_at=None,
        )
        db_session.add(shell)
        db_session.flush()
        db_session.commit()
        return serialize_version_row(self, shell)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def finalize_revision_version(
    self,
    *,
    pipeline_run_id: str,
    persisted_session_id: int,
    model_configuration: dict[str, Any] | None = None,
    version_status: str = "requires_human_review",
    llm_qa_status: str = "not_run",
    clinical_review_status: str = "not_reviewed",
) -> dict[str, Any] | None:
    db_session = self.session_factory()
    try:
        version_row = db_session.execute(
            select(ClinicalSessionVersion).where(
                ClinicalSessionVersion.pipeline_run_id == str(pipeline_run_id)
            )
        ).scalar_one_or_none()
        if version_row is None:
            return None
        version_row.session_id = int(persisted_session_id)
        version_row.version_status = version_status
        version_row.llm_qa_status = llm_qa_status
        version_row.clinical_review_status = clinical_review_status
        version_row.completed_at = datetime.now(UTC)
        if model_configuration:
            version_row.model_configuration_json = self.serialize_json_payload(
                model_configuration
            )
        db_session.commit()
        return serialize_version_row(self, version_row)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def create_or_update_revision_run(
    self,
    *,
    pipeline_run_id: str,
    session_id: int,
    root_session_id: int,
    source_version_id: int,
    target_revision_version_id: int | None,
    revision_mode: str,
    revision_kind: str,
    configuration: dict[str, Any],
    reviewer_note: str | None,
    status: str,
    initiated_by: str | None = None,
    actor_source: str = "unknown",
    actor_confidence: str = "unverified",
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    error: dict[str, Any] | None = None,
    trace_id: str | None = None,
    latency_ms: int | None = None,
) -> dict[str, Any]:
    db_session = self.session_factory()
    try:
        existing = db_session.execute(
            select(ClinicalSessionRevisionRun).where(
                ClinicalSessionRevisionRun.pipeline_run_id == str(pipeline_run_id)
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = ClinicalSessionRevisionRun(
                pipeline_run_id=str(pipeline_run_id),
                session_id=int(session_id),
                root_session_id=int(root_session_id),
                source_version_id=int(source_version_id),
                target_revision_version_id=target_revision_version_id,
                revision_mode=revision_mode,
                revision_kind=revision_kind,
                configuration_json=self.serialize_json_payload(configuration),
                reviewer_note=self.normalize_string(reviewer_note),
                initiated_by=self.normalize_string(initiated_by),
                actor_id=None,
                actor_display_name=self.normalize_string(initiated_by),
                actor_source=actor_source,
                actor_confidence=actor_confidence,
                started_at=started_at or datetime.now(UTC),
                completed_at=completed_at,
                status=status,
                error_json=self.serialize_json_payload(error),
                token_usage_json=None,
                latency_ms=latency_ms,
                cost_estimate=None,
                trace_id=self.normalize_string(trace_id),
            )
            db_session.add(existing)
        else:
            existing.status = status
            existing.target_revision_version_id = target_revision_version_id
            existing.configuration_json = self.serialize_json_payload(configuration)
            existing.reviewer_note = self.normalize_string(reviewer_note)
            existing.initiated_by = self.normalize_string(initiated_by)
            existing.actor_display_name = self.normalize_string(initiated_by)
            if started_at is not None:
                existing.started_at = started_at
            existing.completed_at = completed_at
            existing.error_json = self.serialize_json_payload(error)
            existing.trace_id = self.normalize_string(trace_id)
            existing.latency_ms = latency_ms
        db_session.commit()
        return serialize_revision_run_row(self, existing)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def get_revision_run(self, pipeline_run_id: str) -> dict[str, Any] | None:
    db_session = self.session_factory()
    try:
        row = db_session.execute(
            select(ClinicalSessionRevisionRun).where(
                ClinicalSessionRevisionRun.pipeline_run_id == str(pipeline_run_id)
            )
        ).scalar_one_or_none()
        return None if row is None else serialize_revision_run_row(self, row)
    finally:
        db_session.close()


def list_revision_steps(self, pipeline_run_id: str) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        rows = db_session.execute(
            select(ClinicalSessionRevisionStep)
            .where(ClinicalSessionRevisionStep.pipeline_run_id == str(pipeline_run_id))
            .order_by(
                ClinicalSessionRevisionStep.step_index.asc(),
                ClinicalSessionRevisionStep.attempt_number.asc(),
                ClinicalSessionRevisionStep.id.asc(),
            )
        ).scalars()
        return [serialize_revision_step_row(self, row) for row in rows]
    finally:
        db_session.close()


def persist_revision_artifacts(
    self,
    *,
    pipeline_run_id: str,
    revision_version_id: int,
    result_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        safe_revision_version_id = int(revision_version_id)
        safe_pipeline_run_id = str(pipeline_run_id)
        db_session.query(ClinicalSessionRevisionArtifact).filter(
            ClinicalSessionRevisionArtifact.revision_version_id
            == safe_revision_version_id
        ).delete()

        created_rows: list[ClinicalSessionRevisionArtifact] = []

        structured_case = result_payload.get("structured_case")
        if isinstance(structured_case, dict):
            for entity_type in ("therapy_drugs", "anamnesis_drugs", "anamnesis_diseases"):
                entries = structured_case.get(entity_type)
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    entity_name = entry.get("name") or entry.get("drug_name")
                    row = _create_revision_artifact_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        pipeline_run_id=safe_pipeline_run_id,
                        artifact_kind="structured_case_entity",
                        artifact_key=f"{entity_type}:{index}",
                        entity_type=entity_type,
                        entity_name=str(entity_name).strip() if entity_name else None,
                        status="derived",
                        payload=entry,
                    )
                    db_session.add(row)
                    created_rows.append(row)

        faithfulness_audit = None
        pipeline_artifacts = result_payload.get("pipeline_artifacts")
        if isinstance(pipeline_artifacts, dict):
            raw_faithfulness_audit = pipeline_artifacts.get("faithfulness_audit")
            if isinstance(raw_faithfulness_audit, dict):
                faithfulness_audit = raw_faithfulness_audit
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="llm_qa_output",
                    artifact_key="faithfulness_audit",
                    status=(
                        "failed"
                        if result_payload.get("blocking_issues")
                        else "requires_human_review"
                        if bool(result_payload.get("manual_review_required"))
                        else "passed"
                    ),
                    payload={
                        **faithfulness_audit,
                        "manual_review_required": bool(
                            result_payload.get("manual_review_required")
                        ),
                        "blocking_issues": result_payload.get("blocking_issues") or [],
                    },
                )
                db_session.add(row)
                created_rows.append(row)
            fact_graph_validation = pipeline_artifacts.get("fact_graph_validation")
            if isinstance(fact_graph_validation, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="fact_graph_validation",
                    status="derived",
                    payload=fact_graph_validation,
                )
                db_session.add(row)
                created_rows.append(row)

        report_comparison = result_payload.get("report_comparison")
        if isinstance(report_comparison, dict):
            row = _create_revision_artifact_row(
                self,
                revision_version_id=safe_revision_version_id,
                pipeline_run_id=safe_pipeline_run_id,
                artifact_kind="report_comparison",
                artifact_key="report_comparison",
                status=self.normalize_string(str(report_comparison.get("outcome") or "")),
                payload=report_comparison,
            )
            db_session.add(row)
            created_rows.append(row)

        revision_payload = result_payload.get("revision")
        if isinstance(revision_payload, dict):
            instruction_profile = revision_payload.get("instruction_profile")
            if isinstance(instruction_profile, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="reviewer_instruction_profile",
                    status="derived",
                    payload=instruction_profile,
                )
                db_session.add(row)
                created_rows.append(row)
            instruction_trace = revision_payload.get("instruction_trace")
            if isinstance(instruction_trace, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="reviewer_instruction_trace",
                    status="derived",
                    payload=instruction_trace,
                )
                db_session.add(row)
                created_rows.append(row)
            final_report_rebuild = revision_payload.get("final_report_rebuild")
            if isinstance(final_report_rebuild, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="final_report_rebuild",
                    status=(
                        "warning"
                        if bool(final_report_rebuild.get("warnings"))
                        else "derived"
                    ),
                    payload=final_report_rebuild,
                )
                db_session.add(row)
                created_rows.append(row)
            qa_validation = revision_payload.get("qa_validation")
            if isinstance(qa_validation, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="llm_qa_output",
                    artifact_key="revision_qa_validation",
                    status=self.normalize_string(str(qa_validation.get("status") or "")),
                    payload=qa_validation,
                )
                db_session.add(row)
                created_rows.append(row)
            entity_pipeline = revision_payload.get("entity_pipeline")
            if isinstance(entity_pipeline, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="revision_entity_pipeline",
                    status="derived",
                    payload=entity_pipeline,
                )
                db_session.add(row)
                created_rows.append(row)
            entity_snapshot_context = revision_payload.get("entity_snapshot_context")
            if isinstance(entity_snapshot_context, str) and entity_snapshot_context.strip():
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="revision_entity_snapshot_context",
                    status="derived",
                    payload={"text": entity_snapshot_context.strip()},
                )
                db_session.add(row)
                created_rows.append(row)
            consultation_execution = revision_payload.get("consultation_execution")
            if isinstance(consultation_execution, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="revision_consultation_execution",
                    status="derived",
                    payload=consultation_execution,
                )
                db_session.add(row)
                created_rows.append(row)
            finalization_execution = revision_payload.get("finalization_execution")
            if isinstance(finalization_execution, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="revision_finalization_execution",
                    status="derived",
                    payload=finalization_execution,
                )
                db_session.add(row)
                created_rows.append(row)

        db_session.flush()
        db_session.commit()
        return [serialize_revision_artifact_row(self, row) for row in created_rows]
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def persist_revision_entities(
    self,
    *,
    pipeline_run_id: str,
    revision_version_id: int,
    source_version_id: int | None,
    result_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        safe_revision_version_id = int(revision_version_id)
        safe_pipeline_run_id = str(pipeline_run_id)
        now = datetime.now(UTC)
        db_session.execute(
            ClinicalSessionRevisionEntity.__table__.update()
            .where(
                ClinicalSessionRevisionEntity.revision_version_id
                == safe_revision_version_id,
                ClinicalSessionRevisionEntity.superseded_at.is_(None),
            )
            .values(superseded_at=now)
        )

        created_rows: list[ClinicalSessionRevisionEntity] = []

        structured_case = result_payload.get("structured_case")
        if isinstance(structured_case, dict):
            for section_name, source_section in (
                ("therapy_drugs", "therapy"),
                ("anamnesis_drugs", "anamnesis"),
            ):
                entries = structured_case.get(section_name)
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    validated_entry = validate_revised_drug_payload(entry)
                    serialized_entry = validated_entry.model_dump(exclude_none=True)
                    revised_name = validated_entry.name
                    row = _create_revision_entity_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        source_version_id=(
                            int(source_version_id)
                            if source_version_id is not None
                            else None
                        ),
                        pipeline_run_id=safe_pipeline_run_id,
                        step_name="generate_revision",
                        entity_type="drug",
                        source_section=source_section,
                        original_entity_id=f"{section_name}:{index}",
                        original_name=revised_name,
                        revised_name=revised_name,
                        normalized_name=normalize_text_key(revised_name),
                        payload=serialized_entry,
                        schema_name=REVISION_DRUG_SCHEMA_NAME,
                        requires_human_review=not bool(revised_name),
                    )
                    db_session.add(row)
                    created_rows.append(row)
            disease_entries = structured_case.get("anamnesis_diseases")
            if isinstance(disease_entries, list):
                for index, entry in enumerate(disease_entries):
                    if not isinstance(entry, dict):
                        continue
                    validated_entry = validate_revised_disease_payload(entry)
                    serialized_entry = validated_entry.model_dump(exclude_none=True)
                    revised_name = validated_entry.name
                    row = _create_revision_entity_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        source_version_id=(
                            int(source_version_id)
                            if source_version_id is not None
                            else None
                        ),
                        pipeline_run_id=safe_pipeline_run_id,
                        step_name="generate_revision",
                        entity_type="disease",
                        source_section="anamnesis",
                        original_entity_id=f"anamnesis_diseases:{index}",
                        original_name=revised_name,
                        revised_name=revised_name,
                        normalized_name=normalize_text_key(revised_name),
                        payload=serialized_entry,
                        schema_name=REVISION_DISEASE_SCHEMA_NAME,
                        requires_human_review=not bool(revised_name),
                    )
                    db_session.add(row)
                    created_rows.append(row)

        lab_entries = result_payload.get("lab_timeline")
        if isinstance(lab_entries, list):
            for index, entry in enumerate(lab_entries):
                if not isinstance(entry, dict):
                    continue
                validated_entry = validate_revised_lab_payload(entry)
                serialized_entry = validated_entry.model_dump(exclude_none=True)
                revised_name = validated_entry.marker_name
                row = _create_revision_entity_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    source_version_id=(
                        int(source_version_id) if source_version_id is not None else None
                    ),
                    pipeline_run_id=safe_pipeline_run_id,
                    step_name="generate_revision",
                    entity_type="lab_timeline_entry",
                    source_section="laboratory_analysis",
                    original_entity_id=f"lab_timeline:{index}",
                    original_name=revised_name,
                    revised_name=revised_name,
                    normalized_name=normalize_text_key(revised_name),
                    payload=serialized_entry,
                    schema_name=REVISION_LAB_SCHEMA_NAME,
                    requires_human_review=not bool(revised_name),
                )
                db_session.add(row)
                created_rows.append(row)

        revision_payload = result_payload.get("revision")
        if isinstance(revision_payload, dict):
            livertox_decisions = revision_payload.get("livertox_revision_decisions")
            if isinstance(livertox_decisions, list):
                for index, entry in enumerate(livertox_decisions):
                    if not isinstance(entry, dict):
                        continue
                    validated_entry = validate_revision_livertox_decision(entry)
                    serialized_entry = validated_entry.model_dump(exclude_none=True)
                    revised_name = validated_entry.drug_name
                    row = _create_revision_entity_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        source_version_id=(
                            int(source_version_id)
                            if source_version_id is not None
                            else None
                        ),
                        pipeline_run_id=safe_pipeline_run_id,
                        step_name="resolve_livertox_matches",
                        entity_type="livertox_match",
                        source_section="therapy",
                        original_entity_id=validated_entry.decision_id
                        or f"livertox:{index}",
                        original_name=revised_name,
                        revised_name=revised_name,
                        normalized_name=normalize_text_key(revised_name),
                        payload=serialized_entry,
                        schema_name=REVISION_LIVERTOX_DECISION_SCHEMA_NAME,
                        entity_revision_status=validated_entry.decision,
                        requires_human_review=validated_entry.requires_human_review,
                    )
                    db_session.add(row)
                    created_rows.append(row)
            revised_dili_assessments = revision_payload.get("revised_dili_assessments")
            if isinstance(revised_dili_assessments, list):
                for index, entry in enumerate(revised_dili_assessments):
                    if not isinstance(entry, dict):
                        continue
                    validated_entry = validate_revised_dili_assessment(entry)
                    serialized_entry = validated_entry.model_dump(exclude_none=True)
                    revised_name = validated_entry.drug_name
                    row = _create_revision_entity_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        source_version_id=(
                            int(source_version_id)
                            if source_version_id is not None
                            else None
                        ),
                        pipeline_run_id=safe_pipeline_run_id,
                        step_name="rerun_dili_assessments",
                        entity_type="dili_assessment",
                        source_section="therapy",
                        original_entity_id=validated_entry.revised_drug_entry_id
                        or f"dili:{index}",
                        original_name=revised_name,
                        revised_name=revised_name,
                        normalized_name=normalize_text_key(revised_name),
                        payload=serialized_entry,
                        schema_name=REVISION_DILI_ASSESSMENT_SCHEMA_NAME,
                        entity_revision_status="active",
                        requires_human_review=validated_entry.requires_human_review,
                    )
                    db_session.add(row)
                    created_rows.append(row)

        db_session.flush()
        db_session.commit()
        return [serialize_revision_entity_row(self, row) for row in created_rows]
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def list_revision_artifacts_for_version(
    self,
    *,
    revision_version_id: int,
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        rows = db_session.execute(
            select(ClinicalSessionRevisionArtifact)
            .where(
                ClinicalSessionRevisionArtifact.revision_version_id
                == int(revision_version_id)
            )
            .order_by(
                ClinicalSessionRevisionArtifact.artifact_kind.asc(),
                ClinicalSessionRevisionArtifact.entity_type.asc(),
                ClinicalSessionRevisionArtifact.entity_name.asc(),
                ClinicalSessionRevisionArtifact.id.asc(),
            )
        ).scalars()
        return [serialize_revision_artifact_row(self, row) for row in rows]
    finally:
        db_session.close()


def list_revision_entities_for_version(
    self,
    *,
    revision_version_id: int,
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        rows = db_session.execute(
            select(ClinicalSessionRevisionEntity)
            .where(
                ClinicalSessionRevisionEntity.revision_version_id
                == int(revision_version_id),
                ClinicalSessionRevisionEntity.superseded_at.is_(None),
            )
            .order_by(
                ClinicalSessionRevisionEntity.entity_type.asc(),
                ClinicalSessionRevisionEntity.source_section.asc(),
                ClinicalSessionRevisionEntity.normalized_name.asc(),
                ClinicalSessionRevisionEntity.id.asc(),
            )
        ).scalars()
        return [serialize_revision_entity_row(self, row) for row in rows]
    finally:
        db_session.close()


def _version_status_for_human_review_transition(
    *,
    current_version_status: str,
    clinical_review_status: str,
) -> str:
    if clinical_review_status == "approved_by_human":
        return "human_approved"
    if clinical_review_status == "rejected_by_human":
        return "human_rejected"
    if current_version_status in {"human_approved", "human_rejected"}:
        return "requires_human_review"
    return current_version_status


def record_revision_review_action(
    self,
    *,
    revision_version_id: int,
    clinical_review_status: str,
    reviewer_note: str | None,
    reviewed_by: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    db_session = self.session_factory()
    try:
        version_row = db_session.execute(
            select(ClinicalSessionVersion).where(
                ClinicalSessionVersion.id == int(revision_version_id)
            )
        ).scalar_one_or_none()
        if version_row is None:
            return None
        normalized_status = str(clinical_review_status or "").strip()
        if normalized_status not in {
            "under_review",
            "approved_by_human",
            "rejected_by_human",
        }:
            raise ValueError("Unsupported clinical review status")
        if version_row.revision_kind != "llm_assisted_revision":
            raise ValueError("Only LLM-assisted revision versions can be reviewed")
        reviewed_by_value = self.normalize_string(reviewed_by)
        review_row = ClinicalSessionRevisionReview(
            revision_version_id=int(version_row.id),
            session_id=int(version_row.session_id)
            if version_row.session_id is not None
            else None,
            clinical_review_status=normalized_status,
            reviewer_note=self.normalize_string(reviewer_note),
            reviewed_by=reviewed_by_value,
            actor_id=None,
            actor_display_name=reviewed_by_value,
            actor_source="manual_entry" if reviewed_by_value else "unknown",
            actor_confidence="unverified",
            metadata_json=self.serialize_json_payload(metadata or {}),
            reviewed_at=datetime.now(UTC),
        )
        db_session.add(review_row)
        version_row.clinical_review_status = normalized_status
        version_row.version_status = _version_status_for_human_review_transition(
            current_version_status=version_row.version_status,
            clinical_review_status=normalized_status,
        )
        db_session.flush()
        db_session.commit()
        return serialize_revision_review_row(self, review_row)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def list_revision_reviews_for_version(
    self,
    *,
    revision_version_id: int,
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        rows = db_session.execute(
            select(ClinicalSessionRevisionReview)
            .where(
                ClinicalSessionRevisionReview.revision_version_id
                == int(revision_version_id)
            )
            .order_by(
                ClinicalSessionRevisionReview.reviewed_at.desc(),
                ClinicalSessionRevisionReview.id.desc(),
            )
        ).scalars()
        return [serialize_revision_review_row(self, row) for row in rows]
    finally:
        db_session.close()


def start_revision_step(
    self,
    *,
    pipeline_run_id: str,
    step_name: str,
    step_index: int,
    step_count: int,
    input_summary: dict[str, Any] | None = None,
    input_payload: Any = None,
    schema_name: str | None = None,
    schema_version: str | None = None,
    prompt_version: str | None = None,
    parser_version: str | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    db_session = self.session_factory()
    try:
        safe_pipeline_run_id = str(pipeline_run_id)
        safe_step_name = str(step_name)
        previous_attempt = db_session.execute(
            select(func.max(ClinicalSessionRevisionStep.attempt_number)).where(
                ClinicalSessionRevisionStep.pipeline_run_id == safe_pipeline_run_id,
                ClinicalSessionRevisionStep.step_name == safe_step_name,
            )
        ).scalar_one()
        attempt_number = int(previous_attempt or 0) + 1
        now = started_at or datetime.now(UTC)
        db_session.execute(
            ClinicalSessionRevisionStep.__table__.update()
            .where(
                ClinicalSessionRevisionStep.pipeline_run_id == safe_pipeline_run_id,
                ClinicalSessionRevisionStep.step_name == safe_step_name,
                ClinicalSessionRevisionStep.superseded_at.is_(None),
            )
            .values(superseded_at=now)
        )
        row = ClinicalSessionRevisionStep(
            pipeline_run_id=safe_pipeline_run_id,
            step_name=safe_step_name,
            step_index=int(step_index),
            step_count=int(step_count),
            attempt_number=attempt_number,
            status="running",
            input_hash=(
                build_payload_hash(
                    input_payload if input_payload is not None else input_summary
                )
                if input_payload is not None or input_summary is not None
                else None
            ),
            input_summary_json=self.serialize_json_payload(input_summary),
            schema_name=self.normalize_string(schema_name),
            schema_version=self.normalize_string(schema_version),
            prompt_version=self.normalize_string(prompt_version),
            parser_version=self.normalize_string(parser_version),
            model_provider=self.normalize_string(model_provider),
            model_name=self.normalize_string(model_name),
            retry_count=attempt_number - 1,
            started_at=now,
        )
        db_session.add(row)
        db_session.flush()
        db_session.commit()
        return serialize_revision_step_row(self, row)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def complete_revision_step(
    self,
    *,
    pipeline_run_id: str,
    step_name: str,
    attempt_number: int,
    status: str = "completed",
    output_summary: dict[str, Any] | None = None,
    output_payload: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
    latency_ms: int | None = None,
    completed_at: datetime | None = None,
) -> dict[str, Any] | None:
    db_session = self.session_factory()
    try:
        row = db_session.execute(
            select(ClinicalSessionRevisionStep).where(
                ClinicalSessionRevisionStep.pipeline_run_id == str(pipeline_run_id),
                ClinicalSessionRevisionStep.step_name == str(step_name),
                ClinicalSessionRevisionStep.attempt_number == int(attempt_number),
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        row.status = str(status)
        row.output_hash = (
            build_payload_hash(
                output_payload if output_payload is not None else output_summary
            )
            if output_payload is not None or output_summary is not None
            else None
        )
        row.output_summary_json = self.serialize_json_payload(output_summary)
        row.output_payload_json = self.serialize_json_payload(output_payload)
        row.token_usage_json = self.serialize_json_payload(token_usage)
        row.latency_ms = latency_ms
        row.error_json = None
        row.completed_at = completed_at or datetime.now(UTC)
        db_session.commit()
        return serialize_revision_step_row(self, row)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def fail_revision_step(
    self,
    *,
    pipeline_run_id: str,
    step_name: str,
    attempt_number: int,
    error: dict[str, Any] | None,
    latency_ms: int | None = None,
    completed_at: datetime | None = None,
) -> dict[str, Any] | None:
    db_session = self.session_factory()
    try:
        row = db_session.execute(
            select(ClinicalSessionRevisionStep).where(
                ClinicalSessionRevisionStep.pipeline_run_id == str(pipeline_run_id),
                ClinicalSessionRevisionStep.step_name == str(step_name),
                ClinicalSessionRevisionStep.attempt_number == int(attempt_number),
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        row.status = "failed"
        row.error_json = self.serialize_json_payload(error)
        row.latency_ms = latency_ms
        row.completed_at = completed_at or datetime.now(UTC)
        db_session.commit()
        return serialize_revision_step_row(self, row)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def list_manual_report_edits(self, session_id: int) -> list[dict[str, Any]]:
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        rows = db_session.execute(
            select(ClinicalSessionManualEdit)
            .where(ClinicalSessionManualEdit.session_id == safe_session_id)
            .order_by(
                ClinicalSessionManualEdit.edited_at.desc(),
                ClinicalSessionManualEdit.id.desc(),
            )
        ).scalars()
        return [serialize_manual_edit_row(self, row) for row in rows]
    finally:
        db_session.close()


def serialize_manual_edit_row(
    self, row: ClinicalSessionManualEdit
) -> dict[str, Any]:
    try:
        edited_fields = json.loads(row.edited_fields_json)
    except (TypeError, json.JSONDecodeError):
        edited_fields = []
    metadata = self.parse_session_result_payload(row.metadata_json)
    return {
        "session_id": int(row.session_id),
        "current_version_id": int(row.current_version_id),
        "edited_by": self.normalize_string(row.edited_by),
        "actor_id": self.normalize_string(row.actor_id),
        "actor_display_name": self.normalize_string(row.actor_display_name),
        "actor_source": row.actor_source,
        "actor_confidence": row.actor_confidence,
        "edited_at": row.edited_at,
        "previous_text_hash": row.previous_text_hash,
        "new_text_hash": row.new_text_hash,
        "edited_fields": (
            [str(item) for item in edited_fields if isinstance(item, str)]
            if isinstance(edited_fields, list)
            else []
        ),
        "reviewer_note": self.normalize_string(row.reviewer_note),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


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
    safe_session_id = int(session_id)
    normalized_report = str(report_text).strip()
    if not normalized_report:
        raise ValueError("Report text cannot be empty.")

    db_session = self.session_factory()
    try:
        existing_session = db_session.get(ClinicalSession, safe_session_id)
        if existing_session is None:
            return None
        if metadata is not None:
            existing_session.metadata_json = self.serialize_json_payload(metadata or {})

        result_row = db_session.execute(
            select(ClinicalSessionResult).where(
                ClinicalSessionResult.session_id == safe_session_id
            )
        ).scalar_one_or_none()
        payload = (
            self.parse_session_result_payload(result_row.payload_json)
            if result_row is not None
            else {}
        ) or {}

        previous_report = self.normalize_string(payload.get("report")) or ""
        timestamp = datetime.now(UTC)
        payload["report"] = normalized_report
        payload["manual_edit_saved_at"] = timestamp.isoformat()

        serialized_payload = self.serialize_json_payload(payload)
        if serialized_payload is None:
            raise ValueError("Report payload could not be serialized.")

        if result_row is None:
            db_session.add(
                ClinicalSessionResult(
                    session_id=safe_session_id,
                    payload_json=serialized_payload,
                )
            )
        else:
            result_row.payload_json = serialized_payload

        normalized_metadata = (
            metadata if isinstance(metadata, dict) else {}
        )
        normalized_edited_by = self.normalize_string(edited_by)
        actor_display_name = normalized_edited_by
        actor_source = "manual_entry" if normalized_edited_by else "unknown"
        actor_confidence = "unverified"
        effective_fields = [
            field.strip()
            for field in (edited_fields or ["report_text"])
            if isinstance(field, str) and field.strip()
        ] or ["report_text"]

        db_session.add(
            ClinicalSessionManualEdit(
                session_id=safe_session_id,
                current_version_id=safe_session_id,
                edited_by=normalized_edited_by,
                actor_id=None,
                actor_display_name=actor_display_name,
                actor_source=actor_source,
                actor_confidence=actor_confidence,
                edited_at=timestamp,
                previous_text_hash=build_text_hash(previous_report),
                new_text_hash=build_text_hash(normalized_report),
                edited_fields_json=self.serialize_json_payload(effective_fields) or "[]",
                reviewer_note=self.normalize_string(reviewer_note),
                metadata_json=self.serialize_json_payload(normalized_metadata),
            )
        )
        db_session.commit()
        return {
            "session": self.get_session_detail(safe_session_id),
            "audit": self.list_manual_report_edits(safe_session_id)[0],
        }
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def update_session_metadata(
    self,
    session_id: int,
    *,
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        existing = db_session.get(ClinicalSession, safe_session_id)
        if existing is None:
            return None
        if metadata is not None:
            existing.metadata_json = self.serialize_json_payload(metadata or {})
        db_session.commit()
        return self.get_session_detail(safe_session_id)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()
