"""Pure revision payload validation, hashing, row construction, and serialization."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from domain.clinical.revision import (
    RevisedDiliAssessment,
    RevisedDiseasePayload,
    RevisedDrugPayload,
    RevisedLabPayload,
    RevisionLiverToxDecision,
)
from repositories import values as repository_values
from repositories.schemas.clinical import (
    ClinicalSessionRevisionArtifact,
    ClinicalSessionRevisionReview,
    ClinicalSessionRevisionRun,
    ClinicalSessionRevisionStep,
    ClinicalSessionVersion,
)
from repositories.serialization.session_result_data import (
    parse_session_result_payload,
    serialize_json_payload,
)

REVISION_DRUG_SCHEMA_NAME = "revised_drug_entry"
REVISION_DISEASE_SCHEMA_NAME = "revised_disease_entry"
REVISION_LAB_SCHEMA_NAME = "revised_lab_entry"
REVISION_LIVERTOX_DECISION_SCHEMA_NAME = "revision_livertox_decision"
REVISION_DILI_ASSESSMENT_SCHEMA_NAME = "revised_dili_assessment"
REVISION_ENTITY_SCHEMA_VERSION = "1"


def build_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_payload_hash(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_text_key(value: str | None) -> str | None:
    normalized = repository_values.normalize_string(value)
    if normalized is None:
        return None
    return normalized.casefold()


def default_version_status(*, is_latest: bool) -> str:
    return "current" if is_latest else "superseded"


def sync_preserved_version_status(
    existing_status: str | None, *, is_latest_completed: bool
) -> str:
    normalized = repository_values.normalize_string(existing_status)
    if normalized in {None, "current", "superseded"}:
        return default_version_status(is_latest=is_latest_completed)
    if normalized is None:
        return default_version_status(is_latest=is_latest_completed)
    return normalized


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


def serialize_version_row(row: ClinicalSessionVersion) -> dict[str, Any]:
    return {
        "version_id": int(row.id),
        "session_id": int(row.session_id) if row.session_id is not None else None,
        "root_session_id": int(row.root_session_id),
        "source_version_id": int(row.source_version_id) if row.source_version_id is not None else None,
        "revision_version_id": int(row.id),
        "version_number": int(row.version_number),
        "version_status": row.version_status,
        "revision_kind": row.revision_kind,
        "llm_qa_status": row.llm_qa_status,
        "clinical_review_status": row.clinical_review_status,
        "pipeline_run_id": repository_values.normalize_string(row.pipeline_run_id),
        "model_configuration": parse_session_result_payload(row.model_configuration_json) or {},
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "completed_at": row.completed_at,
    }


def serialize_revision_run_row(row: ClinicalSessionRevisionRun) -> dict[str, Any]:
    return {
        "pipeline_run_id": row.pipeline_run_id,
        "session_id": int(row.session_id),
        "root_session_id": int(row.root_session_id),
        "source_version_id": int(row.source_version_id),
        "target_revision_version_id": int(row.target_revision_version_id) if row.target_revision_version_id is not None else None,
        "revision_mode": row.revision_mode,
        "revision_kind": row.revision_kind,
        "configuration": parse_session_result_payload(row.configuration_json) or {},
        "reviewer_note": repository_values.normalize_string(row.reviewer_note),
        "initiated_by": repository_values.normalize_string(row.initiated_by),
        "actor_id": repository_values.normalize_string(row.actor_id),
        "actor_display_name": repository_values.normalize_string(row.actor_display_name),
        "actor_source": row.actor_source,
        "actor_confidence": row.actor_confidence,
        "started_at": row.started_at,
        "completed_at": row.completed_at,
        "status": row.status,
        "error": parse_session_result_payload(row.error_json),
        "token_usage": parse_session_result_payload(row.token_usage_json),
        "latency_ms": int(row.latency_ms) if row.latency_ms is not None else None,
        "cost_estimate": float(row.cost_estimate) if row.cost_estimate is not None else None,
        "trace_id": repository_values.normalize_string(row.trace_id),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def serialize_revision_step_row(row: ClinicalSessionRevisionStep) -> dict[str, Any]:
    return {
        "pipeline_run_id": row.pipeline_run_id,
        "step_name": row.step_name,
        "step_index": int(row.step_index),
        "step_count": int(row.step_count),
        "attempt_number": int(row.attempt_number),
        "status": row.status,
        "input_hash": repository_values.normalize_string(row.input_hash),
        "output_hash": repository_values.normalize_string(row.output_hash),
        "input_summary": parse_session_result_payload(row.input_summary_json),
        "output_summary": parse_session_result_payload(row.output_summary_json),
        "output_payload": parse_session_result_payload(row.output_payload_json),
        "schema_name": repository_values.normalize_string(row.schema_name),
        "schema_version": repository_values.normalize_string(row.schema_version),
        "prompt_version": repository_values.normalize_string(row.prompt_version),
        "parser_version": repository_values.normalize_string(row.parser_version),
        "model_provider": repository_values.normalize_string(row.model_provider),
        "model_name": repository_values.normalize_string(row.model_name),
        "token_usage": parse_session_result_payload(row.token_usage_json),
        "latency_ms": int(row.latency_ms) if row.latency_ms is not None else None,
        "retry_count": int(row.retry_count),
        "error": parse_session_result_payload(row.error_json),
        "started_at": row.started_at,
        "completed_at": row.completed_at,
        "superseded_at": row.superseded_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def serialize_revision_artifact_row(row: ClinicalSessionRevisionArtifact) -> dict[str, Any]:
    return {
        "revision_version_id": int(row.revision_version_id),
        "pipeline_run_id": row.pipeline_run_id,
        "artifact_kind": row.artifact_kind,
        "artifact_key": repository_values.normalize_string(row.artifact_key),
        "entity_type": repository_values.normalize_string(row.entity_type),
        "entity_name": repository_values.normalize_string(row.entity_name),
        "status": repository_values.normalize_string(row.status),
        "schema_version": repository_values.normalize_string(row.schema_version),
        "payload": parse_session_result_payload(row.payload_json),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def serialize_revision_entity_row(row: ClinicalSessionRevisionArtifact) -> dict[str, Any]:
    artifact_payload = parse_session_result_payload(row.payload_json) or {}
    entity = artifact_payload.get("entity")
    payload = artifact_payload.get("payload")
    entity = entity if isinstance(entity, dict) else {}
    return {
        "revision_version_id": int(row.revision_version_id),
        "source_version_id": entity.get("source_version_id"),
        "pipeline_run_id": row.pipeline_run_id,
        "step_name": entity.get("step_name"),
        "entity_type": row.entity_type,
        "entity_revision_status": entity.get("entity_revision_status"),
        "source_section": entity.get("source_section"),
        "original_entity_id": entity.get("original_entity_id"),
        "original_name": entity.get("original_name"),
        "revised_name": entity.get("revised_name"),
        "normalized_name": entity.get("normalized_name"),
        "requires_human_review": bool(entity.get("requires_human_review")),
        "human_review_status": entity.get("human_review_status"),
        "payload": payload if isinstance(payload, dict) else {},
        "schema_name": entity.get("schema_name"),
        "schema_version": entity.get("schema_version"),
        "prompt_version": None,
        "parser_version": None,
        "model_provider": None,
        "model_name": None,
        "input_hash": None,
        "output_hash": entity.get("output_hash"),
        "created_at": row.created_at,
        "superseded_at": None,
    }


def serialize_revision_review_row(row: ClinicalSessionRevisionReview) -> dict[str, Any]:
    return {
        "revision_version_id": int(row.revision_version_id),
        "session_id": int(row.session_id) if row.session_id is not None else None,
        "clinical_review_status": row.clinical_review_status,
        "reviewer_note": repository_values.normalize_string(row.reviewer_note),
        "reviewed_by": repository_values.normalize_string(row.reviewed_by),
        "actor_id": repository_values.normalize_string(row.actor_id),
        "actor_display_name": repository_values.normalize_string(row.actor_display_name),
        "actor_source": row.actor_source,
        "actor_confidence": row.actor_confidence,
        "metadata": parse_session_result_payload(row.metadata_json) or {},
        "reviewed_at": row.reviewed_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def create_revision_artifact_row(
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
        artifact_key=repository_values.normalize_string(artifact_key)
        or f"{artifact_kind}:{pipeline_run_id}",
        entity_type=repository_values.normalize_string(entity_type),
        entity_name=repository_values.normalize_string(entity_name),
        status=repository_values.normalize_string(status),
        schema_version=repository_values.normalize_string(schema_version),
        payload_json=serialize_json_payload(payload),
    )


def create_revision_entity_row(
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
) -> ClinicalSessionRevisionArtifact:
    metadata = {
        "source_version_id": source_version_id,
        "step_name": step_name,
        "entity_revision_status": entity_revision_status,
        "source_section": repository_values.normalize_string(source_section),
        "original_entity_id": repository_values.normalize_string(original_entity_id),
        "original_name": repository_values.normalize_string(original_name),
        "revised_name": repository_values.normalize_string(revised_name),
        "normalized_name": repository_values.normalize_string(normalized_name),
        "requires_human_review": bool(requires_human_review),
        "human_review_status": "required" if requires_human_review else "not_required",
        "schema_name": schema_name,
        "schema_version": REVISION_ENTITY_SCHEMA_VERSION,
        "output_hash": build_payload_hash(payload),
    }
    return create_revision_artifact_row(
        revision_version_id=revision_version_id,
        pipeline_run_id=pipeline_run_id,
        artifact_kind="structured_case_entity",
        artifact_key=repository_values.normalize_string(original_entity_id)
        or f"{entity_type}:{normalized_name}",
        status=entity_revision_status,
        payload={"entity": metadata, "payload": payload},
        entity_type=entity_type,
        entity_name=revised_name,
        schema_version=REVISION_ENTITY_SCHEMA_VERSION,
    )
