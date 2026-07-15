from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update

from repositories.schemas.clinical import (
    ClinicalSessionRevisionReview,
    ClinicalSessionRevisionRun,
    ClinicalSessionRevisionStep,
    ClinicalSessionVersion,
)
from repositories.serialization.session_revision_data import (
    build_payload_hash,
    serialize_revision_review_row,
    serialize_revision_step_row,
)

###############################################################################
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

###############################################################################
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

###############################################################################
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

###############################################################################
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
        revision_run_id = db_session.execute(
            select(ClinicalSessionRevisionRun.id).where(
                ClinicalSessionRevisionRun.pipeline_run_id == safe_pipeline_run_id
            )
        ).scalar_one_or_none()
        if revision_run_id is None:
            raise ValueError("Revision run must exist before recording a step")
        previous_attempt = db_session.execute(
            select(func.max(ClinicalSessionRevisionStep.attempt_number)).where(
                ClinicalSessionRevisionStep.pipeline_run_id == safe_pipeline_run_id,
                ClinicalSessionRevisionStep.step_name == safe_step_name,
            )
        ).scalar_one()
        attempt_number = int(previous_attempt or 0) + 1
        now = started_at or datetime.now(UTC)
        db_session.execute(
            update(ClinicalSessionRevisionStep)
            .where(
                ClinicalSessionRevisionStep.pipeline_run_id == safe_pipeline_run_id,
                ClinicalSessionRevisionStep.step_name == safe_step_name,
                ClinicalSessionRevisionStep.superseded_at.is_(None),
            )
            .values(superseded_at=now)
        )
        row = ClinicalSessionRevisionStep(
            revision_run_id=int(revision_run_id),
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

###############################################################################
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

###############################################################################
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
