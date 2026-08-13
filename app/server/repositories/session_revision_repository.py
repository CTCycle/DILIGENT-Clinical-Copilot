from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from repositories import values as repository_values
from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext
from repositories.schemas.clinical import (
    ClinicalSession,
    ClinicalSessionResult,
    ClinicalSessionRevisionArtifact,
    ClinicalSessionRevisionReview,
    ClinicalSessionRevisionRun,
    ClinicalSessionRevisionStep,
    ClinicalSessionVersion,
)
from repositories.serialization.session_revision_serializers import (
    REVISION_DISEASE_SCHEMA_NAME,
    REVISION_DRUG_SCHEMA_NAME,
    build_payload_hash,
    build_text_hash,
    create_revision_artifact_row,
    create_revision_entity_row,
    normalize_text_key,
    serialize_revision_artifact_row,
    serialize_revision_entity_row,
    serialize_revision_review_row,
    serialize_revision_run_row,
    serialize_revision_step_row,
    serialize_version_row,
)
from repositories.serialization.session_result_data import (
    parse_session_result_payload,
    serialize_json_payload,
)

###############################################################################
class SessionRevisionRepository:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        context: RepositoryContext,
        clinical_session_repository: ClinicalSessionRepository,
    ) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory
        self.clinical_session_repository = clinical_session_repository

    # -------------------------------------------------------------------------
    def _root_session_id(self, db_session: Session, session_id: int) -> int | None:
        root = db_session.execute(
            select(ClinicalSessionVersion.root_session_id)
            .where(ClinicalSessionVersion.session_id == int(session_id))
            .order_by(ClinicalSessionVersion.version_number.desc())
            .limit(1)
        ).scalar_one_or_none()
        return int(root) if root is not None else int(session_id)

    # -------------------------------------------------------------------------
    def _manual_edit_row(self, row: ClinicalSessionVersion) -> dict[str, Any]:
        metadata = parse_session_result_payload(row.metadata_json) or {}
        audit = metadata.get("manual_edit_audit")
        audit = audit if isinstance(audit, dict) else {}
        return {
            "session_id": int(row.session_id or row.root_session_id),
            "current_version_id": int(row.id),
            "edited_by": repository_values.normalize_string(audit.get("edited_by")),
            "actor_id": repository_values.normalize_string(audit.get("actor_id")),
            "actor_display_name": repository_values.normalize_string(audit.get("actor_display_name")),
            "actor_source": audit.get("actor_source", "unknown"),
            "actor_confidence": audit.get("actor_confidence", "unverified"),
            "edited_at": row.completed_at or row.created_at,
            "previous_text_hash": audit.get("previous_text_hash", ""),
            "new_text_hash": audit.get("new_text_hash", ""),
            "edited_fields": audit.get("edited_fields", []),
            "reviewer_note": repository_values.normalize_string(audit.get("reviewer_note")),
            "metadata": audit.get("metadata", {}),
        }

    # -------------------------------------------------------------------------
    def list_session_versions(self, session_id: int) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            root_session_id = self._root_session_id(db_session, int(session_id))
            rows = db_session.execute(
                select(ClinicalSessionVersion)
                .where(ClinicalSessionVersion.root_session_id == root_session_id)
                .order_by(ClinicalSessionVersion.version_number.asc(), ClinicalSessionVersion.id.asc())
            ).scalars().all()
            return [serialize_version_row(row) for row in rows]

    # -------------------------------------------------------------------------
    def get_session_version_detail(
        self, session_id: int, *, version_id: int
    ) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            root_session_id = self._root_session_id(db_session, int(session_id))
            row = db_session.get(ClinicalSessionVersion, int(version_id))
            if row is None or int(row.root_session_id) != root_session_id:
                return None
            return {
                "version": serialize_version_row(row),
                "session": (
                    self.clinical_session_repository.get_session_detail(int(row.session_id))
                    if row.session_id
                    else None
                ),
            }

    # -------------------------------------------------------------------------
    def get_version_record_for_session(self, session_id: int) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(ClinicalSessionVersion)
                .where(ClinicalSessionVersion.session_id == int(session_id))
                .order_by(ClinicalSessionVersion.version_number.desc(), ClinicalSessionVersion.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            return serialize_version_row(row) if row is not None else None

    # -------------------------------------------------------------------------
    def get_latest_version_record_for_session(self, session_id: int) -> dict[str, Any] | None:
        versions = self.list_session_versions(session_id)
        return versions[-1] if versions else None

    # -------------------------------------------------------------------------
    def get_next_session_version(self, root_session_id: int) -> int:
        with self.session_factory() as db_session:
            maximum = db_session.execute(
                select(func.max(ClinicalSessionVersion.version_number)).where(
                    ClinicalSessionVersion.root_session_id == int(root_session_id)
                )
            ).scalar_one_or_none()
            return int(maximum or 1) + 1

    # -------------------------------------------------------------------------
    def list_manual_report_edits(self, session_id: int) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(ClinicalSessionVersion)
                .where(
                    ClinicalSessionVersion.session_id == int(session_id),
                    ClinicalSessionVersion.revision_kind == "manual_edit",
                )
                .order_by(ClinicalSessionVersion.version_number.desc(), ClinicalSessionVersion.id.desc())
            ).scalars().all()
            return [self._manual_edit_row(row) for row in rows]

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
        normalized_report = report_text.strip()
        if not normalized_report:
            raise ValueError("Report text cannot be empty.")
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
            current = db_session.execute(
                select(ClinicalSessionVersion)
                .where(ClinicalSessionVersion.session_id == safe_session_id)
                .order_by(ClinicalSessionVersion.version_number.desc())
                .limit(1)
            ).scalar_one_or_none()
            existing_session = db_session.get(ClinicalSession, safe_session_id)
            if current is None:
                raise RuntimeError("No persisted version exists for manual report edit")
            if existing_session is None:
                return None
            if metadata is not None:
                existing_session.metadata_json = serialize_json_payload(metadata or {})
            result_row = db_session.execute(
                select(ClinicalSessionResult).where(ClinicalSessionResult.session_id == safe_session_id)
            ).scalar_one_or_none()
            payload = parse_session_result_payload(result_row.payload_json) if result_row else {}
            payload = payload or {}
            previous_report = repository_values.normalize_string(payload.get("report")) or ""
            timestamp = datetime.now(UTC)
            payload["report"] = normalized_report
            payload["manual_edit_saved_at"] = timestamp.isoformat()
            serialized_payload = serialize_json_payload(payload)
            if result_row is None:
                db_session.add(ClinicalSessionResult(session_id=safe_session_id, payload_json=serialized_payload or "{}"))
            else:
                result_row.payload_json = serialized_payload or "{}"
            normalized_editor = repository_values.normalize_string(edited_by)
            fields = [
                field.strip()
                for field in (edited_fields or ["report_text"])
                if isinstance(field, str) and field.strip()
            ] or ["report_text"]
            max_version = db_session.execute(
                select(func.max(ClinicalSessionVersion.version_number)).where(
                    ClinicalSessionVersion.root_session_id == int(current.root_session_id)
                )
            ).scalar_one_or_none()
            current.version_status = "superseded"
            audit_payload = {
                "manual_edit_audit": {
                    "edited_by": normalized_editor,
                    "actor_id": None,
                    "actor_display_name": normalized_editor,
                    "actor_source": "manual_entry" if normalized_editor else "unknown",
                    "actor_confidence": "unverified",
                    "previous_text_hash": build_text_hash(previous_report),
                    "new_text_hash": build_text_hash(normalized_report),
                    "edited_fields": fields,
                    "reviewer_note": repository_values.normalize_string(reviewer_note),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                }
            }
            db_session.add(
                ClinicalSessionVersion(
                    session_id=safe_session_id,
                    root_session_id=int(current.root_session_id),
                    source_version_id=int(current.id),
                    version_number=int(max_version or current.version_number) + 1,
                    version_status="current",
                    revision_kind="manual_edit",
                    llm_qa_status="not_run",
                    clinical_review_status="not_reviewed",
                    pipeline_run_id=None,
                    report_text=normalized_report,
                    metadata_json=serialize_json_payload(audit_payload),
                    completed_at=timestamp,
                )
            )
            db_session.commit()
        audits = self.list_manual_report_edits(safe_session_id)
        return {
            "session": self.clinical_session_repository.get_session_detail(
                safe_session_id
            ),
            "audit": audits[0],
        }

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
        del reviewer_note, initiated_by
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
            source = db_session.execute(
                select(ClinicalSessionVersion)
                .where(ClinicalSessionVersion.session_id == safe_session_id)
                .order_by(ClinicalSessionVersion.version_number.desc())
                .limit(1)
            ).scalar_one_or_none()
            if source is None:
                return None
            run_id = pipeline_run_id or uuid4().hex
            existing = db_session.execute(
                select(ClinicalSessionVersion).where(ClinicalSessionVersion.pipeline_run_id == run_id)
            ).scalar_one_or_none()
            if existing is not None:
                return serialize_version_row(existing)
            maximum = db_session.execute(
                select(func.max(ClinicalSessionVersion.version_number)).where(
                    ClinicalSessionVersion.root_session_id == int(source.root_session_id)
                )
            ).scalar_one_or_none()
            shell = ClinicalSessionVersion(
                session_id=None,
                root_session_id=int(source.root_session_id),
                source_version_id=int(source.id),
                version_number=int(maximum or 1) + 1,
                version_status="draft_revision",
                revision_kind="llm_assisted_revision",
                llm_qa_status="pending",
                clinical_review_status="not_reviewed",
                pipeline_run_id=run_id,
                model_configuration_json=serialize_json_payload(configuration),
            )
            db_session.add(shell)
            db_session.commit()
            db_session.refresh(shell)
            return serialize_version_row(shell)

    # -------------------------------------------------------------------------
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
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(ClinicalSessionRevisionRun).where(
                    ClinicalSessionRevisionRun.pipeline_run_id == str(pipeline_run_id)
                )
            ).scalar_one_or_none()
            if row is None:
                row = ClinicalSessionRevisionRun(
                    pipeline_run_id=str(pipeline_run_id),
                    session_id=int(session_id),
                    root_session_id=int(root_session_id),
                    source_version_id=int(source_version_id),
                    target_revision_version_id=target_revision_version_id,
                    revision_mode=revision_mode,
                    revision_kind=revision_kind,
                    configuration_json=serialize_json_payload(configuration),
                    reviewer_note=repository_values.normalize_string(reviewer_note),
                    initiated_by=repository_values.normalize_string(initiated_by),
                    actor_id=None,
                    actor_display_name=repository_values.normalize_string(initiated_by),
                    actor_source=actor_source,
                    actor_confidence=actor_confidence,
                    started_at=started_at or datetime.now(UTC),
                    completed_at=completed_at,
                    status=status,
                    error_json=serialize_json_payload(error),
                    latency_ms=latency_ms,
                    trace_id=repository_values.normalize_string(trace_id),
                )
                db_session.add(row)
            else:
                existing_configuration = parse_session_result_payload(row.configuration_json) or {}
                if row.status in {"completed", "failed", "cancelled"} and status != row.status:
                    status = row.status
                    completed_at = row.completed_at
                    error = parse_session_result_payload(row.error_json)
                row.status = status
                row.target_revision_version_id = target_revision_version_id
                row.configuration_json = serialize_json_payload({**existing_configuration, **configuration})
                row.reviewer_note = repository_values.normalize_string(reviewer_note)
                row.initiated_by = repository_values.normalize_string(initiated_by)
                row.actor_display_name = repository_values.normalize_string(initiated_by)
                row.started_at = started_at or row.started_at
                row.completed_at = completed_at
                row.error_json = serialize_json_payload(error)
                row.trace_id = repository_values.normalize_string(trace_id)
                row.latency_ms = latency_ms
            db_session.commit()
            db_session.refresh(row)
            return serialize_revision_run_row(row)

    # -------------------------------------------------------------------------
    def get_revision_run(self, pipeline_run_id: str) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(ClinicalSessionRevisionRun).where(
                    ClinicalSessionRevisionRun.pipeline_run_id == str(pipeline_run_id)
                )
            ).scalar_one_or_none()
            return serialize_revision_run_row(row) if row is not None else None

    # -------------------------------------------------------------------------
    def get_revision_run_by_job_id(self, job_id: str) -> dict[str, Any] | None:
        safe_job_id = job_id.strip()
        if not safe_job_id:
            return None
        with self.session_factory() as db_session:
            for row in db_session.execute(
                select(ClinicalSessionRevisionRun).order_by(
                    ClinicalSessionRevisionRun.started_at.desc(), ClinicalSessionRevisionRun.id.desc()
                )
            ).scalars():
                configuration = parse_session_result_payload(row.configuration_json) or {}
                if str(configuration.get("job_id") or "").strip() == safe_job_id:
                    return serialize_revision_run_row(row)
        return None

    # -------------------------------------------------------------------------
    def list_revision_runs_by_status(self, status: str) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(ClinicalSessionRevisionRun)
                .where(ClinicalSessionRevisionRun.status == str(status))
                .order_by(ClinicalSessionRevisionRun.started_at.asc())
            ).scalars().all()
            return [serialize_revision_run_row(row) for row in rows]

    # -------------------------------------------------------------------------
    def fail_revision_run(self, *, pipeline_run_id: str, error: dict[str, Any] | None = None) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(ClinicalSessionRevisionRun).where(
                    ClinicalSessionRevisionRun.pipeline_run_id == str(pipeline_run_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            if row.status != "failed":
                row.status = "failed"
                row.completed_at = datetime.now(UTC)
                row.error_json = serialize_json_payload(error)
            db_session.commit()
            db_session.refresh(row)
            return serialize_revision_run_row(row)

    # -------------------------------------------------------------------------
    def cancel_revision_run(self, *, pipeline_run_id: str) -> None:
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(ClinicalSessionRevisionRun).where(
                    ClinicalSessionRevisionRun.pipeline_run_id == str(pipeline_run_id)
                )
            ).scalar_one_or_none()
            if row is not None and row.status not in {"completed", "failed"}:
                row.status = "cancelled"
                row.completed_at = datetime.now(UTC)
                row.error_json = serialize_json_payload({"message": "Revision was cancelled."})
            db_session.commit()

    # -------------------------------------------------------------------------
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
        with self.session_factory() as db_session:
            run_id = db_session.execute(
                select(ClinicalSessionRevisionRun.id).where(
                    ClinicalSessionRevisionRun.pipeline_run_id == str(pipeline_run_id)
                )
            ).scalar_one_or_none()
            if run_id is None:
                raise ValueError("Revision run must exist before recording a step")
            previous = db_session.execute(
                select(func.max(ClinicalSessionRevisionStep.attempt_number)).where(
                    ClinicalSessionRevisionStep.pipeline_run_id == str(pipeline_run_id),
                    ClinicalSessionRevisionStep.step_name == str(step_name),
                )
            ).scalar_one()
            now = started_at or datetime.now(UTC)
            db_session.execute(
                update(ClinicalSessionRevisionStep)
                .where(
                    ClinicalSessionRevisionStep.pipeline_run_id == str(pipeline_run_id),
                    ClinicalSessionRevisionStep.step_name == str(step_name),
                    ClinicalSessionRevisionStep.superseded_at.is_(None),
                )
                .values(superseded_at=now)
            )
            row = ClinicalSessionRevisionStep(
                revision_run_id=int(run_id),
                pipeline_run_id=str(pipeline_run_id),
                step_name=str(step_name),
                step_index=int(step_index),
                step_count=int(step_count),
                attempt_number=int(previous or 0) + 1,
                status="running",
                input_hash=build_payload_hash(input_payload if input_payload is not None else input_summary) if input_payload is not None or input_summary is not None else None,
                input_summary_json=serialize_json_payload(input_summary),
                schema_name=repository_values.normalize_string(schema_name),
                schema_version=repository_values.normalize_string(schema_version),
                prompt_version=repository_values.normalize_string(prompt_version),
                parser_version=repository_values.normalize_string(parser_version),
                model_provider=repository_values.normalize_string(model_provider),
                model_name=repository_values.normalize_string(model_name),
                started_at=now,
            )
            db_session.add(row)
            db_session.commit()
            db_session.refresh(row)
            return serialize_revision_step_row(row)

    # -------------------------------------------------------------------------
    def complete_revision_step(self, *, pipeline_run_id: str, step_name: str, attempt_number: int | None = None, status: str = "completed", output_summary: dict[str, Any] | None = None, output_payload: Any = None, token_usage: dict[str, Any] | None = None, latency_ms: int | None = None, completed_at: datetime | None = None, retry_count: int | None = None) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(ClinicalSessionRevisionStep)
                .where(
                    ClinicalSessionRevisionStep.pipeline_run_id == str(pipeline_run_id),
                    ClinicalSessionRevisionStep.step_name == str(step_name),
                    ClinicalSessionRevisionStep.superseded_at.is_(None),
                    *([ClinicalSessionRevisionStep.attempt_number == int(attempt_number)] if attempt_number is not None else []),
                )
                .order_by(ClinicalSessionRevisionStep.attempt_number.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            row.status = status
            row.output_hash = build_payload_hash(output_payload if output_payload is not None else output_summary) if output_payload is not None or output_summary is not None else None
            row.output_summary_json = serialize_json_payload(output_summary)
            row.output_payload_json = serialize_json_payload(output_payload)
            row.token_usage_json = serialize_json_payload(token_usage)
            row.latency_ms = latency_ms
            row.retry_count = int(retry_count or row.retry_count)
            row.completed_at = completed_at or datetime.now(UTC)
            db_session.commit()
            db_session.refresh(row)
            return serialize_revision_step_row(row)

    # -------------------------------------------------------------------------
    def fail_revision_step(self, *, pipeline_run_id: str, step_name: str, attempt_number: int | None = None, error: dict[str, Any] | None = None, latency_ms: int | None = None, completed_at: datetime | None = None) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(ClinicalSessionRevisionStep)
                .where(
                    ClinicalSessionRevisionStep.pipeline_run_id == str(pipeline_run_id),
                    ClinicalSessionRevisionStep.step_name == str(step_name),
                    ClinicalSessionRevisionStep.superseded_at.is_(None),
                    *([ClinicalSessionRevisionStep.attempt_number == int(attempt_number)] if attempt_number is not None else []),
                )
                .order_by(ClinicalSessionRevisionStep.attempt_number.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            row.status = "failed"
            row.error_json = serialize_json_payload(error)
            row.latency_ms = latency_ms
            row.completed_at = completed_at or datetime.now(UTC)
            db_session.commit()
            db_session.refresh(row)
            return serialize_revision_step_row(row)

    # -------------------------------------------------------------------------
    def list_revision_steps(self, pipeline_run_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(ClinicalSessionRevisionStep)
                .where(ClinicalSessionRevisionStep.pipeline_run_id == str(pipeline_run_id))
                .order_by(ClinicalSessionRevisionStep.step_index.asc(), ClinicalSessionRevisionStep.attempt_number.asc(), ClinicalSessionRevisionStep.id.asc())
            ).scalars().all()
            return [serialize_revision_step_row(row) for row in rows]

    # -------------------------------------------------------------------------
    def persist_revision_artifact(self, *, pipeline_run_id: str, revision_version_id: int, artifact_key: str, payload: dict[str, Any], status: str = "derived") -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            row = create_revision_artifact_row(
                revision_version_id=int(revision_version_id),
                pipeline_run_id=str(pipeline_run_id),
                artifact_kind="pipeline_artifact",
                artifact_key=artifact_key,
                status=status,
                payload=payload,
            )
            db_session.add(row)
            db_session.commit()
            db_session.refresh(row)
            return [serialize_revision_artifact_row(row)]

    # -------------------------------------------------------------------------
    def persist_revision_artifacts(self, *, pipeline_run_id: str, revision_version_id: int, result_payload: dict[str, Any]) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        structured_case = result_payload.get("structured_case")
        if isinstance(structured_case, dict):
            for entity_type in ("therapy_drugs", "anamnesis_drugs", "anamnesis_diseases"):
                entries = structured_case.get(entity_type)
                if isinstance(entries, list):
                    for index, entry in enumerate(entries):
                        if isinstance(entry, dict):
                            created.extend(self.persist_revision_artifact(pipeline_run_id=pipeline_run_id, revision_version_id=revision_version_id, artifact_key=f"{entity_type}:{index}", payload=entry))
        pipeline_artifacts = result_payload.get("pipeline_artifacts")
        if isinstance(pipeline_artifacts, dict):
            for key, value in pipeline_artifacts.items():
                if isinstance(value, dict):
                    created.extend(self.persist_revision_artifact(pipeline_run_id=pipeline_run_id, revision_version_id=revision_version_id, artifact_key=str(key), payload=value))
        return created

    # -------------------------------------------------------------------------
    def list_revision_artifacts_for_version(self, *, revision_version_id: int, include_superseded: bool = False) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            statement = select(ClinicalSessionRevisionArtifact).where(ClinicalSessionRevisionArtifact.revision_version_id == int(revision_version_id))
            rows = db_session.execute(statement.order_by(ClinicalSessionRevisionArtifact.created_at.asc(), ClinicalSessionRevisionArtifact.id.asc())).scalars().all()
            return [serialize_revision_artifact_row(row) for row in rows if include_superseded or row.status != "superseded"]

    # -------------------------------------------------------------------------
    def persist_revision_entities(self, *, pipeline_run_id: str, revision_version_id: int, source_version_id: int | None, result_payload: dict[str, Any]) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        structured_case = result_payload.get("structured_case")
        if isinstance(structured_case, dict):
            for section_name, entity_type, schema_name in (("therapy_drugs", "drug", REVISION_DRUG_SCHEMA_NAME), ("anamnesis_drugs", "drug", REVISION_DRUG_SCHEMA_NAME), ("anamnesis_diseases", "disease", REVISION_DISEASE_SCHEMA_NAME)):
                entries = structured_case.get(section_name)
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    name = repository_values.normalize_string(entry.get("name") or entry.get("drug_name"))
                    row = create_revision_entity_row(revision_version_id=int(revision_version_id), source_version_id=source_version_id, pipeline_run_id=str(pipeline_run_id), step_name="generate_revision", entity_type=entity_type, source_section="therapy" if section_name == "therapy_drugs" else "anamnesis", original_entity_id=f"{section_name}:{index}", original_name=name, revised_name=name, normalized_name=normalize_text_key(name), requires_human_review=not bool(name), payload=entry, schema_name=schema_name)
                    with self.session_factory() as db_session:
                        db_session.add(row)
                        db_session.commit()
                        db_session.refresh(row)
                    created.append(serialize_revision_entity_row(row))
        return created

    # -------------------------------------------------------------------------
    def list_revision_entities_for_version(self, *, revision_version_id: int, include_superseded: bool = False) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(ClinicalSessionRevisionArtifact)
                .where(
                    ClinicalSessionRevisionArtifact.revision_version_id == int(revision_version_id),
                    ClinicalSessionRevisionArtifact.artifact_kind == "structured_case_entity",
                )
                .order_by(ClinicalSessionRevisionArtifact.created_at.asc(), ClinicalSessionRevisionArtifact.id.asc())
            ).scalars().all()
            return [serialize_revision_entity_row(row) for row in rows if include_superseded or row.status != "superseded"]

    # -------------------------------------------------------------------------
    def record_revision_review_action(self, *, revision_version_id: int, clinical_review_status: str, reviewer_note: str | None, reviewed_by: str | None, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if clinical_review_status not in {"under_review", "approved_by_human", "rejected_by_human"}:
            raise ValueError("Unsupported clinical review status")
        with self.session_factory() as db_session:
            version = db_session.get(ClinicalSessionVersion, int(revision_version_id))
            if version is None:
                return None
            if version.revision_kind != "llm_assisted_revision":
                raise ValueError("Only LLM-assisted revision versions can be reviewed")
            if version.session_id is None or version.completed_at is None:
                raise ValueError("Only completed revision versions can be reviewed")
            reviewer = repository_values.normalize_string(reviewed_by)
            row = ClinicalSessionRevisionReview(revision_version_id=int(version.id), session_id=int(version.session_id) if version.session_id else None, clinical_review_status=clinical_review_status, reviewer_note=repository_values.normalize_string(reviewer_note), reviewed_by=reviewer, actor_id=None, actor_display_name=reviewer, actor_source="manual_entry" if reviewer else "unknown", actor_confidence="unverified", metadata_json=serialize_json_payload(metadata or {}), reviewed_at=datetime.now(UTC))
            db_session.add(row)
            version.clinical_review_status = clinical_review_status
            version.version_status = {"approved_by_human": "human_approved", "rejected_by_human": "human_rejected"}.get(clinical_review_status, "requires_human_review")
            db_session.commit()
            db_session.refresh(row)
            return serialize_revision_review_row(row)

    # -------------------------------------------------------------------------
    def list_revision_reviews_for_version(self, *, revision_version_id: int) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(ClinicalSessionRevisionReview)
                .where(ClinicalSessionRevisionReview.revision_version_id == int(revision_version_id))
                .order_by(ClinicalSessionRevisionReview.reviewed_at.desc(), ClinicalSessionRevisionReview.id.desc())
            ).scalars().all()
            return [serialize_revision_review_row(row) for row in rows]

    # -------------------------------------------------------------------------
    def finalize_revision_version(self, *, pipeline_run_id: str, persisted_session_id: int, model_configuration: dict[str, Any] | None = None, version_status: str = "requires_human_review", llm_qa_status: str = "not_run", clinical_review_status: str = "not_reviewed") -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.execute(select(ClinicalSessionVersion).where(ClinicalSessionVersion.pipeline_run_id == str(pipeline_run_id))).scalar_one_or_none()
            if row is None:
                return None
            row.session_id = int(persisted_session_id)
            row.version_status = version_status
            row.llm_qa_status = llm_qa_status
            row.clinical_review_status = clinical_review_status
            row.completed_at = datetime.now(UTC)
            if model_configuration is not None:
                row.model_configuration_json = serialize_json_payload(model_configuration)
            db_session.commit()
            db_session.refresh(row)
            return serialize_version_row(row)

    # -------------------------------------------------------------------------
    def update_session_metadata(self, session_id: int, *, metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            session = db_session.get(ClinicalSession, int(session_id))
            if session is None:
                return None
            if metadata is not None:
                session.metadata_json = serialize_json_payload(metadata or {})
            db_session.commit()
        return self.clinical_session_repository.get_session_detail(int(session_id))
