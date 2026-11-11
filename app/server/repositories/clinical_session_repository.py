# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any, cast

from sqlalchemy import String, and_, exists, func, or_, select, update
from sqlalchemy import cast as sql_cast
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from common.utils.logger import logger
from common.utils.text_utils import normalize_drug_name
from repositories.database.session import unit_of_work
from repositories import values as repository_values
from repositories.context import RepositoryContext
from repositories.schemas.clinical import (
    ClinicalDrugMention,
    ClinicalLabObservation,
    ClinicalSession,
    ClinicalSessionAttachment,
    ClinicalSessionResult,
    ClinicalSessionSection,
    ClinicalSessionTimeline,
    ClinicalSessionVersion,
)
from repositories.serialization import session_result_data


###############################################################################
def _build_search_pattern(value: str | None) -> str | None:
    normalized = repository_values.normalize_string(value)
    if normalized is None:
        return None
    escaped = re.sub(r"([%_\\])", r"\\\1", normalized.casefold())
    return f"%{escaped}%"


###############################################################################
class SessionStateConflictError(RuntimeError):
    """A lifecycle guard rejected a write because the session state changed."""


###############################################################################
def _report_exists(session_id: Any) -> Any:
    """A session has a report when any of its versions has non-empty report text."""
    return exists(
        select(1).where(
            ClinicalSessionVersion.session_id == session_id,
            ClinicalSessionVersion.report_text.isnot(None),
            ClinicalSessionVersion.report_text != "",
        )
    )


###############################################################################
def _rowcount(result: Any) -> int:
    return cast(CursorResult[Any], result).rowcount


###############################################################################
def _utc_now() -> datetime:
    return datetime.now(UTC)


###############################################################################
def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


###############################################################################
def _same_instant(stored: datetime | None, expected: datetime) -> bool:
    if stored is None:
        return False
    return _as_utc(stored) == _as_utc(expected)


###############################################################################
def session_input_text(row: ClinicalSession) -> str:
    return row.clinical_input or ""


###############################################################################
def session_input_hash(row: ClinicalSession) -> str:
    return hashlib.sha256(session_input_text(row).encode("utf-8")).hexdigest()


###############################################################################
class ClinicalSessionRepository:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        context: RepositoryContext,
    ) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    @staticmethod
    def _apply_result_columns(
        row: ClinicalSession, session_data: dict[str, Any]
    ) -> None:
        row.patient_name = repository_values.normalize_string(
            session_data.get("patient_name")
        )
        row.visit_date = repository_values.normalize_date_value(
            session_data.get("patient_visit_date")
        )
        row.anamnesis = repository_values.normalize_string(session_data.get("anamnesis"))
        row.drugs_text = repository_values.normalize_string(session_data.get("drugs"))
        row.laboratory_analysis = repository_values.normalize_string(
            session_data.get("laboratory_analysis")
        )
        if "patient_image_base64" in session_data:
            row.patient_image_blob = session_result_data.decode_patient_image(
                session_data.get("patient_image_base64")
            )
        row.session_timestamp = session_result_data.parse_datetime(
            session_data.get("session_timestamp")
        )
        row.hepatic_pattern = repository_values.normalize_string(
            session_data.get("hepatic_pattern")
        )
        row.text_extraction_model = repository_values.normalize_string(
            session_data.get("text_extraction_model")
        )
        row.clinical_model = repository_values.normalize_string(
            session_data.get("clinical_model")
        )
        row.total_duration = repository_values.to_float(
            session_data.get("total_duration")
        )
        row.session_kind = repository_values.normalize_string(
            session_data.get("session_kind")
        )
        if "metadata" in session_data:
            row.metadata_json = session_result_data.serialize_json_payload(
                session_data.get("metadata")
            )

    # -------------------------------------------------------------------------
    def _persist_result_children(
        self,
        db_session: Session,
        row: ClinicalSession,
        session_data: dict[str, Any],
    ) -> None:
        session_id = int(row.id)
        if session_data.get("root_session_id") is None:
            db_session.add(
                ClinicalSessionVersion(
                    session_id=session_id,
                    root_session_id=session_id,
                    source_version_id=None,
                    version_number=1,
                    version_status="current",
                    revision_kind="original",
                    llm_qa_status="not_run",
                    clinical_review_status="not_reviewed",
                    pipeline_run_id=None,
                    model_configuration_json=session_result_data.serialize_json_payload(
                        {
                            "text_extraction_model": row.text_extraction_model,
                            "clinical_model": row.clinical_model,
                        }
                    ),
                    completed_at=row.session_timestamp,
                )
            )
            db_session.flush()
        self.persist_session_sections(db_session, session_id, session_data)
        self.persist_session_labs(db_session, session_id, session_data)
        self.persist_session_drugs(db_session, session_id, session_data)
        self.persist_session_result_payload(db_session, session_id, session_data)

    # -------------------------------------------------------------------------
    def save_clinical_session(self, session_data: dict[str, Any]) -> int | None:
        """Insert an already finished session (agentic revisions)."""
        if not session_data:
            logger.warning("Skipping clinical session save; payload is empty")
            return None
        db_session = self.session_factory()
        try:
            # A finished session is created at the moment it was produced.
            finished_at = (
                session_result_data.parse_datetime(session_data.get("session_timestamp"))
                or _utc_now()
            )
            persisted_session = ClinicalSession(
                session_status=repository_values.normalize_session_status(
                    session_data.get("session_status") or "completed"
                ),
                clinical_input=repository_values.normalize_string(
                    session_data.get("clinical_input")
                ),
                created_at=finished_at,
                updated_at=finished_at,
                finished_at=finished_at,
                state_version=1,
            )
            self._apply_result_columns(persisted_session, session_data)
            db_session.add(persisted_session)
            db_session.flush()
            session_id = int(persisted_session.id)
            self._persist_result_children(db_session, persisted_session, session_data)
            db_session.commit()
            return session_id
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def complete_clinical_session(
        self, session_id: int, session_data: dict[str, Any]
    ) -> int:
        """Persist the outcome of a running analysis onto its own session row."""
        status = repository_values.normalize_session_status(
            session_data.get("session_status")
        )
        if status not in {"completed", "failed"}:
            raise ValueError("A finished analysis is either completed or failed.")
        db_session = self.session_factory()
        try:
            # Take the write lock with the guard itself before reading the row.
            guarded = db_session.execute(
                update(ClinicalSession)
                .where(
                    ClinicalSession.id == int(session_id),
                    ClinicalSession.session_status == "running",
                )
                .values(state_version=ClinicalSession.state_version + 1)
            )
            if _rowcount(guarded) != 1:
                raise SessionStateConflictError(
                    "Only a running clinical session can be completed."
                )
            row = cast(ClinicalSession, db_session.get(ClinicalSession, int(session_id)))
            self._apply_result_columns(row, session_data)
            now = _utc_now()
            row.session_status = status
            row.finished_at = now
            row.updated_at = now
            error = session_data.get("lifecycle_error")
            row.error_json = (
                session_result_data.serialize_json_payload(error) if error else None
            )
            if status == "completed":
                row.progress_stage = "completed"
                row.progress_percent = 100.0
                row.progress_message = None
            self._persist_result_children(db_session, row, session_data)
            db_session.commit()
            return int(row.id)
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def create_draft(
        self,
        *,
        patient_name: str | None,
        visit_date: date | None,
        clinical_input: str,
        patient_image_base64: str | None,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self.session_factory() as db_session:
            row = ClinicalSession(
                patient_name=repository_values.normalize_string(patient_name),
                visit_date=visit_date,
                clinical_input=clinical_input,
                patient_image_blob=session_result_data.decode_patient_image(
                    patient_image_base64
                ),
                session_status="draft",
                session_kind="standard_assessment",
                state_version=1,
                created_at=now,
                updated_at=now,
            )
            db_session.add(row)
            db_session.commit()
            session_id = int(row.id)
        return cast(dict[str, Any], self.get_session_state(session_id))

    # -------------------------------------------------------------------------
    def update_draft_input(
        self,
        session_id: int,
        *,
        patient_name: str | None,
        visit_date: date | None,
        clinical_input: str,
        patient_image_base64: str | None,
        expected_updated_at: datetime,
    ) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.get(ClinicalSession, int(session_id))
            if row is None:
                return None
            status, updated_at, state_version = (
                row.session_status,
                row.updated_at,
                row.state_version,
            )
        if status != "draft":
            raise SessionStateConflictError("Only a draft session can be edited.")
        stale = SessionStateConflictError(
            "The draft changed since it was loaded. Reload it before saving."
        )
        if not _same_instant(updated_at, expected_updated_at):
            raise stale
        # A fresh transaction whose first statement is the guarded write.
        with self.session_factory() as db_session:
            result = db_session.execute(
                update(ClinicalSession)
                .where(
                    ClinicalSession.id == int(session_id),
                    ClinicalSession.session_status == "draft",
                    ClinicalSession.state_version == state_version,
                )
                .values(
                    patient_name=repository_values.normalize_string(patient_name),
                    visit_date=visit_date,
                    clinical_input=clinical_input,
                    patient_image_blob=session_result_data.decode_patient_image(
                        patient_image_base64
                    ),
                    updated_at=_utc_now(),
                    state_version=ClinicalSession.state_version + 1,
                )
            )
            if _rowcount(result) != 1:
                db_session.rollback()
                raise stale
            db_session.commit()
        return self.get_session_state(session_id)

    # -------------------------------------------------------------------------
    def launch_draft(
        self,
        session_id: int,
        *,
        patient_name: str | None,
        visit_date: date | None,
        clinical_input: str,
        patient_image_base64: str | None,
        metadata: dict[str, Any],
        run_request: dict[str, Any],
        agent_revision_id: int | None = None,
    ) -> bool:
        """Atomically move a draft into the queue; False when it is not a draft."""
        now = _utc_now()
        with self.session_factory() as db_session:
            result = db_session.execute(
                update(ClinicalSession)
                .where(
                    ClinicalSession.id == int(session_id),
                    ClinicalSession.session_status == "draft",
                )
                .values(
                    session_status="queued",
                    patient_name=repository_values.normalize_string(patient_name),
                    visit_date=visit_date,
                    clinical_input=clinical_input,
                    patient_image_blob=session_result_data.decode_patient_image(
                        patient_image_base64
                    ),
                    metadata_json=session_result_data.serialize_json_payload(metadata),
                    agent_revision_id=agent_revision_id,
                    run_request_json=session_result_data.serialize_json_payload(
                        run_request
                    ),
                    progress_stage=None,
                    progress_percent=None,
                    progress_message=None,
                    error_json=None,
                    queued_at=now,
                    updated_at=now,
                    state_version=ClinicalSession.state_version + 1,
                )
            )
            db_session.commit()
            return _rowcount(result) == 1

    # -------------------------------------------------------------------------
    def claim_next_queued(self, job_id: str) -> int | None:
        """Claim the oldest queued session for execution under ``job_id``.

        One guarded UPDATE selects and claims the row, so the statement takes
        the write lock before reading and two dispatchers cannot claim it twice.
        """
        oldest_queued = (
            select(ClinicalSession.id)
            .where(ClinicalSession.session_status == "queued")
            .order_by(ClinicalSession.queued_at, ClinicalSession.id)
            .limit(1)
            .scalar_subquery()
        )
        now = _utc_now()
        with self.session_factory() as db_session:
            claimed = db_session.execute(
                update(ClinicalSession)
                .where(
                    ClinicalSession.id == oldest_queued,
                    ClinicalSession.session_status == "queued",
                )
                .values(
                    session_status="running",
                    job_id=job_id,
                    started_at=now,
                    updated_at=now,
                    state_version=ClinicalSession.state_version + 1,
                )
                .returning(ClinicalSession.id)
            ).scalar_one_or_none()
            db_session.commit()
            return int(claimed) if claimed is not None else None

    # -------------------------------------------------------------------------
    def count_sessions_with_status(self, *statuses: str) -> int:
        with self.session_factory() as db_session:
            return int(
                db_session.execute(
                    select(func.count())
                    .select_from(ClinicalSession)
                    .where(ClinicalSession.session_status.in_(statuses))
                ).scalar_one()
            )

    # -------------------------------------------------------------------------
    def get_run_request(self, session_id: int) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.get(ClinicalSession, int(session_id))
            if row is None:
                return None
            return {
                "session_id": int(row.id),
                "status": row.session_status,
                "job_id": row.job_id,
                "patient_image_base64": session_result_data.encode_patient_image(
                    row.patient_image_blob
                ),
                "run_request": session_result_data.parse_session_result_payload(
                    row.run_request_json
                )
                or {},
            }

    # -------------------------------------------------------------------------
    def record_progress(
        self,
        session_id: int,
        *,
        stage: str,
        percent: float | None,
        message: str | None,
    ) -> None:
        with self.session_factory() as db_session:
            db_session.execute(
                update(ClinicalSession)
                .where(
                    ClinicalSession.id == int(session_id),
                    ClinicalSession.session_status == "running",
                )
                .values(
                    progress_stage=stage,
                    progress_percent=percent,
                    progress_message=message,
                    updated_at=_utc_now(),
                    state_version=ClinicalSession.state_version + 1,
                )
            )
            db_session.commit()

    # -------------------------------------------------------------------------
    def transition_session(
        self,
        session_id: int,
        *,
        from_statuses: set[str] | frozenset[str],
        to_status: str,
        error: dict[str, Any] | None = None,
    ) -> bool:
        """Apply one guarded lifecycle transition; False when the guard fails."""
        target = repository_values.normalize_session_status(to_status)
        now = _utc_now()
        values: dict[str, Any] = {
            "session_status": target,
            "updated_at": now,
            "state_version": ClinicalSession.state_version + 1,
        }
        if target in repository_values.TERMINAL_SESSION_STATUSES:
            values["finished_at"] = now
        if error is not None:
            values["error_json"] = session_result_data.serialize_json_payload(error)
        with self.session_factory() as db_session:
            result = db_session.execute(
                update(ClinicalSession)
                .where(
                    ClinicalSession.id == int(session_id),
                    ClinicalSession.session_status.in_(sorted(from_statuses)),
                )
                .values(**values)
            )
            db_session.commit()
            return _rowcount(result) == 1

    # -------------------------------------------------------------------------
    def fail_interrupted_sessions(
        self, *, active_job_ids: set[str], error: dict[str, Any]
    ) -> list[int]:
        """Mark running sessions without a live worker as interrupted failures."""
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(ClinicalSession.id, ClinicalSession.job_id).where(
                    ClinicalSession.session_status == "running"
                )
            ).all()
        interrupted: list[int] = []
        for session_id, job_id in rows:
            if job_id and job_id in active_job_ids:
                continue
            if self.transition_session(
                int(session_id),
                from_statuses={"running"},
                to_status="failed",
                error=error,
            ):
                interrupted.append(int(session_id))
        return interrupted

    # -------------------------------------------------------------------------
    def get_session_state(self, session_id: int) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.get(ClinicalSession, int(session_id))
            if row is None:
                return None
            has_report = bool(
                db_session.execute(select(_report_exists(row.id))).scalar()
            )
            queue_position = None
            if row.session_status == "queued":
                queue_position = 1 + int(
                    db_session.execute(
                        select(func.count())
                        .select_from(ClinicalSession)
                        .where(
                            ClinicalSession.session_status == "queued",
                            or_(
                                ClinicalSession.queued_at < row.queued_at,
                                and_(
                                    ClinicalSession.queued_at == row.queued_at,
                                    ClinicalSession.id < row.id,
                                ),
                            ),
                        )
                    ).scalar_one()
                )
            return {
                "session_id": int(row.id),
                "status": row.session_status,
                "patient_name": row.patient_name,
                "visit_date": row.visit_date,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "queued_at": row.queued_at,
                "started_at": row.started_at,
                "finished_at": row.finished_at,
                "queue_position": queue_position,
                "job_id": row.job_id,
                "progress": row.progress_percent,
                "progress_stage": row.progress_stage,
                "progress_message": row.progress_message,
                "error": session_result_data.parse_session_result_payload(
                    row.error_json
                ),
                "has_report": has_report,
                "state_version": int(row.state_version or 0),
            }

    # -------------------------------------------------------------------------
    def get_session_input(self, session_id: int) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.get(ClinicalSession, int(session_id))
            if row is None:
                return None
            return {
                "session_id": int(row.id),
                "status": row.session_status,
                "patient_name": row.patient_name,
                "visit_date": row.visit_date,
                "clinical_input": session_input_text(row),
                "patient_image_base64": session_result_data.encode_patient_image(
                    row.patient_image_blob
                ),
                "updated_at": row.updated_at,
            }

    # -------------------------------------------------------------------------
    def list_sessions(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
        statuses: Sequence[str] | None = None,
        date_field: str = "created",
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        created_before: datetime | None = None,
        sort: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        conditions: list[Any] = []
        search_pattern = _build_search_pattern(search)
        if search_pattern is not None:
            section_match = exists(
                select(1).where(
                    ClinicalSessionSection.session_id == ClinicalSession.id,
                    func.lower(func.coalesce(ClinicalSessionSection.content, "")).like(
                        search_pattern, escape="\\"
                    ),
                )
            )
            result_payload_match = exists(
                select(1).where(
                    ClinicalSessionResult.session_id == ClinicalSession.id,
                    func.lower(
                        func.coalesce(ClinicalSessionResult.payload_json, "")
                    ).like(search_pattern, escape="\\"),
                )
            )
            conditions.append(
                or_(
                    func.lower(func.coalesce(ClinicalSession.patient_name, "")).like(
                        search_pattern, escape="\\"
                    ),
                    sql_cast(ClinicalSession.id, String).like(
                        search_pattern, escape="\\"
                    ),
                    section_match,
                    result_payload_match,
                )
            )
        valid_statuses = [
            value
            for value in (statuses or ())
            if value in repository_values.SESSION_LIFECYCLE_STATUSES
        ]
        if valid_statuses:
            conditions.append(ClinicalSession.session_status.in_(valid_statuses))
        if created_before is not None:
            conditions.append(ClinicalSession.created_at < created_before)
        if date_field == "visit":
            visit_column = ClinicalSession.visit_date
            if date_from is not None:
                conditions.append(visit_column >= date_from.date())
            if date_to is not None:
                conditions.append(visit_column < date_to.date())
        else:
            date_column = (
                ClinicalSession.updated_at
                if date_field == "updated"
                else ClinicalSession.created_at
            )
            if date_from is not None:
                conditions.append(date_column >= date_from)
            if date_to is not None:
                conditions.append(date_column < date_to)
        order_by = {
            "created_desc": (ClinicalSession.created_at.desc(), ClinicalSession.id.desc()),
            "created_asc": (ClinicalSession.created_at.asc(), ClinicalSession.id.asc()),
        }.get(sort or "", (ClinicalSession.updated_at.desc(), ClinicalSession.id.desc()))
        db_session = self.session_factory()
        try:
            report_exists = _report_exists(ClinicalSession.id)
            timeline_exists = exists(
                select(1).where(
                    ClinicalSessionTimeline.session_id == ClinicalSession.id
                )
            )
            version_number = (
                select(func.max(ClinicalSessionVersion.version_number))
                .where(ClinicalSessionVersion.session_id == ClinicalSession.id)
                .scalar_subquery()
            )
            sessions_stmt = select(
                ClinicalSession,
                version_number.label("version_number"),
                report_exists.label("has_report"),
                timeline_exists.label("has_timeline"),
            )
            count_stmt = select(func.count()).select_from(ClinicalSession)
            if conditions:
                combined = and_(*conditions)
                sessions_stmt = sessions_stmt.where(combined)
                count_stmt = count_stmt.where(combined)
            total_rows = int(db_session.execute(count_stmt).scalar_one())
            rows = db_session.execute(
                sessions_stmt.order_by(*order_by)
                .offset(safe_offset)
                .limit(safe_limit)
            ).all()
            items = [
                {
                    "session_id": int(session_row.id),
                    "patient_name": repository_values.normalize_string(
                        session_row.patient_name
                    ),
                    "visit_date": session_row.visit_date,
                    "session_timestamp": session_row.session_timestamp,
                    "created_at": session_row.created_at,
                    "updated_at": session_row.updated_at,
                    "version": int(version_value or 1),
                    "status": session_row.session_status,
                    "progress": session_row.progress_percent,
                    "progress_stage": session_row.progress_stage,
                    "total_duration": repository_values.to_float(
                        session_row.total_duration
                    ),
                    "has_report": bool(has_report),
                    "has_timeline": bool(has_timeline),
                    "can_generate_timeline": bool(
                        session_input_text(session_row).strip()
                    ),
                }
                for session_row, version_value, has_report, has_timeline in rows
            ]
            return items, total_rows
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def get_session_agent_revision_id(self, session_id: int) -> int | None:
        with self.session_factory() as db_session:
            value = db_session.execute(
                select(ClinicalSession.agent_revision_id).where(
                    ClinicalSession.id == int(session_id)
                )
            ).scalar_one_or_none()
            return int(value) if value is not None else None

    # -------------------------------------------------------------------------
    def get_session_detail(self, session_id: int) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        db_session = self.session_factory()
        try:
            session_row = db_session.execute(
                select(ClinicalSession).where(ClinicalSession.id == safe_session_id)
            ).scalar_one_or_none()
            if session_row is None:
                return None
            version_number = db_session.execute(
                select(func.max(ClinicalSessionVersion.version_number)).where(
                    ClinicalSessionVersion.session_id == safe_session_id
                )
            ).scalar_one_or_none()
            section_rows = db_session.execute(
                select(
                    ClinicalSessionSection.section_kind, ClinicalSessionSection.content
                ).where(ClinicalSessionSection.session_id == safe_session_id)
            ).all()
            sections = {
                str(kind): repository_values.normalize_string(content) or ""
                for kind, content in section_rows
            }
            payload = self.get_session_result_payload(safe_session_id) or {}
            metadata = (
                session_result_data.parse_session_result_payload(
                    session_row.metadata_json
                )
                or {}
            )
            session_text = (
                repository_values.normalize_string(payload.get("original_session_text"))
                or session_row.clinical_input
                or ""
            )
            official_report_text = repository_values.normalize_string(
                payload.get("report")
            )
            return {
                "session_id": safe_session_id,
                "patient_name": repository_values.normalize_string(
                    session_row.patient_name
                ),
                "visit_date": session_row.visit_date,
                "session_timestamp": session_row.session_timestamp,
                "version": int(version_number or 1),
                "status": session_row.session_status,
                "created_at": session_row.created_at,
                "updated_at": session_row.updated_at,
                "text_extraction_model": repository_values.normalize_string(
                    session_row.text_extraction_model
                ),
                "clinical_model": repository_values.normalize_string(
                    session_row.clinical_model
                ),
                "metadata": metadata,
                "sections": sections,
                "session_text": session_text,
                "source_clinical_text": session_text,
                "result_payload": payload,
                "report": official_report_text,
                "official_report_text": official_report_text,
            }
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def get_session_result_payload(self, session_id: int) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            payload_json = db_session.execute(
                select(ClinicalSessionResult.payload_json).where(
                    ClinicalSessionResult.session_id == int(session_id)
                )
            ).scalar_one_or_none()
            return session_result_data.parse_session_result_payload(payload_json)

    # -------------------------------------------------------------------------
    def upsert_session_result_payload(
        self, session_id: int, payload: dict[str, Any]
    ) -> bool:
        serialized_payload = session_result_data.serialize_json_payload(payload)
        if serialized_payload is None:
            return False
        with self.session_factory() as db_session:
            # Write first so concurrent analyses never upgrade a stale snapshot.
            updated = db_session.execute(
                update(ClinicalSessionResult)
                .where(ClinicalSessionResult.session_id == int(session_id))
                .values(payload_json=serialized_payload)
            )
            if _rowcount(updated) == 0:
                db_session.add(
                    ClinicalSessionResult(
                        session_id=int(session_id), payload_json=serialized_payload
                    )
                )
            try:
                db_session.commit()
            except IntegrityError:
                db_session.rollback()
                return False
            return True

    # -------------------------------------------------------------------------
    def update_session_text_and_metadata(
        self,
        session_id: int,
        *,
        session_text: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
            existing = db_session.get(ClinicalSession, safe_session_id)
            if existing is None:
                return None
            if session_text is not None:
                result_row = db_session.execute(
                    select(ClinicalSessionResult).where(
                        ClinicalSessionResult.session_id == safe_session_id
                    )
                ).scalar_one_or_none()
                payload = (
                    session_result_data.parse_session_result_payload(
                        result_row.payload_json
                    )
                    if result_row is not None
                    else {}
                ) or {}
                payload["original_session_text"] = str(session_text).strip()
                serialized_payload = (
                    session_result_data.serialize_json_payload(payload) or "{}"
                )
                if result_row is None:
                    db_session.add(
                        ClinicalSessionResult(
                            session_id=safe_session_id, payload_json=serialized_payload
                        )
                    )
                else:
                    result_row.payload_json = serialized_payload
            if metadata is not None:
                existing.metadata_json = session_result_data.serialize_json_payload(
                    metadata or {}
                )
            db_session.commit()
        return self.get_session_detail(safe_session_id)

    # -------------------------------------------------------------------------
    def update_session_metadata(
        self, session_id: int, *, metadata: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
            existing = db_session.get(ClinicalSession, safe_session_id)
            if existing is None:
                return None
            if metadata is not None:
                existing.metadata_json = session_result_data.serialize_json_payload(
                    metadata or {}
                )
            db_session.commit()
        return self.get_session_detail(safe_session_id)

    # -------------------------------------------------------------------------
    @staticmethod
    def _serialize_attachment(row: ClinicalSessionAttachment) -> dict[str, Any]:
        return {
            "attachment_id": int(row.id),
            "session_id": int(row.session_id),
            "version_id": int(row.version_id),
            "kind": row.kind,
            "file_name": row.file_name,
            "media_type": row.media_type,
            "size_bytes": int(row.size_bytes),
            "sha256": row.sha256,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    # -------------------------------------------------------------------------
    def add_attachment(
        self,
        session_id: int,
        *,
        version_id: int,
        kind: str,
        file_name: str,
        media_type: str,
        content: bytes,
    ) -> dict[str, Any] | None:
        """Store an attachment once per session version and content hash."""
        safe_session_id = int(session_id)
        digest = hashlib.sha256(content).hexdigest()
        with self.session_factory() as db_session:
            if db_session.get(ClinicalSession, safe_session_id) is None:
                return None
            row = db_session.execute(
                select(ClinicalSessionAttachment).where(
                    ClinicalSessionAttachment.session_id == safe_session_id,
                    ClinicalSessionAttachment.version_id == int(version_id),
                    ClinicalSessionAttachment.sha256 == digest,
                )
            ).scalar_one_or_none()
            if row is None:
                row = ClinicalSessionAttachment(
                    session_id=safe_session_id,
                    version_id=int(version_id),
                    kind=kind,
                    file_name=file_name,
                    media_type=media_type,
                    size_bytes=len(content),
                    sha256=digest,
                    content=content,
                )
                db_session.add(row)
                db_session.commit()
                db_session.refresh(row)
            return self._serialize_attachment(row)

    # -------------------------------------------------------------------------
    def list_attachments(self, session_id: int) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(ClinicalSessionAttachment)
                .where(ClinicalSessionAttachment.session_id == int(session_id))
                .order_by(ClinicalSessionAttachment.id)
            ).scalars()
            return [self._serialize_attachment(row) for row in rows]

    # -------------------------------------------------------------------------
    def get_attachment_content(
        self, session_id: int, attachment_id: int
    ) -> tuple[dict[str, Any], bytes] | None:
        with self.session_factory() as db_session:
            row = db_session.get(ClinicalSessionAttachment, int(attachment_id))
            if row is None or int(row.session_id) != int(session_id):
                return None
            return self._serialize_attachment(row), bytes(row.content)

    # -------------------------------------------------------------------------
    def delete_attachment(self, session_id: int, attachment_id: int) -> bool:
        with self.session_factory() as db_session:
            row = db_session.get(ClinicalSessionAttachment, int(attachment_id))
            if row is None or int(row.session_id) != int(session_id):
                return False
            db_session.delete(row)
            db_session.commit()
            return True

    # -------------------------------------------------------------------------
    def delete_session(
        self, session_id: int, *, db_session: Session | None = None
    ) -> bool:
        """Delete one session, inside the caller's transaction when given."""
        if db_session is None:
            with unit_of_work(
                session_factory=self.session_factory, immediate=True
            ) as owned_session:
                return self.delete_session(session_id, db_session=owned_session)
        existing = db_session.get(ClinicalSession, int(session_id))
        if existing is None:
            return False
        if existing.session_status in repository_values.ACTIVE_SESSION_STATUSES:
            raise SessionStateConflictError(
                "An active clinical session cannot be deleted."
            )
        db_session.delete(existing)
        db_session.flush()
        return True

    # -------------------------------------------------------------------------
    def persist_session_sections(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        issues_raw = session_data.get("issues")
        if isinstance(issues_raw, (list, dict)):
            issues_content: str | None = json.dumps(issues_raw, ensure_ascii=False)
        else:
            issues_content = repository_values.normalize_string(issues_raw)
        payload = {
            "anamnesis": session_data.get("anamnesis"),
            "drugs": session_data.get("drugs"),
            "laboratory_analysis": session_data.get("laboratory_analysis"),
            "final_report": session_data.get("final_report"),
            "issues": issues_content,
        }
        for section_kind, value in payload.items():
            content = repository_values.normalize_string(value)
            if content is not None:
                db_session.add(
                    ClinicalSessionSection(
                        session_id=session_id,
                        section_kind=section_kind,
                        content=content,
                    )
                )

    # -------------------------------------------------------------------------
    def persist_session_labs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        result_payload = session_data.get("session_result_payload")
        timeline_raw = (
            result_payload.get("lab_timeline")
            if isinstance(result_payload, dict)
            else None
        )
        if not isinstance(timeline_raw, list):
            return
        persisted_codes = {
            "ALT": "alt",
            "AST": "ast",
            "ALP": "alp",
            "TBIL": "tbil",
            "DBIL": "dbil",
            "GGT": "ggt",
            "INR": "inr",
            "ALB": "albumin",
        }
        seen_observations: set[tuple[Any, ...]] = set()
        for ordinal, item in enumerate(timeline_raw):
            if not isinstance(item, dict):
                continue
            marker_name = repository_values.normalize_string(item.get("marker_name"))
            lab_code = persisted_codes.get(marker_name.upper()) if marker_name else None
            if lab_code is None:
                continue
            value_raw = repository_values.normalize_string(
                item.get("value")
            ) or repository_values.normalize_string(item.get("value_text"))
            upper_limit_raw = repository_values.normalize_string(
                item.get("upper_limit_normal")
            ) or repository_values.normalize_string(item.get("upper_limit_text"))
            if value_raw is None and upper_limit_raw is None:
                continue
            observation_at = session_result_data.parse_datetime(
                item.get("observation_at")
                or item.get("date")
                or item.get("sample_date")
            )
            unit = repository_values.normalize_string(item.get("unit"))
            observation_key = (lab_code, observation_at, value_raw, upper_limit_raw, unit)
            if observation_key in seen_observations:
                continue
            seen_observations.add(observation_key)
            db_session.add(
                ClinicalLabObservation(
                    session_id=session_id,
                    marker_code=lab_code,
                    observation_at=observation_at,
                    value_numeric=repository_values.to_float(value_raw),
                    value_text=value_raw,
                    unit=unit,
                    upper_limit_numeric=repository_values.to_float(upper_limit_raw),
                    source_ordinal=ordinal,
                    metadata_json={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "value",
                            "value_text",
                            "upper_limit_normal",
                            "upper_limit_text",
                        }
                    },
                )
            )

    # -------------------------------------------------------------------------
    def persist_session_drugs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        payload = session_data.get("matched_drugs")
        records: list[dict[str, Any]] = []
        if isinstance(payload, list):
            records.extend(
                item if isinstance(item, dict) else {"raw_drug_name": item}
                for item in payload
                if isinstance(item, (dict, str))
            )
        if not records and isinstance(session_data.get("detected_drugs"), list):
            records.extend(
                {"raw_drug_name": item}
                for item in session_data["detected_drugs"]
                if isinstance(item, str)
            )
        seen: set[str] = set()
        for mention_ordinal, item in enumerate(records):
            raw_drug_name = repository_values.normalize_string(
                item.get("raw_drug_name") or item.get("name")
            )
            if raw_drug_name is None:
                continue
            raw_drug_name_norm = normalize_drug_name(raw_drug_name)
            duplicate_mention = not raw_drug_name_norm or raw_drug_name_norm in seen
            seen.add(raw_drug_name_norm)
            matched_drug_name = repository_values.normalize_string(
                item.get("matched_drug_name")
            )
            rxcui = repository_values.normalize_string(
                item.get("rxcui")
                or item.get("accepted_rxnav_rxcui")
                or item.get("rxnorm_rxcui")
            )
            nbk_id = repository_values.normalize_string(item.get("nbk_id"))
            resolved_drug_id = repository_values.to_int(item.get("drug_id"))
            match_reason = repository_values.normalize_string(item.get("match_reason"))
            terminology = item.get("terminology")
            terminology = terminology if isinstance(terminology, dict) else {}
            db_session.add(
                ClinicalDrugMention(
                    session_id=session_id,
                    mention_ordinal=mention_ordinal,
                    raw_name=raw_drug_name,
                    normalized_name=raw_drug_name_norm,
                    drug_id=resolved_drug_id,
                    match_status=str(terminology.get("status") or "unresolved"),
                    confidence=None,
                    match_reason=match_reason,
                    evidence_json={
                        "matched_drug_name": matched_drug_name,
                        "rxcui": rxcui,
                        "accepted_rxnav_rxcui": repository_values.normalize_string(
                            item.get("accepted_rxnav_rxcui")
                        ),
                        "terminology": terminology or None,
                        "nbk_id": nbk_id,
                        "source_provenance": item.get("source_provenance") or {},
                        "extraction_metadata": item.get("extraction_metadata") or [],
                        "duplicate_mention": duplicate_mention,
                    },
                )
            )

    # -------------------------------------------------------------------------
    def persist_session_result_payload(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        payload = session_data.get("session_result_payload")
        serialized_payload = session_result_data.serialize_json_payload(payload)
        if serialized_payload is None:
            return
        db_session.add(
            ClinicalSessionResult(
                session_id=session_id, payload_json=serialized_payload
            )
        )
        current_version = (
            db_session.execute(
                select(ClinicalSessionVersion)
                .where(ClinicalSessionVersion.session_id == int(session_id))
                .order_by(ClinicalSessionVersion.version_number.desc())
            )
            .scalars()
            .first()
        )
        if current_version is not None and isinstance(payload, dict):
            current_version.report_text = repository_values.normalize_string(
                payload.get("report")
            )
            current_version.hepatic_pattern = repository_values.normalize_string(
                payload.get("hepatic_pattern")
            )
            current_version.total_duration = repository_values.to_float(
                payload.get("total_duration")
            )
            current_version.metadata_json = session_result_data.serialize_json_payload(
                payload.get("metadata")
            )
