from __future__ import annotations

from typing import Any

from sqlalchemy import select

from domain.patient_timeline import PatientTimeline, SessionTimelinePreview
from repositories.schemas.clinical import (
    ClinicalSession,
    ClinicalSessionResult,
    ClinicalSessionSection,
    ClinicalSessionTimeline,
)

###############################################################################
def _validate_timeline_payload(
    payload: dict[str, Any] | None,
) -> PatientTimeline | None:
    if not isinstance(payload, dict):
        return None
    try:
        return PatientTimeline.model_validate(payload)
    except Exception:
        return None

###############################################################################
def _build_timeline_preview_payload(payload: PatientTimeline) -> dict[str, Any]:
    dated_events = [event.event_date for event in payload.events if event.event_date]
    sorted_dates = sorted(dated_events)
    title = payload.events[0].title if payload.events else None
    source_evidence_event_count = sum(
        1 for event in payload.events if event.source_evidence and event.source_evidence.strip()
    )
    return SessionTimelinePreview(
        timeline_id=payload.timeline_id,
        session_id=payload.session_id,
        generated_at=payload.generated_at,
        generation_status=payload.generation_status,
        generation_note=payload.generation_note,
        source_model=payload.source_model,
        source_kind=payload.source_kind,
        model_provider=payload.model_provider,
        event_count=len(payload.events),
        start_date=sorted_dates[0] if sorted_dates else None,
        end_date=sorted_dates[-1] if sorted_dates else None,
        title=title,
        source_evidence_event_count=source_evidence_event_count,
        missing_evidence_event_count=len(payload.events) - source_evidence_event_count,
        uncertain_event_count=sum(
            1 for event in payload.events if event.timing_type in {"uncertain", "ordering"}
        ),
        undated_event_count=sum(1 for event in payload.events if not event.event_date),
    ).model_dump(mode="json")

###############################################################################
def _timeline_from_row(
    self,
    *,
    row: ClinicalSessionTimeline,
    session_id: int,
) -> PatientTimeline | None:
    timeline = _validate_timeline_payload(
        self.parse_session_result_payload(row.timeline_payload_json)
    )
    if timeline is None:
        return None
    return PatientTimeline(
        **{
            **timeline.model_dump(),
            "timeline_id": int(row.id),
            "session_id": session_id,
            "generated_at": row.generated_at,
            "generation_status": row.generation_status,
            "generation_note": self.normalize_string(row.generation_note),
            "source_model": self.normalize_string(row.source_model),
            "source_kind": self.normalize_string(row.source_kind),
            "model_provider": self.normalize_string(row.model_provider),
        }
    )

###############################################################################
def list_session_timelines(self, session_id: int) -> list[dict[str, Any]]:
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        session_row = db_session.get(ClinicalSession, safe_session_id)
        if session_row is None:
            return []
        rows = (
            db_session.execute(
                select(ClinicalSessionTimeline)
                .where(ClinicalSessionTimeline.session_id == safe_session_id)
                .order_by(
                    ClinicalSessionTimeline.generated_at.desc(),
                    ClinicalSessionTimeline.id.desc(),
                )
            )
            .scalars()
            .all()
        )
        previews: list[dict[str, Any]] = []
        for row in rows:
            timeline = _timeline_from_row(self, row=row, session_id=safe_session_id)
            if timeline is None:
                continue
            previews.append(_build_timeline_preview_payload(timeline))
        return previews
    finally:
        db_session.close()

###############################################################################
def get_session_timeline_record(
    self,
    session_id: int,
    timeline_id: int,
) -> dict[str, Any] | None:
    safe_session_id = int(session_id)
    safe_timeline_id = int(timeline_id)
    db_session = self.session_factory()
    try:
        row = db_session.execute(
            select(ClinicalSessionTimeline).where(
                ClinicalSessionTimeline.session_id == safe_session_id,
                ClinicalSessionTimeline.id == safe_timeline_id,
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        timeline = _timeline_from_row(self, row=row, session_id=safe_session_id)
        return timeline.model_dump(mode="json") if timeline is not None else None
    finally:
        db_session.close()

###############################################################################
def get_latest_session_timeline_record(self, session_id: int) -> dict[str, Any] | None:
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        row = (
            db_session.execute(
                select(ClinicalSessionTimeline)
                .where(ClinicalSessionTimeline.session_id == safe_session_id)
                .order_by(
                    ClinicalSessionTimeline.generated_at.desc(),
                    ClinicalSessionTimeline.id.desc(),
                )
                .limit(1)
            )
            .scalars()
            .first()
        )
        if row is None:
            return None
        timeline = _timeline_from_row(self, row=row, session_id=safe_session_id)
        return timeline.model_dump(mode="json") if timeline is not None else None
    finally:
        db_session.close()

###############################################################################
def create_session_timeline_record(
    self, session_id: int, payload: dict[str, Any]
) -> dict[str, Any] | None:
    safe_session_id = int(session_id)
    timeline = _validate_timeline_payload(payload)
    if timeline is None:
        return None
    serialized_payload = self.serialize_json_payload(
        {
            **timeline.model_dump(mode="json"),
            "timeline_id": None,
        }
    )
    if serialized_payload is None:
        return None
    db_session = self.session_factory()
    try:
        existing_session = db_session.get(ClinicalSession, safe_session_id)
        if existing_session is None:
            return None
        record = ClinicalSessionTimeline(
            session_id=safe_session_id,
            generated_at=timeline.generated_at,
            generation_status=timeline.generation_status,
            generation_note=timeline.generation_note,
            source_model=timeline.source_model,
            source_kind=timeline.source_kind,
            model_provider=timeline.model_provider,
            timeline_payload_json=serialized_payload,
        )
        db_session.add(record)
        db_session.commit()
        db_session.refresh(record)
        return PatientTimeline(
            **{
                **timeline.model_dump(),
                "timeline_id": int(record.id),
                "session_id": safe_session_id,
            }
        ).model_dump(mode="json")
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()

###############################################################################
def delete_session_timeline_record(
    self,
    session_id: int,
    timeline_id: int,
) -> bool:
    db_session = self.session_factory()
    try:
        row = db_session.execute(
            select(ClinicalSessionTimeline).where(
                ClinicalSessionTimeline.session_id == int(session_id),
                ClinicalSessionTimeline.id == int(timeline_id),
            )
        ).scalar_one_or_none()
        if row is None:
            return False
        db_session.delete(row)
        db_session.commit()
        return True
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()

###############################################################################
def get_session_timeline_source(self, session_id: int) -> dict[str, Any] | None:
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        row = db_session.execute(
            select(ClinicalSession)
            .where(ClinicalSession.id == safe_session_id)
        ).first()
        if row is None:
            return None
        session_row = row[0]
        payload_json = db_session.execute(
            select(ClinicalSessionResult.payload_json).where(
                ClinicalSessionResult.session_id == safe_session_id
            )
        ).scalar_one_or_none()
        session_payload = self.parse_session_result_payload(payload_json) or {}
        section_rows = db_session.execute(
            select(
                ClinicalSessionSection.section_kind, ClinicalSessionSection.content
            ).where(ClinicalSessionSection.session_id == safe_session_id)
        ).all()
        sections = {
            str(kind): self.normalize_string(content)
            for kind, content in section_rows
            if self.normalize_string(kind) is not None
        }
        return {
            "session_id": safe_session_id,
            "patient_name": self.normalize_string(session_row.patient_name),
            "visit_date": session_row.visit_date.isoformat()
            if session_row.visit_date
            else None,
            "session_timestamp": (
                session_row.session_timestamp.isoformat()
                if session_row.session_timestamp
                else None
            ),
            "anamnesis": self.normalize_string(session_row.anamnesis),
            "drugs": self.normalize_string(session_row.drugs_text),
            "laboratory_analysis": self.normalize_string(
                session_row.laboratory_analysis
            ),
            "text_extraction_model": self.normalize_string(
                session_row.text_extraction_model
            ),
            "clinical_model": self.normalize_string(session_row.clinical_model),
            "sections": sections,
            "session_result_payload": session_payload,
        }
    finally:
        db_session.close()
