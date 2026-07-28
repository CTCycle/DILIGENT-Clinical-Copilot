from __future__ import annotations

from typing import Any

from sqlalchemy import select

from domain.patient_timeline import PatientTimeline
from repositories.context import RepositoryContext
from repositories.schemas.clinical import (
    ClinicalSession,
    ClinicalSessionResult,
    ClinicalSessionSection,
    ClinicalSessionTimeline,
)
from repositories.serialization.session_timelines import (
    build_timeline_preview_payload,
    serialize_timeline_payload,
    timeline_from_row,
    validate_timeline_payload,
    parse_timeline_payload,
)
from repositories.values import normalize_string


class SessionTimelineRepository:
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    def list_session_timelines(self, session_id: int) -> list[dict[str, Any]]:
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
            if db_session.get(ClinicalSession, safe_session_id) is None:
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
                timeline = timeline_from_row(row, session_id=safe_session_id)
                if timeline is not None:
                    previews.append(build_timeline_preview_payload(timeline))
            return previews

    def get_session_timeline_record(
        self, session_id: int, timeline_id: int
    ) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        safe_timeline_id = int(timeline_id)
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(ClinicalSessionTimeline).where(
                    ClinicalSessionTimeline.session_id == safe_session_id,
                    ClinicalSessionTimeline.id == safe_timeline_id,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            timeline = timeline_from_row(row, session_id=safe_session_id)
            return timeline.model_dump(mode="json") if timeline is not None else None

    def get_latest_session_timeline_record(
        self, session_id: int
    ) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
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
            timeline = timeline_from_row(row, session_id=safe_session_id)
            return timeline.model_dump(mode="json") if timeline is not None else None

    def create_session_timeline_record(
        self, session_id: int, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        timeline = validate_timeline_payload(payload)
        if timeline is None:
            return None
        serialized_payload = serialize_timeline_payload(
            {**timeline.model_dump(mode="json"), "timeline_id": None}
        )
        if serialized_payload is None:
            return None
        with self.session_factory() as db_session:
            if db_session.get(ClinicalSession, safe_session_id) is None:
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

    def delete_session_timeline_record(self, session_id: int, timeline_id: int) -> bool:
        with self.session_factory() as db_session:
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

    def get_session_timeline_source(self, session_id: int) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
            session_row = db_session.execute(
                select(ClinicalSession).where(ClinicalSession.id == safe_session_id)
            ).scalar_one_or_none()
            if session_row is None:
                return None
            payload_json = db_session.execute(
                select(ClinicalSessionResult.payload_json).where(
                    ClinicalSessionResult.session_id == safe_session_id
                )
            ).scalar_one_or_none()
            session_payload = parse_timeline_payload(payload_json) or {}
            section_rows = db_session.execute(
                select(
                    ClinicalSessionSection.section_kind,
                    ClinicalSessionSection.content,
                ).where(ClinicalSessionSection.session_id == safe_session_id)
            ).all()
            sections = {
                str(kind): normalize_string(content)
                for kind, content in section_rows
                if normalize_string(kind) is not None
            }
            return {
                "session_id": safe_session_id,
                "patient_name": normalize_string(session_row.patient_name),
                "visit_date": session_row.visit_date.isoformat()
                if session_row.visit_date
                else None,
                "session_timestamp": (
                    session_row.session_timestamp.isoformat()
                    if session_row.session_timestamp
                    else None
                ),
                "anamnesis": normalize_string(session_row.anamnesis),
                "drugs": normalize_string(session_row.drugs_text),
                "laboratory_analysis": normalize_string(session_row.laboratory_analysis),
                "text_extraction_model": normalize_string(session_row.text_extraction_model),
                "clinical_model": normalize_string(session_row.clinical_model),
                "sections": sections,
                "session_result_payload": session_payload,
            }
