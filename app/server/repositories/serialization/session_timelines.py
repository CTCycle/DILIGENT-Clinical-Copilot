from __future__ import annotations

import json
from typing import Any

from domain.patient_timeline import PatientTimeline, SessionTimelinePreview
from domain.timeline_dates import timeline_date_sort_key
from repositories.schemas.clinical import ClinicalSessionTimeline
from repositories.values import normalize_string


def parse_timeline_payload(payload_json: str | None) -> dict[str, Any] | None:
    normalized_payload = normalize_string(payload_json)
    if normalized_payload is None:
        return None
    try:
        payload = json.loads(normalized_payload)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def serialize_timeline_payload(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return normalize_string(payload)
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return normalize_string(payload)


def validate_timeline_payload(payload: dict[str, Any] | None) -> PatientTimeline | None:
    if not isinstance(payload, dict):
        return None
    try:
        return PatientTimeline.model_validate(payload)
    except Exception:
        return None


def build_timeline_preview_payload(payload: PatientTimeline) -> dict[str, Any]:
    dated_events = [event.event_date for event in payload.events if event.event_date]
    sorted_dates = sorted(dated_events, key=timeline_date_sort_key)
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


def timeline_from_row(
    row: ClinicalSessionTimeline,
    *,
    session_id: int,
) -> PatientTimeline | None:
    timeline = validate_timeline_payload(parse_timeline_payload(row.timeline_payload_json))
    if timeline is None:
        return None
    return PatientTimeline(
        **{
            **timeline.model_dump(),
            "timeline_id": int(row.id),
            "session_id": session_id,
            "generated_at": row.generated_at,
            "generation_status": row.generation_status,
            "generation_note": normalize_string(row.generation_note),
            "source_model": normalize_string(row.source_model),
            "source_kind": normalize_string(row.source_kind),
            "model_provider": normalize_string(row.model_provider),
        }
    )
