from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from domain.patient_timeline import (
    PatientTimelineEvent,
    PatientTimelineExtraction,
)
from services.clinical.timeline import PatientTimelineExtractor

###############################################################################
class FakeTimelineClient:

    # -------------------------------------------------------------------------
    def __init__(self, payload: PatientTimelineExtraction) -> None:
        self.payload = payload
        self.call_count = 0
        self.last_kwargs: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientTimelineExtraction:
        self.last_kwargs = kwargs
        self.call_count += 1
        return self.payload

###############################################################################
def test_timeline_extractor_sorts_and_deduplicates_events() -> None:
    extractor = PatientTimelineExtractor(
        client=FakeTimelineClient(
            PatientTimelineExtraction(
                events=[
                    PatientTimelineEvent(
                        event_id="b",
                        title="ALT peak",
                        event_type="lab",
                        event_date="2025-03-01",
                        source_evidence="ALT reached 450 U/L on 2025-03-01.",
                        confidence=0.6,
                    ),
                    PatientTimelineEvent(
                        event_id="a",
                        title="Therapy started",
                        event_type="therapy",
                        event_date="2025-01-10",
                        source_evidence="Therapy started on 2025-01-10.",
                        confidence=0.8,
                    ),
                    PatientTimelineEvent(
                        event_id="c",
                        title="ALT peak",
                        event_type="lab",
                        event_date="2025-03-01",
                        source_evidence="ALT reached 450 U/L on 2025-03-01.",
                        confidence=0.9,
                    ),
                ]
            )
        )
    )

    result = asyncio.run(
        extractor.extract_timeline(
            session_id=7,
            source_payload={"anamnesis": "timeline source"},
        )
    )

    assert result.session_id == 7
    assert isinstance(result.generated_at, datetime)
    assert result.generated_at.tzinfo in {UTC, None}
    assert len(result.events) == 2
    assert result.events[0].title == "Therapy started"
    assert result.events[0].sort_order == 0
    assert result.events[1].title == "ALT peak"
    assert result.events[1].confidence == 0.9

###############################################################################
def test_timeline_extractor_rejects_events_without_source_evidence() -> None:
    extractor = PatientTimelineExtractor(
        client=FakeTimelineClient(
            PatientTimelineExtraction(
                events=[
                    PatientTimelineEvent(
                        event_id="missing-evidence",
                        title="Therapy started",
                        event_type="therapy",
                        event_date="2025-01-10",
                    ),
                    PatientTimelineEvent(
                        event_id="grounded",
                        title="ALT peak",
                        event_type="lab",
                        event_date="2025-03-01",
                        source_evidence="ALT 450 U/L on 2025-03-01.",
                    ),
                ]
            )
        )
    )

    result = asyncio.run(
        extractor.extract_timeline(
            session_id=9,
            source_payload={"laboratory_analysis": "ALT 450 U/L on 2025-03-01."},
        )
    )

    assert [event.event_id for event in result.events] == ["grounded"]

###############################################################################
def test_normalize_date_token_keeps_month_precision_without_promoting_day() -> None:
    assert PatientTimelineExtractor.normalize_date_token("2025-02") == "2025-02"


def test_timeline_sort_orders_year_month_and_day_without_changing_display_values() -> None:
    extractor = PatientTimelineExtractor(client=FakeTimelineClient(PatientTimelineExtraction()))
    events = [
        PatientTimelineEvent(event_id="relative", title="Later", relative_time="after discharge", source_evidence="Later."),
        PatientTimelineEvent(event_id="day", title="Day", event_date="2025-02-03", source_evidence="Day."),
        PatientTimelineEvent(event_id="month", title="Month", event_date="2025-02", source_evidence="Month."),
        PatientTimelineEvent(event_id="year", title="Year", event_date="2025", source_evidence="Year."),
    ]

    normalized = extractor.normalize_events(events)

    assert [event.event_id for event in normalized] == ["year", "month", "day", "relative"]
    assert [event.event_date for event in normalized[:3]] == ["2025", "2025-02", "2025-02-03"]


def test_timeline_prompt_uses_canonical_json_and_hash() -> None:
    client = FakeTimelineClient(PatientTimelineExtraction())
    extractor = PatientTimelineExtractor(client=client)

    asyncio.run(extractor.extract_timeline(session_id=5, source_payload={"b": 2, "a": "x"}))

    prompt = client.last_kwargs["user_prompt"]
    assert '{"a":"x","b":2}' in prompt
    assert "Source payload SHA-256:" in prompt
    assert "'a':" not in prompt
