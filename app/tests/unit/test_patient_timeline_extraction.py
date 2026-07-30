from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from domain.patient_timeline import (
    PatientTimelineEvent,
    PatientTimelineExtraction,
    SessionTimelineModelOverrides,
)
from repositories.serialization.session_timelines import build_timeline_preview_payload
from domain.patient_timeline import PatientTimeline
from domain.timeline_dates import normalize_timeline_interval
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

###############################################################################
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

###############################################################################
def test_timeline_uses_the_single_explicit_date_in_source_evidence() -> None:
    extractor = PatientTimelineExtractor(client=FakeTimelineClient(PatientTimelineExtraction()))
    event = PatientTimelineEvent(
        event_id="alt",
        title="ALT 300 U/L",
        event_type="lab",
        event_date="2026-07-15",
        source_evidence="2026-07-21 ALT 300 U/L.",
    )

    normalized = extractor.normalize_events([event])

    assert normalized[0].event_date == "2026-07-21"

###############################################################################
def test_timeline_preserves_precise_model_date_when_evidence_only_matches_year() -> None:
    extractor = PatientTimelineExtractor(client=FakeTimelineClient(PatientTimelineExtraction()))
    event = PatientTimelineEvent(
        event_id="natural-date",
        title="Symptoms began",
        event_type="disease",
        timing_type="explicit_date",
        event_date="2026-07-18",
        date_precision="day",
        source_evidence="Symptoms began 18 July 2026.",
    )

    normalized = extractor.normalize_events([event])

    assert normalized[0].event_date == "2026-07-18"
    assert normalized[0].date_precision == "day"

###############################################################################
def test_timeline_preserves_partial_date_ranges_and_rejects_invalid_date_tokens() -> None:
    extractor = PatientTimelineExtractor(client=FakeTimelineClient(PatientTimelineExtraction()))
    events = [
        PatientTimelineEvent(
            event_id="range",
            title="Admission",
            event_date="2025-01",
            event_date_end="2025-02",
            date_precision="month",
            date_certainty="explicit",
            source_evidence="Admission during the documented interval.",
        ),
        PatientTimelineEvent(
            event_id="invalid",
            title="Unsupported date",
            event_date="05.01.2025",
            source_evidence="Timing was recorded in a localized format.",
        ),
    ]

    normalized = extractor.normalize_events(events)

    assert normalized[0].event_date == "2025-01"
    assert normalized[0].event_date_end == "2025-02"
    assert normalized[0].date_precision == "month"
    assert normalized[1].event_date is None

###############################################################################
def test_timeline_date_interval_validates_calendar_days_and_reversed_ranges() -> None:
    assert normalize_timeline_interval("2024-02-29") is not None
    assert normalize_timeline_interval("2025-02-29") is None
    reversed_range = normalize_timeline_interval("2025-03", "2025-02")
    assert reversed_range is not None
    assert reversed_range.end_value is None

###############################################################################
def test_timeline_prompt_uses_canonical_json_and_hash() -> None:
    client = FakeTimelineClient(PatientTimelineExtraction())
    extractor = PatientTimelineExtractor(client=client)

    asyncio.run(extractor.extract_timeline(session_id=5, source_payload={"b": 2, "a": "x"}))

    prompt = client.last_kwargs["user_prompt"]
    assert '{"a":"x","b":2}' in prompt
    assert "Source payload SHA-256:" in prompt
    assert "'a':" not in prompt

###############################################################################
def test_timeline_model_override_validation_requires_an_unambiguous_runtime_model() -> None:
    assert SessionTimelineModelOverrides(
        use_cloud_services=False, text_extraction_model="qwen3:8b"
    ).text_extraction_model == "qwen3:8b"
    assert SessionTimelineModelOverrides(
        use_cloud_services=True, llm_provider="openai", cloud_model="gpt-4.1-mini"
    ).cloud_model == "gpt-4.1-mini"

    import pytest

    with pytest.raises(ValueError):
        SessionTimelineModelOverrides(use_cloud_services=False)
    with pytest.raises(ValueError):
        SessionTimelineModelOverrides(use_cloud_services=True, llm_provider="openai")

###############################################################################
def test_timeline_preview_includes_evidence_and_timing_quality_counts() -> None:
    preview = build_timeline_preview_payload(
        PatientTimeline(
            timeline_id=3,
            session_id=2,
            generated_at=datetime.now(UTC),
            events=[
                    PatientTimelineEvent(event_id="a", title="A", timing_type="explicit_date", event_date="2025-01", source_evidence="source"),
                PatientTimelineEvent(event_id="b", title="B", timing_type="uncertain"),
                PatientTimelineEvent(event_id="c", title="C", timing_type="ordering"),
            ],
        )
    )

    assert preview["source_evidence_event_count"] == 1
    assert preview["missing_evidence_event_count"] == 2
    assert preview["uncertain_event_count"] == 2
    assert preview["undated_event_count"] == 2
