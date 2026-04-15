from __future__ import annotations

import asyncio
from datetime import datetime, UTC
from typing import Any

from DILIGENT.server.domain.patient_timeline import (
    PatientTimelineEvent,
    PatientTimelineExtraction,
)
from DILIGENT.server.services.clinical.timeline import PatientTimelineExtractor


class FakeTimelineClient:
    def __init__(self, payload: PatientTimelineExtraction) -> None:
        self.payload = payload
        self.call_count = 0

    async def llm_structured_call(self, **kwargs: Any) -> PatientTimelineExtraction:
        _ = kwargs
        self.call_count += 1
        return self.payload


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
                        confidence=0.6,
                    ),
                    PatientTimelineEvent(
                        event_id="a",
                        title="Therapy started",
                        event_type="therapy",
                        event_date="2025-01-10",
                        confidence=0.8,
                    ),
                    PatientTimelineEvent(
                        event_id="c",
                        title="ALT peak",
                        event_type="lab",
                        event_date="2025-03-01",
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
