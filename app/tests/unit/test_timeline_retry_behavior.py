from __future__ import annotations

import asyncio
from typing import Any

import pytest

from domain.patient_timeline import PatientTimelineEvent, PatientTimelineExtraction
from services.clinical.timeline import PatientTimelineExtractor
from services.llm.cloud import LLMError


###############################################################################
class RetryThenSuccessClient:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.call_count = 0

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientTimelineExtraction:
        del kwargs
        self.call_count += 1
        if self.call_count == 1:
            raise LLMError(
                "Cloud provider connection failed",
                error_code="network_unavailable",
                retryable=True,
            )
        return PatientTimelineExtraction(
            events=[
                PatientTimelineEvent(
                    event_id="event-1",
                    title="Symptoms started",
                    event_date="2026-07-20",
                    source_evidence="Symptoms started on 2026-07-20.",
                )
            ]
        )


###############################################################################
class NonRetryableClient:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.call_count = 0

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientTimelineExtraction:
        del kwargs
        self.call_count += 1
        raise LLMError("invalid structured response", error_code="invalid_response")


###############################################################################
def test_timeline_extractor_retries_transient_provider_failures() -> None:
    client = RetryThenSuccessClient()
    extractor = PatientTimelineExtractor(client=client)

    result = asyncio.run(
        extractor.extract_timeline(
            session_id=7,
            source_payload={"anamnesis": "Symptoms started on 2026-07-20."},
        )
    )

    assert client.call_count == 2
    assert result.events[0].title == "Symptoms started"


###############################################################################
def test_timeline_extractor_does_not_retry_non_transient_provider_failures() -> None:
    client = NonRetryableClient()
    extractor = PatientTimelineExtractor(client=client)

    with pytest.raises(LLMError) as error:
        asyncio.run(
            extractor.extract_timeline(
                session_id=8,
                source_payload={"anamnesis": "Malformed provider response."},
            )
        )

    assert error.value.error_code == "invalid_response"
    assert client.call_count == 1
