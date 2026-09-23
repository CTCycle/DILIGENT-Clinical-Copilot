from __future__ import annotations

import pytest
from domain.patient_timeline import PatientTimelineGenerationErrorCode
from services.inspection.timeline import _timeline_error_code, _timeline_fallback_note
from services.llm.cloud import LLMError


###############################################################################
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("Provider request timed out"), "timeout"),
        (
            LLMError("HTTP 401: API key rejected", error_code="authentication"),
            "authentication",
        ),
        (
            LLMError("HTTP 429: provider rate limit", error_code="rate_limited"),
            "rate_limited",
        ),
    ],
)
def test_timeline_error_code_classifies_timeout_authentication_and_rate_limit(
    error: BaseException, expected: PatientTimelineGenerationErrorCode
) -> None:
    wrapped = RuntimeError("timeline extraction failed")
    wrapped.__cause__ = error

    assert _timeline_error_code(wrapped) == expected

###############################################################################
def test_timeline_fallback_note_includes_provider_contract_detail() -> None:
    note = _timeline_fallback_note(
        use_cloud_services=True,
        provider="opencode_go",
        model="deepseek-v4-flash",
        error_code="provider_error",
        diagnostic=(
            "Check the provider connection, credentials, rate limits, or transient service status. "
            "Detail: opencode-go rejected deepseek-v4-flash during structured_output "
            "(HTTP 400): Model is unavailable."
        ),
    )

    assert "opencode-go / deepseek-v4-flash" in note
    assert "during structured_output (HTTP 400)" in note
    assert len(note) <= 500

###############################################################################
def test_timeline_fallback_note_truncates_long_diagnostics() -> None:
    note = _timeline_fallback_note(
        use_cloud_services=True,
        provider="opencode_go",
        model="deepseek-v4-flash",
        error_code="provider_error",
        diagnostic="x" * 1000,
    )

    assert len(note) == 500


###############################################################################
@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        ("timeout", "did not respond before the configured timeout"),
        ("authentication", "provider rejected authentication"),
        ("rate_limited", "provider rate-limited the request"),
    ],
)
def test_cloud_timeline_fallback_note_explains_provider_failure(
    error_code: PatientTimelineGenerationErrorCode, expected: str
) -> None:
    note = _timeline_fallback_note(
        use_cloud_services=True,
        provider="opencode_go",
        model="deepseek-v4-flash",
        error_code=error_code,
    )

    assert expected in note
