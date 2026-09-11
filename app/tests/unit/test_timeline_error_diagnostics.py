from __future__ import annotations

from services.inspection.timeline import _timeline_fallback_note

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
