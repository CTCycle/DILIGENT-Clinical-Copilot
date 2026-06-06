from __future__ import annotations

from types import SimpleNamespace

from services.clinical.revision.report_builder import (
    build_revision_final_report_payload,
)


def test_revision_report_builder_marks_manual_review_warning_from_report_comparison() -> None:
    payload = build_revision_final_report_payload(
        result_payload={
            "report": "Rebuilt revised report body",
            "report_comparison": {
                "outcome": "aligned_with_changes",
                "manual_review": "required",
            },
        },
        selected_text="Focus on chronology section.",
        instruction_profile=SimpleNamespace(
            instruction_summary="Clarify chronology and update final report wording.",
            target_sections=["final_report", "therapy"],
        ),
    )

    assert payload.report_present is True
    assert payload.report_character_count == len("Rebuilt revised report body")
    assert payload.source_excerpt_present is True
    assert payload.comparison_outcome == "aligned_with_changes"
    assert payload.changed_focus_areas == ["final_report", "therapy"]
    assert "Report comparison still requests manual review." in payload.warnings


def test_revision_report_builder_warns_when_final_report_is_missing() -> None:
    payload = build_revision_final_report_payload(
        result_payload={},
        selected_text=None,
        instruction_profile=None,
    )

    assert payload.report_present is False
    assert payload.report_text == ""
    assert payload.changed_focus_areas == ["unknown"]
    assert "Revision output did not produce a final report body." in payload.warnings
