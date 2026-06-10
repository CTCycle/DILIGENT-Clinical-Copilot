from __future__ import annotations

from types import SimpleNamespace

from services.clinical.revision.qa import build_revision_qa_validation_payload
from services.clinical.revision.report_builder import RevisionFinalReportPayload


###############################################################################
def test_revision_qa_marks_unaddressed_requested_sections_as_warning() -> None:
    final_report_payload = RevisionFinalReportPayload(
        report_text="Rebuilt report",
        report_present=True,
        report_character_count=len("Rebuilt report"),
        source_excerpt_present=True,
        reviewer_instruction_summary="Update labs and DILI assessment.",
        changed_focus_areas=["labs", "dili_assessment"],
        warnings=[],
    )

    qa_payload = build_revision_qa_validation_payload(
        result_payload={
            "structured_case": {
                "therapy_drugs": [{"name": "drug-a"}],
            },
            "pipeline_artifacts": {
                "faithfulness_audit": {"manual_review_required": False},
            },
            "revision": {
                "livertox_revision_decisions": [],
                "revised_dili_assessments": [],
            },
        },
        instruction_profile=SimpleNamespace(
            target_sections=["labs", "dili_assessment", "final_report"],
            target_entities=["labs", "report_wording", "causality_reasoning"],
        ),
        final_report_payload=final_report_payload,
    )

    assert qa_payload.status == "passed_with_warnings"
    assert qa_payload.version_status == "llm_qa_passed"
    assert "section:final_report" in qa_payload.addressed_items
    assert "section:labs" in qa_payload.unaddressed_items
    assert "entity:report_wording" in qa_payload.addressed_items
    assert "Some reviewer-requested sections or entities could not be verified as addressed." in qa_payload.warnings


###############################################################################
def test_revision_qa_fails_when_blocking_issues_exist() -> None:
    final_report_payload = RevisionFinalReportPayload(
        report_text="Rebuilt report",
        report_present=True,
        report_character_count=len("Rebuilt report"),
        source_excerpt_present=False,
        warnings=[],
    )

    qa_payload = build_revision_qa_validation_payload(
        result_payload={
            "blocking_issues": ["Mismatch between revised labs and final report."],
            "manual_review_required": False,
            "pipeline_artifacts": {},
            "revision": {},
        },
        instruction_profile=None,
        final_report_payload=final_report_payload,
    )

    assert qa_payload.status == "failed"
    assert qa_payload.version_status == "qa_failed"
    assert qa_payload.blocking_issues == [
        "Mismatch between revised labs and final report."
    ]
    assert qa_payload.finding_count == 1
