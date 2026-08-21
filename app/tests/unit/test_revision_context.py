from __future__ import annotations

from services.inspection.revision_context import build_revision_context

###############################################################################
def test_revision_context_deduplicates_reports_and_retains_user_instruction() -> None:
    context = build_revision_context(
        session={
            "session_id": 4,
            "source_clinical_text": "Source evidence " * 100,
            "sections": {
                "anamnesis": "Anamnesis " * 500,
                "therapy": "Therapy " * 500,
            },
            "official_report_text": "Canonical report",
            "result_payload": {
                "report": "Canonical report",
                "matched_drugs": [{"name": "amoxicillin"}],
            },
        },
        manual_edits=[],
        lineage=[],
        selected_text="Selected sentence",
        instruction="Correct unsupported causality.",
        input_budget=500,
    )

    assert context["review_target"]["official_report"]["text"] == "Canonical report"
    assert context["review_target"]["final_report"] == {"omitted": True}
    assert context["user_steering"]["instruction"]["text"] == (
        "Correct unsupported causality."
    )
    report = context["context_budget"]["selection_report"]
    assert report["deduplicated_count"] >= 1
    assert report["capacity_known"] is True

###############################################################################
def test_revision_context_uses_conservative_budget_when_capacity_is_unknown() -> None:
    context = build_revision_context(
        session={
            "session_id": 5,
            "source_clinical_text": "Short source",
            "official_report_text": "Short report",
        },
        manual_edits=[],
        lineage=[],
        selected_text=None,
        instruction="Review it.",
    )

    budget = context["context_budget"]
    assert budget["selection_report"]["capacity_known"] is False
    assert budget["selection_report"]["unknown_capacity_fallback"] is True
    assert budget["selection_report"]["planning_input_budget"] == 8192
