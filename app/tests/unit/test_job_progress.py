from __future__ import annotations

from services.clinical.job_progress import CLINICAL_PROGRESS_MESSAGES

###############################################################################
def test_progress_messages_cover_ordered_events() -> None:
    ordered = [
        "preflight.validated",
        "sections.loaded",
        "assessment.bundle",
        "drugs.extracting",
        "drugs.resolving",
        "diseases.extracting",
        "labs.extracting",
        "pattern.assessing",
        "candidates.selecting",
        "rucam.initial",
        "retrieval.query",
        "retrieval.evidence",
        "rucam.refined",
        "report.generating",
        "session.saving",
    ]
    for event in ordered:
        assert event in CLINICAL_PROGRESS_MESSAGES

    assert "vector retrieval disabled" in CLINICAL_PROGRESS_MESSAGES[
        "livertox_lookup.no_rag"
    ]
    assert "vector evidence" in CLINICAL_PROGRESS_MESSAGES["livertox_lookup.rag"]
