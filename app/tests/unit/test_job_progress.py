from __future__ import annotations

from services.clinical.job_progress import (
    CLINICAL_PROGRESS_MESSAGES,
    ClinicalJobProgressCallback,
    build_clinical_progress_message,
)


def test_progress_message_for_anamnesis_detail() -> None:
    message = build_clinical_progress_message("x", 30.0, "anamnesis.extracting")
    assert message == CLINICAL_PROGRESS_MESSAGES["anamnesis.extracting"]


def test_progress_messages_cover_ordered_events() -> None:
    ordered = [
        "preflight.validated",
        "sections.loaded",
        "assessment.bundle",
        "therapy.extracting",
        "anamnesis.extracting",
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


def test_old_two_argument_progress_callback_usage_still_works(monkeypatch) -> None:
    calls = []

    def fake_report(job_id: str, *, stage: str, progress: float, detail=None) -> None:
        calls.append((job_id, stage, progress, detail))

    monkeypatch.setattr(
        "services.clinical.job_progress.report_clinical_job_progress", fake_report
    )
    callback = ClinicalJobProgressCallback(job_id="job-1")
    callback("clinical", 12.0)
    assert calls == [("job-1", "clinical", 12.0, None)]
