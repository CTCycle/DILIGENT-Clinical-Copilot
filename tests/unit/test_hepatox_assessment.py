from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from DILIGENT.server.models.prompts import LIVERTOX_CLINICAL_USER_PROMPT
from DILIGENT.server.domain.clinical import (
    ClinicalPipelineValidationError,
    DrugEntry,
    DrugSuspensionContext,
    PatientData,
)
from DILIGENT.server.services.clinical.hepatox import (
    HepatotoxicityPatternAnalyzer,
    HepatoxConsultation,
)


###############################################################################
class FlakyChatClient:
    def __init__(self, *, fail_count: int, response: object) -> None:
        self.fail_count = max(int(fail_count), 0)
        self.response = response
        self.calls = 0

    # -------------------------------------------------------------------------
    async def chat(self, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        self.calls += 1
        if self.calls <= self.fail_count:
            raise RuntimeError("temporary provider failure")
        return self.response


def test_assess_payload_returns_determined_score_when_labs_present() -> None:
    analyzer = HepatotoxicityPatternAnalyzer()
    payload = PatientData(
        drugs="Acetaminophen 500 mg 1-0-0-0",
        alt="100",
        alt_max="50",
        alp="200",
        alp_max="100",
    )

    assessment = analyzer.assess_payload(payload, allow_missing_labs=False)

    assert assessment.status == "ok"
    assert assessment.issues == []
    assert assessment.score.classification == "cholestatic"
    assert assessment.score.r_score == pytest.approx(1.0)


def test_assess_payload_raises_when_labs_missing_and_not_overridden() -> None:
    analyzer = HepatotoxicityPatternAnalyzer()
    payload = PatientData(
        drugs="Acetaminophen 500 mg 1-0-0-0",
        alt=None,
        alt_max="50",
        alp="200",
        alp_max="100",
    )

    with pytest.raises(ClinicalPipelineValidationError) as exc_info:
        analyzer.assess_payload(payload, allow_missing_labs=False)

    assert any(issue.code == "missing_labs" for issue in exc_info.value.issues)


def test_assess_payload_allows_missing_labs_when_overridden() -> None:
    analyzer = HepatotoxicityPatternAnalyzer()
    payload = PatientData(
        drugs="Acetaminophen 500 mg 1-0-0-0",
        alt=None,
        alt_max="50",
        alp="200",
        alp_max="100",
    )

    assessment = analyzer.assess_payload(payload, allow_missing_labs=True)

    assert assessment.status == "undetermined_due_to_missing_labs"
    assert assessment.score.classification == "indeterminate"
    assert any(issue.code == "missing_labs" for issue in assessment.issues)


def test_evaluate_suspension_marks_anamnesis_mentions_as_uncertain_exposure() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)
    entry = DrugEntry(
        name="Paracetamol",
        source="anamnesis",
        historical_flag=True,
    )

    suspension = consultation.evaluate_suspension(entry, visit_date=date(2025, 4, 14))

    assert suspension.suspended is False
    assert suspension.note is not None
    assert "Historical mention from anamnesis" in suspension.note
    assert "Active therapy; no suspension reported." not in suspension.note


def test_format_visit_date_anchor_handles_missing_and_present_values() -> None:
    assert HepatoxConsultation.format_visit_date_anchor(None) == "Not provided."
    assert (
        HepatoxConsultation.format_visit_date_anchor(date(2025, 4, 14))
        == "2025-04-14"
    )


def test_request_drug_analysis_retries_on_transient_failure() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)
    consultation.llm_client = FlakyChatClient(
        fail_count=1,
        response={"content": "Recovered clinical paragraph."},
    )
    consultation.llm_model = "test-model"
    consultation.chat_supports_temperature = True
    consultation.temperature = 0.0
    consultation.analysis_retry_attempts = 2

    result = asyncio.run(
        consultation.request_drug_analysis(
            drug_name="Acetaminophen",
            canonical_name="acetaminophen",
            origins=["therapy"],
            extraction_metadata=[],
            livertox_status="matched",
            excerpt="Acetaminophen can cause dose-related liver injury.",
            rag_documents=None,
            clinical_context="No additional context.",
            suspension=DrugSuspensionContext(suspended=False),
            visit_date=date(2025, 4, 14),
            pattern_summary="Observed liver injury pattern: hepatocellular.",
            metadata={"likelihood_score": "A"},
            web_evidence="No web evidence available (reason: web search disabled for this session).",
        )
    )

    assert result == "Recovered clinical paragraph."
    assert consultation.llm_client.calls == 2


def test_livertox_prompt_removes_per_drug_management_recommendation_directive() -> None:
    assert "Provide monitoring or clinical management recommendations" not in LIVERTOX_CLINICAL_USER_PROMPT
    assert "Do not provide drug-level monitoring or management recommendations" in LIVERTOX_CLINICAL_USER_PROMPT


def test_finalize_patient_report_uses_global_synthesis_section_header() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)

    async def fake_generate_conclusion(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return "Integrated recommendations."

    consultation.generate_conclusion = fake_generate_conclusion  # type: ignore[method-assign]
    report = asyncio.run(
        consultation.finalize_patient_report(
            [SimpleNamespace(paragraph="Per-drug assessment.")],
            clinical_context="Clinical context",
        )
    )

    assert report is not None
    assert "## Global Synthesis and Clinical Recommendations" in report
    assert "## Conclusion" not in report
