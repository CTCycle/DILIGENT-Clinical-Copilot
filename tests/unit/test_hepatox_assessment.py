from __future__ import annotations

import asyncio
from datetime import date

import pytest

from DILIGENT.server.models.prompts import LIVERTOX_CLINICAL_USER_PROMPT
from DILIGENT.server.domain.clinical import (
    ClinicalPipelineValidationError,
    DrugClinicalAssessment,
    DrugEntry,
    DrugRucamAssessment,
    RucamComponentAssessment,
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
            rucam=DrugRucamAssessment(
                drug_name="Acetaminophen",
                injury_type_for_rucam="hepatocellular",
                total_score=7,
                causality_category="probable",
                confidence="moderate",
                components=[
                    RucamComponentAssessment(
                        component_key="time_to_onset",
                        label="Time to onset",
                        score=2,
                        status="scored",
                    )
                ],
                limitations=["Sparse follow-up"],
                summary="Estimated RUCAM summary.",
            ),
        )
    )

    assert result == "Recovered clinical paragraph."
    assert consultation.llm_client.calls == 2


def test_livertox_prompt_removes_per_drug_management_recommendation_directive() -> None:
    assert "Provide monitoring or clinical management recommendations" not in LIVERTOX_CLINICAL_USER_PROMPT
    assert "Do not provide drug-level monitoring or management recommendations" in LIVERTOX_CLINICAL_USER_PROMPT
    assert "Do not add any appendix or extra section after the bibliography line." in LIVERTOX_CLINICAL_USER_PROMPT
    assert "Do not output JSON, YAML, XML, tables, or fenced code blocks" in LIVERTOX_CLINICAL_USER_PROMPT
    assert "# Estimated RUCAM" in LIVERTOX_CLINICAL_USER_PROMPT
    assert "Integrate the supplied estimated RUCAM" in LIVERTOX_CLINICAL_USER_PROMPT


def test_render_matched_drug_section_contains_deterministic_rucam_summary() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)
    entry = DrugClinicalAssessment(
        drug_name="Pantozol",
        match_status="matched",
        matched_livertox_row={"likelihood_score": "C"},
        paragraph="Core clinical narrative.",
        rucam=DrugRucamAssessment(
            drug_name="Pantozol",
            injury_type_for_rucam="cholestatic",
            total_score=6,
            causality_category="probable",
            confidence="moderate",
            components=[
                RucamComponentAssessment(
                    component_key="course",
                    label="Course after withdrawal",
                    score=2,
                    status="scored",
                )
            ],
            limitations=["No confirmed rechallenge"],
            summary="Estimated RUCAM for Pantozol.",
        ),
    )

    rendered = consultation.render_matched_drug_section(entry)

    assert "**Estimated RUCAM**: 6, probable, confidence moderate" in rendered
    assert "**RUCAM component summary**:" in rendered
    assert "**RUCAM limitations**:" in rendered


def test_finalize_patient_report_uses_global_synthesis_section_header() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)

    async def fake_generate_conclusion(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return "Integrated recommendations."

    consultation.generate_conclusion = fake_generate_conclusion  # type: ignore[method-assign]
    report = asyncio.run(
        consultation.finalize_patient_report(
            [
                DrugClinicalAssessment(
                    drug_name="Acetaminophen",
                    match_status="matched",
                    matched_livertox_row={"likelihood_score": "A"},
                    paragraph="Per-drug assessment.",
                )
            ],
            clinical_context="Clinical context",
        )
    )

    assert report is not None
    assert "## Global Synthesis and Clinical Recommendations" in report
    assert "## Conclusion" not in report


def test_finalize_patient_report_renders_deterministic_matched_and_unresolved_sections() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)

    async def fake_generate_conclusion(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return None

    consultation.generate_conclusion = fake_generate_conclusion  # type: ignore[method-assign]
    report = asyncio.run(
        consultation.finalize_patient_report(
            [
                DrugClinicalAssessment(
                    drug_name="Pantozol",
                    match_status="matched",
                    matched_livertox_row={"likelihood_score": "C"},
                    paragraph=(
                        "**Pantozol - LiverTox score C**\n\n"
                        "**Report**\n\n"
                        "**Pantozol - LiverTox score C**\n"
                        "Core clinical narrative.\n"
                        "**Bibliography source**: LiverTox\n"
                        "medication\n"
                        "- **Simvastatin** - 20 mg orally."
                    ),
                ),
                DrugClinicalAssessment(
                    drug_name="Carboplatin",
                    match_status="matched",
                    matched_livertox_row={"likelihood_score": "D"},
                    paragraph="Carboplatin narrative.",
                ),
                DrugClinicalAssessment(
                    drug_name="ulteriore ciclo (originariamente previsto il",
                    match_status="missing",
                    missing_livertox=True,
                    paragraph="Ignored unresolved paragraph.",
                ),
            ],
            clinical_context="Clinical context",
        )
    )

    assert report is not None
    assert report.count("**Pantozol - LiverTox score C**") == 1
    assert report.count("**Carboplatin - LiverTox score D**") == 1
    assert report.count("**Report**") == 2
    assert "Simvastatin" not in report
    assert "## Unresolved Drug Mentions" in report
    assert "ulteriore ciclo (originariamente previsto il" in report
    assert "No matching drug record found in the local knowledge base." in report


def test_finalize_patient_report_keeps_matched_drug_without_excerpt() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)

    async def fake_generate_conclusion(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return None

    consultation.generate_conclusion = fake_generate_conclusion  # type: ignore[method-assign]
    report = asyncio.run(
        consultation.finalize_patient_report(
            [
                DrugClinicalAssessment(
                    drug_name="Valium",
                    match_status="matched",
                    missing_livertox=True,
                    matched_livertox_row={"likelihood_score": "D"},
                    paragraph="Valium: local LiverTox excerpt not available.",
                )
            ],
            clinical_context="Clinical context",
        )
    )

    assert report is not None
    assert "**Valium - LiverTox score D**" in report
    assert "No local LiverTox excerpt is currently available" in report


def test_unresolved_mentions_include_rucam_summary_when_available() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)
    section = consultation.render_unresolved_mentions_section(
        [
            DrugClinicalAssessment(
                drug_name="UnknownX",
                match_status="missing",
                missing_livertox=True,
                rucam=DrugRucamAssessment(
                    drug_name="UnknownX",
                    injury_type_for_rucam="indeterminate",
                    total_score=3,
                    causality_category="possible",
                    confidence="low",
                    components=[],
                    limitations=["Missing serial labs"],
                    summary="Estimated RUCAM only.",
                ),
            )
        ]
    )

    assert section is not None
    assert "UnknownX" in section
    assert "No matching drug record found in the local knowledge base." in section
    assert "RUCAM 3 (possible, confidence low)" in section


def test_sanitize_renderable_body_removes_structured_dili_section() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)
    entry = DrugClinicalAssessment(
        drug_name="Pembrolizumab",
        paragraph=(
            "Clinical narrative before appendix.\n\n"
            "### Structured DILI Assessment Report\n\n"
            "```json\n"
            '{"liver_injury_probable":"High"}\n'
            "```\n"
        ),
    )

    sanitized = consultation.sanitize_renderable_body(entry)

    assert sanitized == "Clinical narrative before appendix."


def test_remove_redundant_report_sentence_truncates_structured_dili_section() -> None:
    raw = (
        "Clinical narrative before appendix.\n\n"
        "### Structured DILI Assessment Report\n\n"
        "```json\n"
        '{"liver_injury_probable":"High"}\n'
        "```\n"
    )

    cleaned = HepatoxConsultation.remove_redundant_report_sentence(raw)

    assert cleaned == "Clinical narrative before appendix."
