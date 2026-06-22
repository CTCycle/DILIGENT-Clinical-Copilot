from __future__ import annotations

from domain.clinical import DrugClinicalAssessment, DrugRucamAssessment
from services.clinical.analysis_runner import AnalysisRunner
from services.clinical.hepatox_core import HepatoxConsultation

###############################################################################
def test_source_text_evidence_produces_high_confidence_claim() -> None:
    narrative = AnalysisRunner.build_clinical_narrative(
        drug_name="Amoxicillin",
        excerpts=["Patient started amoxicillin before ALT increase."],
        rucam=None,
        missing_livertox=False,
        evidence_warnings=[],
    )

    assert narrative.claims[0].source == "source_text"
    assert narrative.claims[0].confidence == "high"
    assert narrative.claims[0].requires_review is False

###############################################################################
def test_missing_evidence_claim_requires_review_and_renders_warning() -> None:
    narrative = AnalysisRunner.build_clinical_narrative(
        drug_name="Unknown Herb",
        excerpts=[],
        rucam=DrugRucamAssessment(
            drug_name="Unknown Herb",
            total_score=None,
            causality_category="not assessable",
            confidence="low",
            calculation_method="structured_rucam",
            limitations=["insufficient follow-up labs"],
            data_sufficient=False,
        ),
        missing_livertox=True,
        evidence_warnings=["missing_livertox_match"],
    )
    entry = DrugClinicalAssessment(
        drug_name="Unknown Herb",
        match_status="missing_match",
        evidence_quality="weak",
        claims=narrative.claims,
        narrative=narrative,
    )

    assert narrative.claims[0].source == "unknown"
    assert narrative.claims[0].confidence == "low"
    assert narrative.claims[0].requires_review is True
    assert all(claim.confidence != "high" for claim in narrative.claims)
    rendered = HepatoxConsultation.render_claim_review_lines(entry)
    assert "Claim review" in rendered
    assert "review required" in rendered
    assert "insufficient follow-up labs" in rendered
