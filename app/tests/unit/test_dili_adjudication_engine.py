from __future__ import annotations

from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugEntry,
    DrugRucamAssessment,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
    RucamComponentAssessment,
)
from services.clinical.dili_evidence import DiliEvidenceBuilder
from services.clinical.dili_pattern import DiliPatternEngine


def test_r_ratio_boundary_values_follow_livertox_definitions() -> None:
    assert DiliPatternEngine.classify(5.0) == "hepatocellular"
    assert DiliPatternEngine.classify(2.0) == "cholestatic"
    assert DiliPatternEngine.classify(3.0) == "mixed"


def test_dili_pattern_requires_explicit_alt_and_alp_uln() -> None:
    patterns = DiliPatternEngine().assess(
        PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=300,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALP",
                    value=120,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
            ]
        )
    )

    assert patterns[0].pattern == "indeterminate"
    assert patterns[0].pattern_source == "unavailable"
    assert patterns[0].r_ratio is None
    assert patterns[0].alt_uln is None
    assert patterns[0].alp_uln is None


def test_dili_pattern_rejects_partial_uln_for_r_ratio() -> None:
    patterns = DiliPatternEngine().assess(
        PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=300,
                    upper_limit_normal=40,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALP",
                    value=120,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
            ]
        )
    )

    assert patterns[0].pattern == "indeterminate"
    assert patterns[0].pattern_source == "unavailable"
    assert patterns[0].r_ratio is None
    assert patterns[0].alt_uln == 40
    assert patterns[0].alp_uln is None


def test_dili_pattern_uses_explicit_uln_for_r_ratio() -> None:
    patterns = DiliPatternEngine().assess(
        PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=300,
                    upper_limit_normal=40,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALP",
                    value=120,
                    upper_limit_normal=120,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
            ]
        )
    )

    assert patterns[0].pattern == "hepatocellular"
    assert patterns[0].pattern_source == "calculated"
    assert patterns[0].r_ratio == 7.5


def test_structured_dossier_preserves_missing_competing_causes() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(
            anamnesis="HAV negative. Autoimmune hepatitis excluded.",
            drugs="Drug A started 2026-01-01 and stopped 2026-01-20.",
            laboratory_analysis="ALT and bilirubin elevated.",
        ),
        drugs=PatientDrugs(
            entries=[
                DrugEntry(
                    name="Drug A",
                    source="therapy",
                    therapy_start_date="2026-01-01",
                    suspension_date="2026-01-20",
                    evidence="Drug A started 2026-01-01 and stopped 2026-01-20.",
                )
            ]
        ),
        labs=PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=240,
                    upper_limit_normal=40,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALP",
                    value=120,
                    upper_limit_normal=120,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="BILIRUBIN",
                    value=3,
                    upper_limit_normal=1,
                    sample_date="2026-01-16",
                    source="laboratory_analysis",
                ),
            ]
        ),
        resolved_drugs={
            "drug a": {
                "decision_status": "accepted_exact_livertox",
                "accepted_livertox_name": "Drug A",
                "match_confidence": 1.0,
                "matched_livertox_row": {"likelihood_score": "A"},
            }
        },
        rucam_bundle=PatientRucamAssessmentBundle(
            entries=[
                DrugRucamAssessment(
                    drug_name="Drug A",
                    total_score=6,
                    causality_category="probable",
                    components=[
                        RucamComponentAssessment(
                            component_key="time_to_onset",
                            label="Time to onset",
                            score=2,
                            evidence="2026-01-01 to 2026-01-15",
                        )
                    ],
                )
            ]
        ),
    )

    assert bundle.patterns[0].pattern == "hepatocellular"
    assert bundle.hys_law.status == "possible"
    assert bundle.hys_law.same_episode is True
    assert bundle.exposures[0].livertox_likelihood == "A"
    assert bundle.exposures[0].rucam.components[0].evidence_quote
    assert "ebv_cmv_hsv" in bundle.differential.unresolved_causes
    assert len(bundle.acceptance_questions) == 12
    assert any(
        question.question.startswith("Does the episode satisfy Hy's Law")
        for question in bundle.acceptance_questions
    )
    assert bundle.manual_review_required is True


def test_report_has_required_fda_style_sections() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(drugs="Drug A"),
        drugs=PatientDrugs(entries=[DrugEntry(name="Drug A")]),
        labs=PatientLabTimeline(entries=[]),
        resolved_drugs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )
    report = DiliEvidenceBuilder.render(bundle)
    for section_number in range(1, 15):
        assert f"## {section_number}." in report
    assert "Manual hepatology review required" in report
