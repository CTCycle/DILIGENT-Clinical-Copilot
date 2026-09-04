from __future__ import annotations

from domain.clinical.dili import DiliDifferentialAssessment
from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugEntry,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
from services.clinical.dili_case_qualification import DiliCaseQualificationEngine
from services.clinical.dili_causality import DiliCausalityEngine
from services.clinical.dili_evidence import DiliEvidenceBuilder
from services.inspection.revision_clinical_safety import audit_revised_dili_report


###############################################################################
def test_case_qualification_requires_repeat_confirmation_for_enzyme_threshold() -> None:
    labs = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=240,
                upper_limit_normal=40,
                sample_date="2026-01-10",
                source="laboratory_analysis",
            )
        ]
    )
    result = DiliCaseQualificationEngine().assess(labs=labs, drugs=[])
    assert result.status == "insufficient_data"
    assert result.pending_confirmation
    assert not result.qualifying_criteria


###############################################################################
def test_case_qualification_accepts_repeated_aminotransferase_threshold() -> None:
    labs = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=240,
                upper_limit_normal=40,
                sample_date="2026-01-10",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALT",
                value=220,
                upper_limit_normal=40,
                sample_date="2026-01-12",
                source="laboratory_analysis",
            ),
        ]
    )
    result = DiliCaseQualificationEngine().assess(labs=labs, drugs=[])
    assert result.status == "meets_typical_detection_criteria"
    assert any("ALT/AST" in criterion for criterion in result.qualifying_criteria)


###############################################################################
def test_case_qualification_uses_abnormal_pretreatment_value_as_reference() -> None:
    drug = DrugEntry(name="Drug A", therapy_start_date="2026-01-05")
    labs = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=100,
                upper_limit_normal=40,
                sample_date="2026-01-01",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALT",
                value=300,
                upper_limit_normal=40,
                sample_date="2026-01-10",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALT",
                value=280,
                upper_limit_normal=40,
                sample_date="2026-01-12",
                source="laboratory_analysis",
            ),
        ]
    )
    result = DiliCaseQualificationEngine().assess(labs=labs, drugs=[drug])
    assert result.baseline_abnormal is True
    assert result.baseline_date == "2026-01-01"
    assert result.status == "below_typical_detection_criteria"


###############################################################################
def test_dechallenge_is_calculated_for_each_drug_stop_date() -> None:
    engine = DiliCausalityEngine()
    labs = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=500,
                upper_limit_normal=40,
                sample_date="2026-01-10",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALT",
                value=450,
                upper_limit_normal=40,
                sample_date="2026-01-15",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALT",
                value=180,
                upper_limit_normal=40,
                sample_date="2026-01-25",
                source="laboratory_analysis",
            ),
        ]
    )
    early_stop = DrugEntry(name="Drug A", suspension_date="2026-01-09")
    late_stop = DrugEntry(name="Drug B", suspension_date="2026-01-15")
    early = engine._dechallenge_for_drug(
        drug=early_stop,
        labs=labs,
        primary_pattern="hepatocellular",
    )
    late = engine._dechallenge_for_drug(
        drug=late_stop,
        labs=labs,
        primary_pattern="hepatocellular",
    )
    assert early == "improving_after_stop"
    assert late == "improving_after_stop"


###############################################################################
def test_weak_livertox_prior_does_not_cap_patient_specific_support() -> None:
    engine = DiliCausalityEngine()
    drug = DrugEntry(
        name="Drug A",
        source="therapy",
        therapy_start_date="2026-01-01",
        suspension_date="2026-01-10",
        evidence="Drug A started 2026-01-01 and stopped 2026-01-10.",
    )
    labs = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=400,
                upper_limit_normal=40,
                sample_date="2026-01-10",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALT",
                value=150,
                upper_limit_normal=40,
                sample_date="2026-01-20",
                source="laboratory_analysis",
            ),
        ]
    )
    exposure = engine.exposure(
        drug,
        {
            "decision_status": "accepted_exact_livertox",
            "accepted_livertox_name": "Drug A",
            "matched_livertox_row": {"likelihood_score": "E"},
        },
        None,
        DiliDifferentialAssessment(all_major_causes_excluded=True),
        labs,
        "hepatocellular",
        "2026-01-08",
    )
    assert exposure.causality is not None
    assert exposure.causality.category == "supportive"
    assert exposure.causality.known_hepatotoxic_potential == "E"
    assert exposure.causality.drug_signature_concordance == "reference_evidence_sparse"


###############################################################################
def test_injury_before_exposure_argues_against_causality() -> None:
    engine = DiliCausalityEngine()
    exposure = engine.exposure(
        DrugEntry(name="Drug A", therapy_start_date="2026-01-10"),
        {
            "decision_status": "accepted_exact_livertox",
            "accepted_livertox_name": "Drug A",
        },
        None,
        DiliDifferentialAssessment(all_major_causes_excluded=True),
        PatientLabTimeline(entries=[]),
        "hepatocellular",
        "2026-01-05",
    )
    assert exposure.causality is not None
    assert exposure.causality.temporal_compatibility == "incompatible_pre_exposure"
    assert exposure.causality.category == "argues_against"


###############################################################################
def test_long_latency_requires_review_instead_of_automatic_exclusion() -> None:
    engine = DiliCausalityEngine()
    exposure = engine.exposure(
        DrugEntry(name="Drug A", therapy_start_date="2024-01-01"),
        {
            "decision_status": "accepted_exact_livertox",
            "accepted_livertox_name": "Drug A",
        },
        None,
        DiliDifferentialAssessment(all_major_causes_excluded=False),
        PatientLabTimeline(entries=[]),
        "hepatocellular",
        "2026-01-05",
    )
    assert exposure.causality is not None
    assert (
        exposure.causality.temporal_compatibility
        == "long_latency_requires_drug_specific_review"
    )
    assert exposure.causality.category == "limited"


###############################################################################
def test_revision_safety_reuses_structured_dili_evidence_gate() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(
            anamnesis="Jaundice and fatigue. Alternative-cause workup is pending.",
            drugs="Drug A started 2026-01-01.",
            laboratory_analysis="ALT elevated.",
        ),
        drugs=PatientDrugs(
            entries=[
                DrugEntry(
                    name="Drug A",
                    source="therapy",
                    therapy_start_date="2026-01-01",
                    evidence="Drug A started 2026-01-01.",
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
                )
            ]
        ),
        resolved_drugs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )
    issues = audit_revised_dili_report(
        session={
            "result_payload": {
                "pipeline_artifacts": {
                    "dili_evidence_bundle": bundle.model_dump(mode="json")
                }
            }
        },
        report_text="All competing causes were excluded. This is a definitive diagnosis.",
    )
    assert any("competing causes" in issue.lower() for issue in issues)
    assert any("definitive" in issue.lower() for issue in issues)
