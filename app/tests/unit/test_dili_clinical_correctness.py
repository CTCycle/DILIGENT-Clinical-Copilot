# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from domain.clinical.dili import DiliDifferentialAssessment
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
from services.clinical.dili_case_qualification import DiliCaseQualificationEngine
from services.clinical.dili_causality import DiliCausalityEngine
from services.clinical.dili_differential import DiliDifferentialEngine
from services.clinical.dili_evidence import DiliEvidenceBuilder
from services.clinical.dili_timeline import DiliTimelineEngine
from services.inspection.revision_clinical_safety import audit_revised_dili_report

###############################################################################
def test_non_source_rucam_score_is_not_promoted_into_dili_bundle() -> None:
    assessment = DiliCausalityEngine.rucam(
        DrugRucamAssessment(
            drug_name="Drug A",
            total_score=6,
            causality_category="probable",
            components=[
                RucamComponentAssessment(
                    component_key="time_to_onset",
                    label="Time to onset",
                    score=2,
                    status="scored",
                )
            ],
        ),
        "Drug A",
    )

    assert assessment is not None
    assert assessment.total_score is None
    assert assessment.category == "not assessable"
    assert assessment.components[0].score is None
    assert assessment.components[0].status == "not_assessable"
    assert any("provenance" in item for item in assessment.limitations)

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

###############################################################################
def _alt(value: float, sample_date: str) -> ClinicalLabEntry:
    return ClinicalLabEntry(
        marker_name="ALT",
        value=value,
        upper_limit_normal=40,
        sample_date=sample_date,
        source="laboratory_analysis",
    )

###############################################################################
def _cause_status(text: str, cause: str) -> str:
    assessment = DiliDifferentialEngine().assess(text)
    return {item.cause: item.status for item in assessment.causes}[cause]

###############################################################################
def test_documented_competing_cause_is_not_excluded_by_nearby_negative_test() -> None:
    assert (
        _cause_status(
            "HBsAg negative, anti-HCV positive with detectable HCV RNA.",
            "viral_hepatitis_a_b_c_d_e",
        )
        == "not_excluded"
    )
    assert (
        _cause_status(
            "Hepatitis E IgM positive and HEV RNA detected. Hepatitis A IgM negative.",
            "viral_hepatitis_a_b_c_d_e",
        )
        == "not_excluded"
    )
    assert (
        _cause_status(
            "Common bile duct dilated with choledocholithiasis. Serologies negative.",
            "biliary_obstruction_gallstones",
        )
        == "not_excluded"
    )
    assert (
        _cause_status("Alcohol: 60 g/day, ongoing.", "alcoholic_hepatitis")
        == "not_excluded"
    )

###############################################################################
def test_negated_cause_is_excluded_and_substrings_do_not_match() -> None:
    assert (
        _cause_status("No alcohol use. No known liver disease.", "alcoholic_hepatitis")
        == "excluded"
    )
    assert (
        _cause_status(
            "Anamnesis reviewed. Plasma glucose normal.", "autoimmune_hepatitis"
        )
        == "missing_data"
    )
    assert (
        _cause_status("HEV infection cannot be excluded.", "viral_hepatitis_a_b_c_d_e")
        == "unknown"
    )

###############################################################################
def test_persistent_injury_after_withdrawal_is_not_reported_as_resolved() -> None:
    timeline = DiliTimelineEngine().build(
        [DrugEntry(name="Drug A", therapy_start_date="2024-01-10", suspension_date="2024-03-01")],
        PatientLabTimeline(
            entries=[
                _alt(30, "2024-01-05"),
                _alt(800, "2024-02-28"),
                _alt(780, "2024-03-02"),
                _alt(600, "2024-03-31"),
                _alt(420, "2024-05-01"),
            ]
        ),
        primary_pattern="hepatocellular",
    )
    assert timeline.dechallenge_status == "chronic_or_persistent"
    assert timeline.recovery_date is None

###############################################################################
def test_dechallenge_uses_rucam_window_and_resolution_to_reference() -> None:
    drug = DrugEntry(name="Drug A", therapy_start_date="2024-01-10", suspension_date="2024-03-01")
    improving = DiliTimelineEngine().build(
        [drug],
        PatientLabTimeline(entries=[_alt(800, "2024-03-01"), _alt(320, "2024-03-08")]),
        primary_pattern="hepatocellular",
    )
    resolved = DiliTimelineEngine().build(
        [drug],
        PatientLabTimeline(
            entries=[_alt(800, "2024-03-01"), _alt(400, "2024-03-08"), _alt(36, "2024-04-10")]
        ),
        primary_pattern="hepatocellular",
    )
    assert improving.dechallenge_status == "improving_after_stop"
    assert improving.recovery_date is None
    assert resolved.dechallenge_status == "resolved_to_baseline"
    assert resolved.recovery_date == "2024-04-10"
