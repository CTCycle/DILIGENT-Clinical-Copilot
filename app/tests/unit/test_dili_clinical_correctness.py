# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import date
from functools import partial

import pytest

from domain.clinical.dili import DiliDifferentialAssessment, DiliEvidenceBundle
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
from services.clinical.candidate_selection import select_relevant_candidates
from services.clinical.dili_case_qualification import DiliCaseQualificationEngine
from services.clinical.dili_causality import DiliCausalityEngine
from services.clinical.dili_differential import DiliDifferentialEngine
from services.clinical.dili_evidence import DiliEvidenceBuilder
from services.clinical.dili_hys_law import HysLawDetector
from services.clinical.dili_severity import DiliSeverityGrader
from services.clinical.dili_timeline import DiliTimelineEngine
from services.clinical.drug_deduplication import find_drug_date_conflicts
from services.clinical.drug_resolution.serialization import index_resolved_drugs
from services.clinical.labs import ClinicalLabExtractor
from services.clinical.livertox_likelihood import normalize_livertox_likelihood
from services.clinical.parser import DrugsParser
from services.clinical.rucam import RucamScoreEstimator
from services.inspection.revision_clinical_safety import audit_revised_dili_report
from services.text.normalization import normalize_drug_query_name

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
def _qualification_lab(
    marker: str, value: float, uln: float | None, unit: str | None = None
) -> ClinicalLabEntry:
    return ClinicalLabEntry(
        marker_name=marker,
        value=value,
        upper_limit_normal=uln,
        unit=unit,
        sample_date="2026-01-10",
        source="laboratory_analysis",
    )

###############################################################################
@pytest.mark.parametrize(
    ("entries", "status", "criterion"),
    [
        # Single panel ALT 6x ULN qualifies; repeat testing is only recommended.
        ([_qualification_lab("ALT", 240, 40)], "meets_typical_detection_criteria", "ALT/AST >=5x"),
        # ALT 4x + TBIL 3x ULN in µmol/L.
        (
            [
                _qualification_lab("ALT", 160, 40),
                _qualification_lab("TBIL", 61.6, 20.5, "umol/L"),
            ],
            "meets_typical_detection_criteria",
            "total bilirubin >2x",
        ),
        # µmol/L value against a mg/dL ULN is converted (61.6 / 17.1 = 3.6 mg/dL).
        (
            [
                _qualification_lab("ALT", 160, 40),
                _qualification_lab("TBIL", 61.6, 1.2, "umol/L"),
            ],
            "meets_typical_detection_criteria",
            "total bilirubin >2x",
        ),
        # ALT 4x with TBIL 1.5x ULN meets no EASL criterion.
        (
            [
                _qualification_lab("ALT", 160, 40),
                _qualification_lab("TBIL", 1.8, 1.2, "mg/dL"),
            ],
            "below_typical_detection_criteria",
            None,
        ),
        # INR >1.5 alone is not an EASL case-definition criterion.
        (
            [_qualification_lab("ALT", 70, 40), _qualification_lab("INR", 1.7, 1.1)],
            "below_typical_detection_criteria",
            None,
        ),
        # Live C14: ALT 410 U/L without a ULN cannot be called below criteria.
        ([_qualification_lab("ALT", 410, None, "U/L")], "insufficient_data", None),
    ],
)
def test_case_qualification_follows_easl_biochemical_criteria(
    entries: list[ClinicalLabEntry], status: str, criterion: str | None
) -> None:
    result = DiliCaseQualificationEngine().assess(
        labs=PatientLabTimeline(entries=entries), drugs=[]
    )

    assert result.status == status
    if criterion is None:
        assert not result.qualifying_criteria
    else:
        assert any(criterion in item for item in result.qualifying_criteria)
    if status == "meets_typical_detection_criteria" and len(entries) == 1:
        assert any("repeat testing" in note for note in result.pending_confirmation)
    if any(entry.marker_name == "INR" for entry in entries):
        assert any("not a qualifying criterion" in item for item in result.rationale)

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

###############################################################################
def _lab(
    marker: str, value: float, uln: float, sample_date: str, unit: str | None = None
) -> ClinicalLabEntry:
    return ClinicalLabEntry(
        marker_name=marker,
        value=value,
        unit=unit,
        upper_limit_normal=uln,
        sample_date=sample_date,
        source="laboratory_analysis",
    )

###############################################################################
def _hys_law_status(
    labs: list[ClinicalLabEntry], causes_excluded: bool
) -> tuple[str, bool | None]:
    drugs = [DrugEntry(name="Drug A", therapy_start_date="2024-01-05", suspension_date="2024-03-03")]
    timeline = DiliTimelineEngine().build(drugs, PatientLabTimeline(entries=labs))
    result = HysLawDetector().assess(
        labs=PatientLabTimeline(entries=labs),
        differential=DiliDifferentialAssessment(all_major_causes_excluded=causes_excluded),
        timeline=timeline,
        drugs=drugs,
        source_text="",
    )
    return result.status, result.initial_cholestasis_present

###############################################################################
def test_hys_law_uses_alp_at_aminotransferase_rise() -> None:
    # FDA 2009: ALT >=3x ULN with TBIL >2x ULN, without initial ALP >=2x ULN.
    injury = [_lab("ALT", 600, 40, "2024-03-01"), _lab("TBIL", 4.0, 1.2, "2024-03-02", "mg/dL")]
    cholestatic_onset = [
        _lab("ALP", 100, 120, "2024-01-02"),
        _lab("ALP", 300, 120, "2024-03-01"),
        *injury,
    ]
    hepatocellular_onset = [
        _lab("ALP", 300, 120, "2023-12-01"),
        _lab("ALP", 140, 120, "2024-03-01"),
        *injury,
    ]

    assert _hys_law_status(cholestatic_onset, causes_excluded=True) == ("not_met", True)
    assert _hys_law_status(hepatocellular_onset, causes_excluded=True) == (
        "meets_criteria",
        False,
    )
    assert _hys_law_status(hepatocellular_onset, causes_excluded=False) == (
        "possible",
        False,
    )

###############################################################################
@pytest.mark.parametrize(
    ("text", "labs", "grade"),
    [
        ("Outpatient. No hospitalization required. No jaundice.", [_lab("TBIL", 3.0, 1.2, "2024-03-01", "mg/dL")], "2_moderate"),
        ("Hospitalized for subarachnoid haemorrhage.", [_lab("TBIL", 3.0, 1.2, "2024-03-01", "mg/dL")], "2_moderate"),
        ("History of kidney transplant in 2015. Outpatient review.", [_lab("TBIL", 1.0, 1.2, "2024-03-01", "mg/dL")], "1_mild"),
        ("Father died of myocardial infarction.", [_lab("TBIL", 1.0, 1.2, "2024-03-01", "mg/dL")], "1_mild"),
        ("Outpatient review, no encephalopathy.", [_lab("TBIL", 6.0, 1.2, "2024-03-01", "mg/dL")], "2_moderate"),
        ("Outpatient review.", [_lab("TBIL", 2.6, 1.2, "2024-03-01", "mg/dL")], "2_moderate"),
        ("Outpatient review.", [_lab("TBIL", 50.0, 21.0, "2024-03-01", "umol/L")], "2_moderate"),
        ("Outpatient review.", [_lab("TBIL", 40.0, 21.0, "2024-03-01", "umol/L")], "1_mild"),
        ("Outpatient review.", [_lab("INR", 1.6, 1.1, "2024-03-01")], "2_moderate"),
        ("Admitted for jaundice; no encephalopathy.", [_lab("TBIL", 6.0, 1.2, "2024-03-01", "mg/dL")], "3_moderate_severe"),
        ("Admitted for jaundice.", [_lab("TBIL", 6.0, 1.2, "2024-03-01", "mg/dL"), _lab("INR", 1.6, 1.1, "2024-03-01")], "4_severe"),
        ("Hepatic encephalopathy developed on day 5.", [_lab("TBIL", 6.0, 1.2, "2024-03-01", "mg/dL")], "4_severe"),
        ("She underwent liver transplantation for acute liver failure.", [_lab("TBIL", 20.0, 1.2, "2024-03-01", "mg/dL")], "5_fatal_or_transplant"),
    ],
)
def test_severity_follows_dilin_scale_with_negation_aware_events(
    text: str, labs: list[ClinicalLabEntry], grade: str
) -> None:
    # DILIN (Fontana 2009): bilirubin in mg/dL (umol/L / 17.1); events must
    # concern the patient's DILI episode.
    labs = [_lab("ALT", 400, 40, "2024-03-01"), *labs]
    assert DiliSeverityGrader().assess(PatientLabTimeline(entries=labs), text).grade == grade

###############################################################################
def _resolve_rucam(text: str) -> tuple[int, str | None, str | None]:
    result = RucamScoreEstimator().resolve_provided_rucam_score(text)
    assert result is not None
    return result.score, result.causality_category, result.stated_category

###############################################################################
def test_patient_record_rucam_score_is_parsed_without_years_or_category_override() -> None:
    estimator = RucamScoreEstimator()
    assert _resolve_rucam("The updated RUCAM 2016 scale was applied: total score 5 (possible).") == (5, "possible", "possible")
    assert _resolve_rucam("RUCAM score of 7.") == (7, "probable", None)
    assert _resolve_rucam("RUCAM: -12.") == (-9, "excluded", None)
    assert _resolve_rucam("RUCAM 4; team considered the drug a probable culprit.") == (4, "possible", "probable")

    source = estimator.resolve_provided_rucam_score("RUCAM 4; team considered the drug a probable culprit.")
    assert source is not None
    assessment = estimator.build_source_reported_assessment(
        drug=DrugEntry(name="Drug A"),
        injury_type="hepatocellular",
        source=source,
        report_language="en",
    )
    assert assessment.causality_category == "possible"
    assert any("'probable'" in item for item in assessment.limitations)

###############################################################################
def _first_abnormal(labs: list[ClinicalLabEntry]) -> str | None:
    return DiliTimelineEngine().build(
        [DrugEntry(name="Drug A", therapy_start_date="2024-01-15")],
        PatientLabTimeline(entries=labs),
    ).first_abnormal_liver_test_date

###############################################################################
def test_first_abnormal_liver_test_ignores_baseline_and_non_liver_markers() -> None:
    # EASL 2019: an abnormal pre-treatment baseline replaces the ULN.
    assert _first_abnormal([_alt(72, "2023-12-01"), _alt(70, "2024-02-01"), _alt(290, "2024-03-01")]) == "2024-03-01"
    assert _first_abnormal([_lab("INR", 1.3, 1.1, "2024-02-01"), _alt(30, "2024-02-01"), _alt(400, "2024-03-01")]) == "2024-03-01"
    assert _first_abnormal([_alt(30, "2024-02-01"), _alt(35, "2024-03-01")]) is None

###############################################################################
def test_abnormal_baseline_does_not_make_later_exposure_argue_against() -> None:
    # Audit case C09: MASLD with chronically raised ALT before duloxetine.
    labs = [
        _lab("ALT", 72, 40, "2023-12-01"),
        _lab("ALP", 100, 120, "2023-12-01"),
        _lab("ALT", 290, 40, "2024-03-01"),
        _lab("ALP", 115, 120, "2024-03-01"),
        _lab("ALT", 160, 40, "2024-04-01"),
    ]
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(anamnesis="MASLD with chronically elevated ALT."),
        drugs=PatientDrugs(
            entries=[DrugEntry(name="Duloxetine", therapy_start_date="2024-01-15", suspension_date="2024-03-16")]
        ),
        labs=PatientLabTimeline(entries=labs),
        resolved_drugs={
            "Duloxetine": {
                "decision_status": "accepted_exact_livertox",
                "accepted_livertox_name": "Duloxetine",
            }
        },
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )
    causality = bundle.exposures[0].causality
    assert bundle.timeline.first_abnormal_liver_test_date == "2024-03-01"
    assert causality is not None
    assert causality.temporal_compatibility != "incompatible_pre_exposure"
    assert causality.category != "argues_against"

###############################################################################
def test_timeline_ignores_negated_jaundice_and_ranks_bilirubin_by_multiple() -> None:
    timeline = DiliTimelineEngine().build(
        [DrugEntry(name="Drug A", therapy_start_date="2024-01-05")],
        PatientLabTimeline(
            entries=[
                _lab("BILIRUBIN", 1.0, 1.2, "2024-03-01", "mg/dL"),
                _lab("TBIL", 5.0, 1.2, "2024-03-08", "mg/dL"),
                _lab("TBIL", 30.0, 21.0, "2024-03-12", "umol/L"),
            ]
        ),
        source_text="Patient denies jaundice; bilirubin normal.",
    )
    assert not [
        event
        for event in timeline.events
        if event.event_type == "jaundice_or_bilirubin_rise" and event.marker is None
    ]
    assert timeline.peak_dates["BILIRUBIN"] == "2024-03-08"

###############################################################################
def _parse_lab_text(text: str) -> list[tuple[str, float | None, str | None, float | None]]:
    return [
        (item.marker_name, item.value, item.unit, item.upper_limit_normal)
        for item in ClinicalLabExtractor(client=None).extract_entries_from_text(
            text=text, source="laboratory_analysis", visit_date=date(2024, 1, 10)
        )
    ]

###############################################################################
def test_lab_text_reference_ranges_units_and_aliases() -> None:
    assert _parse_lab_text("2024-01-08: ALT 120 U/L (ref 10-40)") == [("ALT", 120.0, "U/L", 40.0)]
    assert _parse_lab_text("2024-01-08: ALT 120 U/L (normal range 7-56 U/L)") == [("ALT", 120.0, "U/L", 56.0)]
    assert _parse_lab_text("2024-01-08: ALT 2.5 µkat/L, ULN 0.75") == [("ALT", 2.5, "µkat/L", 0.75)]
    assert _parse_lab_text("2024-01-08: Albumin 38 g/L") == [("ALB", 38.0, "g/L", None)]
    assert _parse_lab_text("2024-01-08: Bil tot 3.2 mg/dL, ULN 1.2") == [("TBIL", 3.2, "mg/dL", 1.2)]

###############################################################################
def _narrative_codes(narrative: str, *, bundle: DiliEvidenceBundle) -> set[str]:
    issues = DiliEvidenceBuilder.audit_generated_narrative(
        clinical_narrative=narrative, bundle=bundle
    )
    return {issue["code"] for issue in issues}

###############################################################################
def _gate_bundle(alp: float) -> DiliEvidenceBundle:
    # ALT 15x ULN with HCV RNA detectable; latency 29 days from 2024-02-01.
    return DiliEvidenceBuilder().build(
        payload=PatientData(
            anamnesis="Jaundice. Anti-HCV positive, HCV RNA detectable. No alcohol use.",
        ),
        drugs=PatientDrugs(
            entries=[
                DrugEntry(
                    name="Amoxicillin-clavulanate",
                    therapy_start_date="2024-02-01",
                    suspension_date="2024-02-10",
                )
            ]
        ),
        labs=PatientLabTimeline(
            entries=[
                _lab("ALT", 600, 40, "2024-03-01"),
                _lab("ALP", alp, 120, "2024-03-01"),
                _lab("TBIL", 5.0, 1.2, "2024-03-01", "mg/dL"),
            ]
        ),
        resolved_drugs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

###############################################################################
def test_narrative_gate_checks_claims_against_structured_bundle() -> None:
    bundle = _gate_bundle(alp=180)

    codes = partial(_narrative_codes, bundle=bundle)
    assert "clinical_narrative_contradicts_cause_status" in codes("Hepatitis C was ruled out.")
    assert "clinical_narrative_contradicts_cause_status" in codes("Viral causes appear unlikely.")
    assert "clinical_narrative_contradicts_injury_pattern" in codes("The pattern is cholestatic.")
    assert "clinical_narrative_contradicts_injury_pattern" in codes("Injury with an R ratio of 1.4.")
    assert "clinical_narrative_contradicts_injury_pattern" in codes(
        "The injury pattern is cholestatic (R ratio 1.4), typical of amoxicillin-clavulanate."
    )
    assert "clinical_narrative_contradicts_latency" in codes("Latency of 3 days after first dose.")
    assert "clinical_narrative_overstates_causality" in codes(
        "Amoxicillin-clavulanate is the definite cause."
    )
    assert "clinical_narrative_overstates_causality" in codes(
        "Amoxicillin-clavulanate is very likely responsible."
    )
    assert not codes(
        "The injury pattern is hepatocellular (R ratio 10). Hepatitis C cannot be "
        "excluded given detectable HCV RNA. Alcohol was excluded. Latency was about "
        "4 weeks. Amoxicillin-clavulanate typically has a latency of 1 to 8 weeks."
    )

###############################################################################
def test_narrative_gate_ignores_drug_level_and_rounded_patient_statements() -> None:
    # Sentences from the live P1 rerun (2026-10-28) that were wrongly blocked.
    hepatocellular = partial(_narrative_codes, bundle=_gate_bundle(alp=180))
    mixed = partial(_narrative_codes, bundle=_gate_bundle(alp=600))

    assert not hepatocellular(
        "LiverTox describes terbinafine hepatotoxicity as initially hepatocellular "
        "or mixed, often evolving to a prolonged cholestatic pattern."
    )
    assert not hepatocellular(
        "The observed hepatocellular pattern is compatible with the LiverTox "
        "description of atorvastatin hepatotoxicity, which notes that although "
        "cholestatic hepatitis is the most common presentation, a distinctly "
        "hepatocellular pattern with marked aminotransferase elevations occurs."
    )
    assert not hepatocellular(
        "Drug B has a far weaker case: an atypical mixed rather than "
        "aminotransferase-predominant or bland-cholestatic pattern."
    )
    assert not mixed(
        "She presented with a mixed hepatocellular-cholestatic injury pattern (R ≈ 3.0)."
    )
    assert not hepatocellular("On 2024-03-01, roughly 4.3 weeks after initiation, ALT rose.")
    assert not hepatocellular(
        "LiverTox states that the latency is characteristically short, 1 to 8 weeks "
        "after initiation."
    )
    assert not hepatocellular("The latency here (~6 weeks) falls within that window.")
    assert not hepatocellular(
        "LiverTox classifies the drug as a likelihood score B agent (highly likely "
        "cause of clinically apparent liver injury)."
    )

###############################################################################
def _rechallenge_exposure(evidence: str, *, causes_excluded: bool) -> dict:
    exposure = DiliCausalityEngine().exposure(
        DrugEntry(
            name="Drug A",
            therapy_start_date="2026-01-01",
            suspension_date="2026-01-10",
            evidence=evidence,
        ),
        {"decision_status": "accepted_exact_livertox", "accepted_livertox_name": "Drug A"},
        None,
        DiliDifferentialAssessment(all_major_causes_excluded=causes_excluded),
        PatientLabTimeline(entries=[]),
        "hepatocellular",
        "2026-01-20",
    )
    assert exposure.causality is not None
    return {
        "rechallenge": exposure.rechallenge_status,
        "category": exposure.causality.category,
    }

###############################################################################
@pytest.mark.parametrize(
    ("evidence", "causes_excluded", "rechallenge", "category"),
    [
        ("Drug A was restarted in March.", True, "present_unclear", "limited"),
        ("Rechallenge: positive.", True, "positive", "supportive"),
        (
            "Drug A was resumed on 2026-03-01 and ALT rose again to 300 U/L.",
            True,
            "positive",
            "supportive",
        ),
        (
            "Drug A was resumed without recurrence of ALT elevation.",
            True,
            "present_unclear",
            "limited",
        ),
        # Probe X-01: positive rechallenge with unresolved HCV stays limited.
        (
            "Drug A restarted later; rechallenge positive per notes.",
            False,
            "positive",
            "limited",
        ),
    ],
)
def test_positive_rechallenge_requires_recurrence_and_is_capped_by_competing_causes(
    evidence: str, causes_excluded: bool, rechallenge: str, category: str
) -> None:
    result = _rechallenge_exposure(evidence, causes_excluded=causes_excluded)

    assert result == {"rechallenge": rechallenge, "category": category}

###############################################################################
@pytest.mark.parametrize(
    ("evidence", "stop_date"),
    [
        (
            "Flucloxacillin 1 g oral four times daily from 2024-05-02 to 2024-05-12.",
            "2024-05-12",
        ),
        (
            "Cellulitis treated with flucloxacillin twice daily from 2024-02-05 "
            "to 2024-02-12 (course completed). Jaundice from 2024-03-01.",
            "2024-02-12",
        ),
        ("Flucloxacillin 02/05/2024 - 12/05/2024", "12/05/2024"),
        ("Flucloxacillin until 12.05.2024", "12.05.2024"),
        ("Flucloxacillin 500 mg since 2019, ongoing.", None),
        ("Flucloxacillin 1 g, dose increased from 2 to 4 g daily.", None),
    ],
)
def test_therapy_stop_date_is_recovered_from_explicit_range(
    evidence: str, stop_date: str | None
) -> None:
    entry = DrugsParser().normalize_entry(
        DrugEntry(name="Flucloxacillin", evidence=evidence),
        source="therapy",
        historical_flag=False,
    )

    assert entry is not None
    assert entry.suspension_date == stop_date
    assert entry.suspension_status is (True if stop_date else None)

###############################################################################
def test_recently_stopped_anamnesis_drug_stays_in_consultation_set() -> None:
    # Live C02: amoxicillin-clavulanate stopped 18 days before onset.
    def anamnesis_drug(name: str, start: str, stop: str) -> DrugEntry:
        return DrugEntry(
            name=name,
            source="anamnesis",
            historical_flag=True,
            temporal_classification="temporal_known",
            therapy_start_date=start,
            suspension_date=stop,
        )

    selection = select_relevant_candidates(
        PatientDrugs(
            entries=[
                anamnesis_drug("Amoxicillin-clavulanate", "2024-02-05", "2024-02-12"),
                anamnesis_drug("Isoniazid", "2019-01-01", "2019-06-30"),
            ]
        ),
        PatientDrugs(entries=[]),
        visit_date=date(2024, 3, 5),
        onset_date=date(2024, 3, 1),
    )

    assert [item["drug"] for item in selection.relevant] == ["Amoxicillin-clavulanate"]
    assert "Isoniazid" not in [item["drug"] for item in selection.relevant]

###############################################################################
def test_conflicting_therapy_and_anamnesis_dates_raise_blocking_issue() -> None:
    # Live C13: anamnesis start 2024-02-20 versus therapy start 2024-01-20.
    therapy = PatientDrugs(
        entries=[
            DrugEntry(
                name="Celecoxib",
                source="therapy",
                therapy_start_date="2024-01-20",
                suspension_date="2024-03-05",
            )
        ]
    )
    anamnesis = PatientDrugs(
        entries=[
            DrugEntry(
                name="Celecoxib", source="anamnesis", therapy_start_date="2024-02-20"
            ),
            DrugEntry(name="Ibuprofen", source="anamnesis", attribution="allergy"),
        ]
    )

    issues = find_drug_date_conflicts(therapy, anamnesis)

    assert [(issue["code"], issue["severity"]) for issue in issues] == [
        ("drug_date_conflict", "error")
    ]
    assert "2024-01-20" in issues[0]["message"]
    assert "2024-02-20" in issues[0]["message"]
    agreeing = PatientDrugs(
        entries=[
            DrugEntry(
                name="Celecoxib", source="anamnesis", therapy_start_date="2024-01-20"
            )
        ]
    )
    assert find_drug_date_conflicts(therapy, agreeing) == []

###############################################################################
@pytest.mark.parametrize(
    ("raw", "grade"),
    [
        ("B", "B"),
        (" e* ", "E*"),
        ("A [HD]", "A"),
        ("NAN", None),
        (float("nan"), None),
        ("Z", None),
        (None, None),
    ],
)
def test_livertox_likelihood_is_validated_against_the_livertox_scale(
    raw: object, grade: str | None
) -> None:
    assert normalize_livertox_likelihood(raw) == grade

###############################################################################
def test_malformed_livertox_likelihood_is_reported_as_unknown() -> None:
    # Live C03: TMP-SMX stored likelihood "NAN".
    exposure = DiliCausalityEngine().exposure(
        DrugEntry(name="Drug A"),
        {
            "decision_status": "accepted_exact_livertox",
            "accepted_livertox_name": "Drug A",
            "matched_livertox_row": {"likelihood_score": "NAN"},
        },
        None,
        DiliDifferentialAssessment(),
        PatientLabTimeline(entries=[]),
        "hepatocellular",
        None,
    )

    assert exposure.livertox_likelihood is None
    assert exposure.causality is not None
    assert exposure.causality.known_hepatotoxic_potential == "unknown"

###############################################################################
def test_identity_fallback_payload_is_found_by_extracted_mention() -> None:
    # Live C14: the identity fallback re-keyed "Valproic acid" as "valproate".
    payload = {"raw_mentions": ["Valproic acid"], "accepted_livertox_name": "Valproate"}
    index = index_resolved_drugs({"valproate": payload})

    assert index[normalize_drug_query_name("Valproic acid")] is payload
    assert index["valproate"] is payload
