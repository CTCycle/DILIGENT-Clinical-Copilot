from __future__ import annotations

from domain.clinical import (
    ClinicalLabEntry,
    DrugEntry,
    HepatotoxicityPatternScore,
    LiverInjuryOnsetContext,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
    RucamComponentAssessment,
)
from services.clinical.rucam import RucamScoreEstimator


###############################################################################
def _base_inputs() -> tuple[PatientData, PatientDrugs, PatientLabTimeline]:
    payload = PatientData(
        anamnesis="Viral causes excluded by serology.", drugs="Drug A"
    )
    drugs = PatientDrugs(
        entries=[
            DrugEntry(
                name="Drug A",
                therapy_start_date="2025-01-01",
                suspension_status=True,
            )
        ]
    )
    timeline = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=320,
                upper_limit_normal=40,
                sample_date="2025-01-10",
                source="anamnesis",
            ),
            ClinicalLabEntry(
                marker_name="ALT",
                value=180,
                upper_limit_normal=40,
                sample_date="2025-01-20",
                source="anamnesis",
            ),
        ]
    )
    return payload, drugs, timeline


###############################################################################
def test_livertox_case_rucam_score_is_never_used_as_patient_score() -> None:
    estimator = RucamScoreEstimator()
    payload, analysis, timeline = _base_inputs()
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=timeline,
        onset_context=LiverInjuryOnsetContext(
            onset_date="2025-01-10", onset_basis="first_abnormal_lab"
        ),
        pattern_score=HepatotoxicityPatternScore(classification="hepatocellular"),
        resolved_drugs={
            "drug a": {
                "match_status": "accepted_exact_livertox",
                "extracted_excerpts": [
                    "LiverTox monograph: RUCAM score 8 in representative case."
                ],
            }
        },
        report_language="en",
    )
    item = bundle.entries[0]
    assert item.total_score is None
    assert item.calculation_method == "not_calculated"
    assert item.score_source is None
    assert any("LiverTox" in limitation for limitation in item.limitations)


###############################################################################
def test_livertox_context_in_laboratory_text_is_not_patient_rucam() -> None:
    estimator = RucamScoreEstimator()

    assert (
        estimator.resolve_provided_rucam_score(
            "LiverTox monograph: RUCAM score 8 in a representative case."
        )
        is None
    )


###############################################################################
def test_laboratory_history_patient_rucam_score_has_priority() -> None:
    estimator = RucamScoreEstimator()
    payload, analysis, timeline = _base_inputs()
    payload.laboratory_analysis = "Drug A RUCAM score: 6"
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=timeline,
        onset_context=LiverInjuryOnsetContext(
            onset_date="2025-01-10", onset_basis="first_abnormal_lab"
        ),
        pattern_score=HepatotoxicityPatternScore(classification="hepatocellular"),
        resolved_drugs={
            "drug a": {
                "extracted_excerpts": [
                    "LiverTox monograph: RUCAM score 8 in representative case."
                ]
            }
        },
        report_language="en",
    )
    item = bundle.entries[0]
    assert item.total_score == 6
    assert item.calculation_method == "source_reported"
    assert item.score_source == "patient_laboratory_history"


###############################################################################
def test_unattributed_patient_rucam_is_not_copied_across_polypharmacy() -> None:
    estimator = RucamScoreEstimator()
    payload = PatientData(
        anamnesis="Viral causes excluded by serology.",
        drugs="Drug A and Drug B",
        laboratory_analysis="RUCAM score: 7",
    )
    drugs = PatientDrugs(
        entries=[
            DrugEntry(name="Drug A", therapy_start_date="2025-01-01"),
            DrugEntry(name="Drug B", therapy_start_date="2025-01-02"),
        ]
    )
    timeline = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=320,
                upper_limit_normal=40,
                sample_date="2025-01-10",
                source="laboratory_analysis",
            )
        ]
    )
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=drugs,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=timeline,
        onset_context=LiverInjuryOnsetContext(
            onset_date="2025-01-10", onset_basis="first_abnormal_lab"
        ),
        pattern_score=HepatotoxicityPatternScore(classification="hepatocellular"),
        resolved_drugs={},
        report_language="en",
    )
    assert all(item.total_score is None for item in bundle.entries)


###############################################################################
def test_livertox_likelihood_score_is_not_treated_as_rucam() -> None:
    estimator = RucamScoreEstimator()
    payload, analysis, timeline = _base_inputs()
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=timeline,
        onset_context=LiverInjuryOnsetContext(
            onset_date="2025-01-10", onset_basis="first_abnormal_lab"
        ),
        pattern_score=HepatotoxicityPatternScore(classification="hepatocellular"),
        resolved_drugs={"drug a": {"matched_livertox_row": {"likelihood_score": "B"}}},
        report_language="en",
    )
    item = bundle.entries[0]
    assert item.calculation_method != "source_reported"
    assert item.total_score is None


###############################################################################
def test_insufficient_data_returns_not_calculated_assessment() -> None:
    estimator = RucamScoreEstimator()
    payload = PatientData(anamnesis="No exclusion details.", drugs="Drug A")
    analysis = PatientDrugs(entries=[DrugEntry(name="Drug A")])
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=PatientLabTimeline(entries=[]),
        onset_context=None,
        pattern_score=HepatotoxicityPatternScore(classification="indeterminate"),
        resolved_drugs={},
        report_language="en",
    )
    item = bundle.entries[0]
    assert item.total_score is None
    assert item.causality_category == "not assessable"
    assert item.calculation_method == "not_calculated"
    assert item.data_sufficient is False


###############################################################################
def test_select_pattern_anchor_returns_qualifying_lab() -> None:
    estimator = RucamScoreEstimator()
    anchor = estimator.select_pattern_anchor(
        payload=PatientData(drugs="x"),
        lab_timeline=PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=200,
                    upper_limit_normal=40,
                    sample_date="2025-01-10",
                    source="anamnesis",
                )
            ]
        ),
    )
    assert anchor.source == "qualifying_lab"
    assert anchor.is_score_eligible is True


###############################################################################
def test_ast_alone_does_not_create_rucam_anchor() -> None:
    estimator = RucamScoreEstimator()
    anchor = estimator.select_pattern_anchor(
        payload=PatientData(drugs="x", visit_date="2025-01-10"),
        lab_timeline=PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="AST",
                    value=400,
                    upper_limit_normal=40,
                    sample_date="2025-01-10",
                    source="laboratory_analysis",
                )
            ]
        ),
    )
    assert anchor.source == "visit_proxy"
    assert anchor.is_score_eligible is False


###############################################################################
def test_visit_proxy_anchor_is_not_score_eligible() -> None:
    estimator = RucamScoreEstimator()
    anchor = estimator.select_pattern_anchor(
        payload=PatientData(drugs="x", visit_date="2025-01-10"),
        lab_timeline=PatientLabTimeline(entries=[]),
    )
    assert anchor.source == "visit_proxy"
    assert anchor.is_score_eligible is False


###############################################################################
def test_suspension_only_high_likelihood_timing_is_not_scored_incompatible() -> None:
    estimator = RucamScoreEstimator()
    payload, _, timeline = _base_inputs()
    drug = DrugEntry(
        name="Synthetic A",
        suspension_status=True,
        suspension_date="2025-01-09",
    )
    anchor = estimator.select_pattern_anchor(payload=payload, lab_timeline=timeline)
    component, _ = estimator.score_time_to_onset(
        payload=payload,
        drug=drug,
        onset_context=LiverInjuryOnsetContext(
            onset_date="2025-01-10", onset_basis="first_abnormal_lab"
        ),
        anchor=anchor,
        injury_type="hepatocellular",
        resolved_item={"matched_livertox_row": {"likelihood_score": "A"}},
    )
    assert component.score == 0
    assert component.status == "not_assessable"
    assert "do not establish latency" in (component.rationale or "")


###############################################################################
def test_standard_rucam_low_positive_scores_are_unlikely() -> None:
    estimator = RucamScoreEstimator()
    assert estimator.resolve_causality_bucket(1) == "unlikely"
    assert estimator.resolve_causality_bucket(2) == "unlikely"
    assert estimator.resolve_causality_bucket(0) == "excluded"
    assert estimator.resolve_causality_bucket(6) == "probable"
    assert estimator.resolve_causality_bucket(9) == "highly probable"


###############################################################################
def test_rechallenge_component_carries_supporting_text_when_present() -> None:
    estimator = RucamScoreEstimator()
    component = estimator.score_rechallenge(
        payload=PatientData(anamnesis="Rechallenge positive after restart."),
        drug=DrugEntry(
            name="Drug A", evidence="Drug A restarted and enzymes recurred."
        ),
    )
    assert component.status in {"scored", "not_assessable"}
    assert component.evidence


###############################################################################
def test_rucam_component_accepts_relative_exposure_date_phrase() -> None:
    evidence_date = "21 days before synthetic laboratory elevation"
    component = RucamComponentAssessment(
        component_key="time_to_onset",
        label="Time to onset",
        evidence_date=evidence_date,
    )
    assert component.evidence_date == evidence_date


###############################################################################
def _inputs():
    payload = PatientData(
        anamnesis="No alternative causes reported.",
        drugs="Drug A",
        laboratory_analysis="ALT 240 U/L (ULN 40) ALP 120 U/L (ULN 120)",
    )
    analysis_drugs = PatientDrugs(
        entries=[
            DrugEntry(
                name="Drug A",
                therapy_start_date="2026-01-01",
                suspension_status=True,
            )
        ]
    )
    timeline = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=240,
                upper_limit_normal=40,
                sample_date="2026-01-20",
                source="laboratory_analysis",
            )
        ]
    )
    return payload, analysis_drugs, timeline


###############################################################################
def test_provided_rucam_score_is_used_directly() -> None:
    estimator = RucamScoreEstimator()
    payload, analysis_drugs, timeline = _inputs()
    payload.laboratory_analysis = "Drug A RUCAM score: 7"
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis_drugs,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=timeline,
        onset_context=LiverInjuryOnsetContext(onset_basis="first_abnormal_lab"),
        pattern_score=HepatotoxicityPatternScore(classification="hepatocellular"),
        resolved_drugs={},
        report_language="en",
    )
    assert bundle.entries[0].total_score == 7
    assert bundle.entries[0].calculation_method == "source_reported"


###############################################################################
def test_complete_evidence_without_patient_score_remains_non_scoring() -> None:
    estimator = RucamScoreEstimator()
    payload, analysis_drugs, timeline = _inputs()
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis_drugs,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=timeline,
        onset_context=LiverInjuryOnsetContext(
            onset_date="2026-01-20", onset_basis="first_abnormal_lab"
        ),
        pattern_score=HepatotoxicityPatternScore(classification="hepatocellular"),
        resolved_drugs={},
        report_language="en",
    )
    assert bundle.entries[0].calculation_method == "not_calculated"
    assert bundle.entries[0].total_score is None
    assert bundle.entries[0].components


###############################################################################
def test_incomplete_inputs_skip_calculation() -> None:
    estimator = RucamScoreEstimator()
    bundle = estimator.estimate(
        payload=PatientData(drugs="Drug A"),
        analysis_drugs=PatientDrugs(entries=[DrugEntry(name="Drug A")]),
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=PatientLabTimeline(entries=[]),
        onset_context=None,
        pattern_score=HepatotoxicityPatternScore(classification="indeterminate"),
        resolved_drugs={},
        report_language="en",
    )
    assert bundle.entries[0].calculation_method == "not_calculated"
    assert bundle.entries[0].total_score is None
