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
                name="Drug A", therapy_start_date="2025-01-01", suspension_status=True
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
def test_source_reported_rucam_score_is_used_directly() -> None:
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
                "extracted_excerpts": [
                    "LiverTox monograph: RUCAM score 8 in representative case."
                ]
            }
        },
        report_language="en",
    )
    item = bundle.entries[0]
    assert item.total_score == 8
    assert item.calculation_method == "source_reported"
    assert item.data_sufficient is True

###############################################################################
def test_laboratory_history_rucam_score_has_priority() -> None:
    estimator = RucamScoreEstimator()
    payload, analysis, timeline = _base_inputs()
    payload.laboratory_analysis = "RUCAM score: 6"
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
    assert item.score_source == "laboratory_history"

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
def test_low_positive_rucam_scores_are_indeterminate() -> None:
    estimator = RucamScoreEstimator()
    assert estimator.resolve_causality_bucket(1) == "indeterminate"
    assert estimator.resolve_causality_bucket(2) == "indeterminate"
    assert estimator.resolve_causality_bucket(0) == "excluded"

###############################################################################
def test_rechallenge_component_carries_supporting_text_when_present() -> None:
    estimator = RucamScoreEstimator()
    component = estimator.score_rechallenge(
        payload=PatientData(anamnesis="Rechallenge positive after restart."),
        drug=DrugEntry(name="Drug A", evidence="Drug A restarted and enzymes recurred."),
    )
    assert component.status in {"scored", "not_assessable"}
    assert component.evidence
