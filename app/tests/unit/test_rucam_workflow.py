from __future__ import annotations

from domain.clinical.entities import (
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


def test_provided_rucam_score_is_used_directly() -> None:
    estimator = RucamScoreEstimator()
    payload, analysis_drugs, timeline = _inputs()
    payload.laboratory_analysis = "RUCAM score: 7"
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

