from __future__ import annotations

from DILIGENT.server.domain.clinical import (
    ClinicalLabEntry,
    DiseaseContextEntry,
    DrugEntry,
    HepatotoxicityPatternScore,
    LiverInjuryOnsetContext,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
)
from DILIGENT.server.services.clinical.rucam import RucamScoreEstimator


def test_hepatocellular_case_uses_serial_alt_decline() -> None:
    estimator = RucamScoreEstimator()
    payload = PatientData(anamnesis="No alcohol use.", drugs="Drug A")
    analysis = PatientDrugs(entries=[DrugEntry(name="Drug A", therapy_start_date="2025-01-01")])
    timeline = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(marker_name="ALT", value=400, upper_limit_normal=40, sample_date="2025-01-10", source="anamnesis"),
            ClinicalLabEntry(marker_name="ALT", value=150, upper_limit_normal=40, sample_date="2025-01-18", source="anamnesis"),
        ]
    )
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=timeline,
        onset_context=LiverInjuryOnsetContext(onset_date="2025-01-10", onset_basis="first_abnormal_lab"),
        pattern_score=HepatotoxicityPatternScore(classification="hepatocellular"),
        resolved_drugs={"drug a": {"matched_livertox_row": {"likelihood_score": "B"}}},
    )
    item = bundle.entries[0]
    course = next(component for component in item.components if component.component_key == "course")
    assert course.status == "scored"
    assert course.score >= 2


def test_cholestatic_or_mixed_uses_alp_or_bilirubin_decline() -> None:
    estimator = RucamScoreEstimator()
    payload = PatientData(anamnesis="No notable history.", drugs="Drug A")
    analysis = PatientDrugs(entries=[DrugEntry(name="Drug A", therapy_start_date="2025-01-01")])
    timeline = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(marker_name="ALP", value=400, upper_limit_normal=120, sample_date="2025-01-10", source="anamnesis"),
            ClinicalLabEntry(marker_name="ALP", value=180, upper_limit_normal=120, sample_date="2025-03-10", source="anamnesis"),
        ]
    )
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=timeline,
        onset_context=LiverInjuryOnsetContext(onset_date="2025-01-10", onset_basis="first_abnormal_lab"),
        pattern_score=HepatotoxicityPatternScore(classification="mixed"),
        resolved_drugs={},
    )
    item = bundle.entries[0]
    assert item.injury_type_for_rucam == "cholestatic"
    course = next(component for component in item.components if component.component_key == "course")
    assert course.score >= 1


def test_visit_date_fallback_when_onset_missing() -> None:
    estimator = RucamScoreEstimator()
    payload = PatientData(anamnesis="No onset clue.", drugs="Drug A", visit_date="2025-01-20")
    analysis = PatientDrugs(entries=[DrugEntry(name="Drug A", therapy_start_date="2025-01-01")])
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=PatientLabTimeline(entries=[]),
        onset_context=None,
        pattern_score=HepatotoxicityPatternScore(classification="indeterminate"),
        resolved_drugs={},
    )
    onset = next(component for component in bundle.entries[0].components if component.component_key == "time_to_onset")
    assert onset.status == "scored"


def test_missing_longitudinal_data_yields_course_zero_and_low_confidence() -> None:
    estimator = RucamScoreEstimator()
    payload = PatientData(anamnesis="Sparse follow-up.", drugs="Drug A")
    analysis = PatientDrugs(entries=[DrugEntry(name="Drug A", therapy_start_date="2025-01-01")])
    timeline = PatientLabTimeline(
        entries=[ClinicalLabEntry(marker_name="ALT", value=300, upper_limit_normal=40, sample_date="2025-01-10", source="anamnesis")]
    )
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=timeline,
        onset_context=LiverInjuryOnsetContext(onset_date="2025-01-10", onset_basis="first_abnormal_lab"),
        pattern_score=HepatotoxicityPatternScore(classification="hepatocellular"),
        resolved_drugs={},
    )
    item = bundle.entries[0]
    course = next(component for component in item.components if component.component_key == "course")
    assert course.score == 0
    assert item.confidence == "low"


def test_alternative_cause_negative_scoring() -> None:
    estimator = RucamScoreEstimator()
    payload = PatientData(anamnesis="Known chronic viral hepatitis.", drugs="Drug A", has_hepatic_diseases=True)
    analysis = PatientDrugs(entries=[DrugEntry(name="Drug A")])
    disease_context = PatientDiseaseContext(
        entries=[DiseaseContextEntry(name="Chronic viral hepatitis", hepatic_related=True, evidence="HBV chronic infection.")]
    )
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=disease_context,
        lab_timeline=PatientLabTimeline(entries=[]),
        onset_context=None,
        pattern_score=HepatotoxicityPatternScore(classification="indeterminate"),
        resolved_drugs={},
    )
    non_drug = next(component for component in bundle.entries[0].components if component.component_key == "non_drug_causes")
    assert non_drug.score == -3


def test_risk_factor_extraction_and_prior_hepatotoxicity_proxy() -> None:
    estimator = RucamScoreEstimator()
    payload = PatientData(anamnesis="67 years old, occasional alcohol use.", drugs="Drug A")
    analysis = PatientDrugs(entries=[DrugEntry(name="Drug A")])
    bundle = estimator.estimate(
        payload=payload,
        analysis_drugs=analysis,
        anamnesis_drugs=PatientDrugs(entries=[]),
        disease_context=PatientDiseaseContext(entries=[]),
        lab_timeline=PatientLabTimeline(entries=[]),
        onset_context=None,
        pattern_score=HepatotoxicityPatternScore(classification="hepatocellular"),
        resolved_drugs={"drug a": {"matched_livertox_row": {"likelihood_score": "A"}}},
    )
    item = bundle.entries[0]
    risk = next(component for component in item.components if component.component_key == "risk_factors")
    prev = next(component for component in item.components if component.component_key == "previous_hepatotoxicity")
    assert risk.score >= 1
    assert prev.score == 2
