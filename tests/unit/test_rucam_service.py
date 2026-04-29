from __future__ import annotations

from DILIGENT.server.domain.clinical import ClinicalLabEntry, DrugEntry, HepatotoxicityPatternScore, LiverInjuryOnsetContext, PatientData, PatientDiseaseContext, PatientDrugs, PatientLabTimeline
from DILIGENT.server.services.clinical.rucam import RucamScoreEstimator


def _base_inputs() -> tuple[PatientData, PatientDrugs, PatientLabTimeline]:
    payload = PatientData(anamnesis='Viral causes excluded by serology.', drugs='Drug A')
    drugs = PatientDrugs(entries=[DrugEntry(name='Drug A', therapy_start_date='2025-01-01', suspension_status=True)])
    timeline = PatientLabTimeline(entries=[
        ClinicalLabEntry(marker_name='ALT', value=320, upper_limit_normal=40, sample_date='2025-01-10', source='anamnesis'),
        ClinicalLabEntry(marker_name='ALT', value=180, upper_limit_normal=40, sample_date='2025-01-20', source='anamnesis'),
    ])
    return payload, drugs, timeline


def test_source_reported_rucam_score_is_used_directly() -> None:
    estimator = RucamScoreEstimator()
    payload, analysis, timeline = _base_inputs()
    bundle = estimator.estimate(payload=payload, analysis_drugs=analysis, anamnesis_drugs=PatientDrugs(entries=[]), disease_context=PatientDiseaseContext(entries=[]), lab_timeline=timeline, onset_context=LiverInjuryOnsetContext(onset_date='2025-01-10', onset_basis='first_abnormal_lab'), pattern_score=HepatotoxicityPatternScore(classification='hepatocellular'), resolved_drugs={'drug a': {'extracted_excerpts': ['LiverTox monograph: RUCAM score 8 in representative case.']}}, report_language='en')
    item = bundle.entries[0]
    assert item.total_score == 8
    assert item.calculation_method == 'source_reported'
    assert item.data_sufficient is True


def test_livertox_likelihood_score_is_not_treated_as_rucam() -> None:
    estimator = RucamScoreEstimator()
    payload, analysis, timeline = _base_inputs()
    bundle = estimator.estimate(payload=payload, analysis_drugs=analysis, anamnesis_drugs=PatientDrugs(entries=[]), disease_context=PatientDiseaseContext(entries=[]), lab_timeline=timeline, onset_context=LiverInjuryOnsetContext(onset_date='2025-01-10', onset_basis='first_abnormal_lab'), pattern_score=HepatotoxicityPatternScore(classification='hepatocellular'), resolved_drugs={'drug a': {'matched_livertox_row': {'likelihood_score': 'B'}}}, report_language='en')
    item = bundle.entries[0]
    assert item.calculation_method != 'source_reported'


def test_insufficient_data_returns_not_calculated_assessment() -> None:
    estimator = RucamScoreEstimator()
    payload = PatientData(anamnesis='No exclusion details.', drugs='Drug A')
    analysis = PatientDrugs(entries=[DrugEntry(name='Drug A')])
    bundle = estimator.estimate(payload=payload, analysis_drugs=analysis, anamnesis_drugs=PatientDrugs(entries=[]), disease_context=PatientDiseaseContext(entries=[]), lab_timeline=PatientLabTimeline(entries=[]), onset_context=None, pattern_score=HepatotoxicityPatternScore(classification='indeterminate'), resolved_drugs={}, report_language='en')
    item = bundle.entries[0]
    assert item.total_score is None
    assert item.causality_category == 'not assessable'
    assert item.calculation_method == 'not_calculated'
    assert item.data_sufficient is False


def test_select_pattern_anchor_returns_qualifying_lab() -> None:
    estimator = RucamScoreEstimator()
    anchor = estimator.select_pattern_anchor(payload=PatientData(drugs='x'), lab_timeline=PatientLabTimeline(entries=[ClinicalLabEntry(marker_name='ALT', value=200, upper_limit_normal=40, sample_date='2025-01-10', source='anamnesis')]))
    assert anchor.source == 'qualifying_lab'
    assert anchor.is_score_eligible is True


def test_visit_proxy_anchor_is_not_score_eligible() -> None:
    estimator = RucamScoreEstimator()
    anchor = estimator.select_pattern_anchor(payload=PatientData(drugs='x', visit_date='2025-01-10'), lab_timeline=PatientLabTimeline(entries=[]))
    assert anchor.source == 'visit_proxy'
    assert anchor.is_score_eligible is False
