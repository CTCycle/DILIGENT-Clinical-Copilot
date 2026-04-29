from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Any

from DILIGENT.server.api import session as session_module
from DILIGENT.server.domain.clinical import ClinicalLabEntry, DrugEntry, DrugRucamAssessment, LiverInjuryOnsetContext, PatientData, PatientDiseaseContext, PatientDrugs, PatientLabTimeline, PatientRucamAssessmentBundle
from DILIGENT.server.services.clinical.preparation import HepatoxPreparedInputs

class FakeSerializer:
    def save_clinical_session(self, payload: dict[str, Any]) -> None:
        _ = payload

class FakeInputPreparator:
    async def prepare_inputs(self, *args: Any, **kwargs: Any) -> HepatoxPreparedInputs:
        _ = args
        _ = kwargs
        return HepatoxPreparedInputs(resolved_drugs={}, pattern_prompt='Pattern summary.', clinical_context='Clinical context.')

class FakeHepatoxConsultation:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ = args
        _ = kwargs
        self.llm_model = 'fake-model'

    async def run_analysis(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {'final_report': 'ok'}

class FakeDrugsParser:
    model = 'fake-parser'
    def clean_text(self, text: str) -> str:
        return text
    async def extract_drugs_from_therapy(self, text: str, **kwargs: Any) -> PatientDrugs:
        _ = text
        _ = kwargs
        return PatientDrugs(entries=[DrugEntry(name='Drug A', therapy_start_date='2024-12-20', suspension_status=True, source='therapy')])
    async def extract_drugs_from_anamnesis(self, text: str | None, **kwargs: Any) -> PatientDrugs:
        _ = text
        _ = kwargs
        return PatientDrugs(entries=[])

class FakeDiseaseExtractor:
    timeout_s = 1.0
    async def extract_diseases_from_anamnesis(self, text: str | None, **kwargs: Any) -> PatientDiseaseContext:
        _ = text
        _ = kwargs
        return PatientDiseaseContext(entries=[])

class FakeLabExtractor:
    async def extract_from_payload(self, payload: PatientData, *, progress_callback: Any | None = None) -> tuple[PatientLabTimeline, LiverInjuryOnsetContext | None]:
        _ = payload
        _ = progress_callback
        return PatientLabTimeline(entries=[
            ClinicalLabEntry(marker_name='ALT', value=300, upper_limit_normal=40, sample_date='2025-01-10', source='laboratory_analysis'),
            ClinicalLabEntry(marker_name='ALP', value=150, upper_limit_normal=120, sample_date='2025-01-10', source='laboratory_analysis'),
        ]), LiverInjuryOnsetContext(onset_date='2025-01-10', onset_basis='first_abnormal_lab')

@dataclass
class FakeRucamEstimator:
    captured_language: str | None = None
    def estimate(self, **kwargs: Any) -> PatientRucamAssessmentBundle:
        self.captured_language = kwargs.get('report_language')
        return PatientRucamAssessmentBundle(entries=[DrugRucamAssessment(drug_name='Drug A', injury_type_for_rucam='hepatocellular', total_score=6, causality_category='probable', confidence='moderate', calculation_method='structured_rucam', data_sufficient=True)])


def _payload() -> PatientData:
    return PatientData(name='Patient', visit_date=date(2025, 1, 20), anamnesis='Paziente con ittero, cause virali escluse.', drugs='Drug A 50 mg', laboratory_analysis='ALT 300 U/L')


def test_session_passes_report_language_to_rucam_estimator(monkeypatch) -> None:
    endpoint = session_module.service
    endpoint.serializer = FakeSerializer()
    endpoint.drugs_parser = FakeDrugsParser()
    endpoint.disease_extractor = FakeDiseaseExtractor()
    endpoint.lab_extractor = FakeLabExtractor()
    fake_estimator = FakeRucamEstimator()
    endpoint.rucam_estimator = fake_estimator
    monkeypatch.setattr(endpoint, 'input_preparator', FakeInputPreparator())
    monkeypatch.setattr(endpoint, 'hepatox_consultation_cls', FakeHepatoxConsultation)

    asyncio.run(endpoint.process_single_patient(_payload()))
    assert fake_estimator.captured_language == 'it'
