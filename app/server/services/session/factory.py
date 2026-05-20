from __future__ import annotations

from configurations.startup import server_settings
from repositories.serialization.data import DataSerializer
from services.clinical.disease import DiseaseExtractor
from services.clinical.hepatox_core import (
    HepatotoxicityPatternAnalyzer,
    HepatoxConsultation,
)
from services.clinical.labs import ClinicalLabExtractor
from services.clinical.parser import DrugsParser
from services.clinical.preparation import ClinicalKnowledgePreparation
from services.clinical.rucam import RucamScoreEstimator
from services.runtime.jobs import JobManager
from services.session.payload import PayloadSanitizationService
from services.session.session_service import ClinicalSessionService


def build_clinical_session_service(job_manager: JobManager) -> ClinicalSessionService:
    timeout_s = server_settings.external_data.default_llm_timeout
    parser_timeout_s = min(float(timeout_s), 6.0)
    return ClinicalSessionService(
        drugs_parser=DrugsParser(timeout_s=parser_timeout_s),
        disease_extractor=DiseaseExtractor(timeout_s=parser_timeout_s),
        lab_extractor=ClinicalLabExtractor(timeout_s=parser_timeout_s),
        pattern_analyzer=HepatotoxicityPatternAnalyzer(),
        rucam_estimator=RucamScoreEstimator(),
        serializer=DataSerializer(),
        payload_sanitizer=PayloadSanitizationService(),
        input_preparator=ClinicalKnowledgePreparation(),
        hepatox_consultation_cls=HepatoxConsultation,
        job_manager=job_manager,
    )
