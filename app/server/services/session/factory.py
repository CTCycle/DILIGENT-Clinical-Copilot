from __future__ import annotations

from configurations.startup import get_server_settings
from repositories.context import RepositoryContext
from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository
from services.clinical.disease import DiseaseExtractor
from services.clinical.hepatox_core import HepatoxConsultation
from services.clinical.pattern_analyzer import (
    HepatotoxicityPatternAnalyzer,
)
from services.clinical.labs import ClinicalLabExtractor
from services.clinical.parser import DrugsParser
from services.clinical.preparation import ClinicalKnowledgePreparation
from services.clinical.rucam import RucamScoreEstimator
from services.runtime.jobs import JobManager
from services.session.payload import PayloadSanitizationService
from services.session.session_service import ClinicalSessionService

###############################################################################
def build_clinical_session_service(job_manager: JobManager) -> ClinicalSessionService:
    parser_timeout_s = float(get_server_settings().runtime.parser_llm_timeout)
    disease_timeout_s = float(get_server_settings().runtime.disease_llm_timeout)
    context = RepositoryContext.create()
    drug_catalog_repository = DrugCatalogRepository(context)
    knowledge_repository = KnowledgeRepository(context, drug_catalog_repository)
    session_repository = ClinicalSessionRepository(
        context, drug_catalog_repository, knowledge_repository
    )
    return ClinicalSessionService(
        drugs_parser=DrugsParser(timeout_s=parser_timeout_s),
        disease_extractor=DiseaseExtractor(timeout_s=disease_timeout_s),
        lab_extractor=ClinicalLabExtractor(timeout_s=parser_timeout_s),
        pattern_analyzer=HepatotoxicityPatternAnalyzer(),
        rucam_estimator=RucamScoreEstimator(),
        session_repository=session_repository,
        payload_sanitizer=PayloadSanitizationService(),
        input_preparator=ClinicalKnowledgePreparation(
            knowledge_repository=knowledge_repository,
            drug_catalog_repository=drug_catalog_repository,
        ),
        hepatox_consultation_cls=HepatoxConsultation,
        job_manager=job_manager,
    )
