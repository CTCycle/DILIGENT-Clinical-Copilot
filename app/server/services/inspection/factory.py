from __future__ import annotations

from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository
from repositories.session_revision_repository import SessionRevisionRepository
from repositories.session_timeline_repository import SessionTimelineRepository
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager


###############################################################################
def build_data_inspection_service(job_manager: JobManager) -> DataInspectionService:
    context = RepositoryContext.create()
    drug_catalog_repository = DrugCatalogRepository(context)
    knowledge_repository = KnowledgeRepository(context, drug_catalog_repository)
    clinical_session_repository = ClinicalSessionRepository(
        context, drug_catalog_repository, knowledge_repository
    )
    return DataInspectionService(
        clinical_session_repository=clinical_session_repository,
        drug_catalog_repository=drug_catalog_repository,
        knowledge_repository=knowledge_repository,
        session_timeline_repository=SessionTimelineRepository(context),
        session_revision_repository=SessionRevisionRepository(
            context, clinical_session_repository
        ),
        jobs=job_manager,
    )
