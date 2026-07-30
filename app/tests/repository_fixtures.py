from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository
from repositories.session_revision_repository import SessionRevisionRepository
from repositories.session_timeline_repository import SessionTimelineRepository

###############################################################################
@dataclass(frozen=True, slots=True)
class RepositoryGraph:
    context: RepositoryContext
    drug_catalog_repository: DrugCatalogRepository
    knowledge_repository: KnowledgeRepository
    clinical_session_repository: ClinicalSessionRepository
    session_timeline_repository: SessionTimelineRepository
    session_revision_repository: SessionRevisionRepository

###############################################################################
def build_repository_graph(
    *, engine: Engine | None = None, session_factory: sessionmaker | None = None
) -> RepositoryGraph:
    context = RepositoryContext.create(engine=engine, session_factory=session_factory)
    drug_catalog_repository = DrugCatalogRepository(context)
    knowledge_repository = KnowledgeRepository(context, drug_catalog_repository)
    clinical_session_repository = ClinicalSessionRepository(
        context, drug_catalog_repository, knowledge_repository
    )
    return RepositoryGraph(
        context=context,
        drug_catalog_repository=drug_catalog_repository,
        knowledge_repository=knowledge_repository,
        clinical_session_repository=clinical_session_repository,
        session_timeline_repository=SessionTimelineRepository(context),
        session_revision_repository=SessionRevisionRepository(
            context, clinical_session_repository
        ),
    )
