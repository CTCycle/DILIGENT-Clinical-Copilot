from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository
from repositories.session_revision_repository import SessionRevisionRepository
from repositories.session_timeline_repository import SessionTimelineRepository


def build_repository_graph(
    *, engine: Engine | None = None, session_factory: sessionmaker | None = None
) -> SimpleNamespace:
    context = RepositoryContext.create(engine=engine, session_factory=session_factory)
    drug_catalog_repository = DrugCatalogRepository(context)
    knowledge_repository = KnowledgeRepository(context, drug_catalog_repository)
    return SimpleNamespace(
        context=context,
        drug_catalog_repository=drug_catalog_repository,
        knowledge_repository=knowledge_repository,
        clinical_session_repository=ClinicalSessionRepository(
            context, drug_catalog_repository, knowledge_repository
        ),
        session_timeline_repository=SessionTimelineRepository(context),
        session_revision_repository=SessionRevisionRepository(context),
    )
