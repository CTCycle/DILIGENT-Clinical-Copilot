# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from domain.agents import (
    AGENT_CONFIG_SCHEMA_VERSION,
    AgentExecutionContext,
    AgentConfig,
    AgentModelRef,
    AgentRetrievalPolicy,
    AssessmentConfig,
    AssessmentModels,
    Capability,
    RevisionConfig,
    RevisionModels,
    TimelineConfig,
    agent_config_hash,
)
from repositories.agent_repository import AgentRepository
from repositories.agent_seed import default_agent_config
from repositories.schemas.base import Base
from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository
from repositories.session_revision_repository import SessionRevisionRepository
from repositories.session_timeline_repository import SessionTimelineRepository
from services.agents import AgentService
from services.llm.runtime_config import LLMRuntimeConfig
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


###############################################################################
@dataclass(frozen=True, slots=True)
class RepositoryGraph:
    context: RepositoryContext
    drug_catalog_repository: DrugCatalogRepository
    knowledge_repository: KnowledgeRepository
    clinical_session_repository: ClinicalSessionRepository
    session_timeline_repository: SessionTimelineRepository
    session_revision_repository: SessionRevisionRepository
    agent_repository: AgentRepository


###############################################################################
def build_repository_graph(
    *, engine: Engine | None = None, session_factory: sessionmaker | None = None
) -> RepositoryGraph:
    context = RepositoryContext.create(engine=engine, session_factory=session_factory)
    drug_catalog_repository = DrugCatalogRepository(context)
    knowledge_repository = KnowledgeRepository(context)
    clinical_session_repository = ClinicalSessionRepository(context)
    return RepositoryGraph(
        context=context,
        drug_catalog_repository=drug_catalog_repository,
        knowledge_repository=knowledge_repository,
        clinical_session_repository=clinical_session_repository,
        session_timeline_repository=SessionTimelineRepository(context),
        session_revision_repository=SessionRevisionRepository(context),
        agent_repository=AgentRepository(context),
    )


###############################################################################
def build_agent_service(
    db_path: Path,
    *,
    provider: str = "ollama",
    model: str = "qwen3:8b",
    evidence_sources: list[str] | None = None,
    active_key: bool = True,
    requested_scopes: list[str] | None = None,
) -> AgentService:
    """Agent service on a throwaway DB whose default agent runs one model."""

    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    repository = AgentRepository(RepositoryContext.create(engine=engine))
    ref = AgentModelRef(provider=cast(Any, provider), model=model)
    config = AgentConfig(
        assessment=AssessmentConfig(
            models=AssessmentModels(extraction=ref, clinical=ref),
            retrieval=AgentRetrievalPolicy(
                evidence_sources=cast(Any, evidence_sources or ["livertox"])
            ),
        ),
        revision=RevisionConfig(models=RevisionModels(planning=ref)),
        timeline=TimelineConfig(model=ref),
    )
    repository.create_agent(
        name="Test agent",
        description="",
        configuration=config.model_dump(mode="json"),
        configuration_hash=agent_config_hash(config),
        schema_version=AGENT_CONFIG_SCHEMA_VERSION,
        is_default=True,
    )

    def list_access_keys(scope: Any) -> list[Any]:
        if requested_scopes is not None:
            requested_scopes.append(scope)
        return [SimpleNamespace(is_active=active_key)]

    models = SimpleNamespace(
        known_local_model_names=lambda: set(),
        known_provider_model_names=lambda _provider: set(),
    )
    return AgentService(
        repository,
        model_config_service=cast(Any, models),
        access_key_service=cast(Any, SimpleNamespace(list_access_keys=list_access_keys)),
    )


###############################################################################
def agent_test_config(provider: str | None = None, **models: str) -> AgentConfig:
    """The default agent config with an optional provider for every role and
    per-role model names (``extraction``, ``clinical``, ``revision``, ``timeline``)."""
    data = default_agent_config().model_dump()
    refs = {
        "extraction": data["assessment"]["models"]["extraction"],
        "clinical": data["assessment"]["models"]["clinical"],
        "revision": data["revision"]["models"]["planning"],
        "timeline": data["timeline"]["model"],
    }
    unknown = set(models) - set(refs)
    if unknown:
        raise ValueError(f"Unknown agent roles: {sorted(unknown)}")
    for role, ref in refs.items():
        if provider is not None:
            ref["provider"] = provider
        if role in models:
            ref["model"] = models[role]
    return AgentConfig.model_validate(data)


###############################################################################
def agent_context(
    capability: Capability = "assessment", config: AgentConfig | None = None
) -> AgentExecutionContext:
    config = config or default_agent_config()
    return AgentExecutionContext(
        agent_id=f"test-{capability}",
        agent_name=f"Test {capability} agent",
        agent_revision_id=1,
        revision_number=1,
        capability=capability,
        config_hash=agent_config_hash(config),
        configuration=config.capability(capability),
    )


###############################################################################
@contextmanager
def agent_runtime(
    capability: Capability = "assessment", config: AgentConfig | None = None
) -> Iterator[None]:
    """Execute under an agent run override for ``config`` (default agent)."""
    context = agent_context(capability, config)
    with LLMRuntimeConfig.override_for_run(
        LLMRuntimeConfig.agent_runtime_overrides(context)
    ):
        yield
