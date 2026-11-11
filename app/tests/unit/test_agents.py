# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from common.exceptions import ServiceConflictError, ServiceValidationError
from domain.agents import (
    AgentCreateRequest,
    AgentUpdateRequest,
    DiliAgentConfig,
    RevisionAgentConfig,
    agent_config_hash,
    parse_agent_config,
)
from domain.model_configs import ModelConfigSnapshot
from pydantic import ValidationError
from repositories.agent_seed import ensure_default_agents
from repositories.schemas.base import Base
from repository_fixtures import build_repository_graph
from services.agents import AgentService
from sqlalchemy import create_engine


###############################################################################
def dili_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "dili",
        "models": {
            "extraction": {"provider": "ollama", "model": "qwen3:8b"},
            "clinical": {"provider": "openai", "model": "gpt-4.1"},
            "timeline": {"provider": "ollama", "model": "qwen3:8b"},
        },
    }
    payload.update(overrides)
    return payload


###############################################################################
def revision_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "revision",
        "models": {"planning": {"provider": "openai", "model": "gpt-4.1"}},
    }
    payload.update(overrides)
    return payload


###############################################################################
def test_agent_config_discriminates_kind_and_rejects_foreign_fields() -> None:
    assert isinstance(parse_agent_config(dili_payload()), DiliAgentConfig)
    assert isinstance(parse_agent_config(revision_payload()), RevisionAgentConfig)
    with pytest.raises(ValidationError):
        parse_agent_config(revision_payload(retrieval={}))
    with pytest.raises(ValidationError):
        parse_agent_config(dili_payload(limits={"max_tasks": 2}))


###############################################################################
def test_agent_config_hash_is_canonical_and_content_sensitive() -> None:
    first = parse_agent_config(dili_payload())
    reordered = parse_agent_config({"models": dili_payload()["models"], "kind": "dili"})
    changed = parse_agent_config(
        dili_payload(instructions={"assessment_focus": "Chronology first."})
    )
    assert agent_config_hash(first) == agent_config_hash(reordered)
    assert agent_config_hash(first) != agent_config_hash(changed)


###############################################################################
def test_retrieval_policy_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValidationError):
        parse_agent_config(
            dili_payload(
                retrieval={
                    "retrieval_candidate_count": 2,
                    "retrieval_selected_count": 5,
                }
            )
        )


###############################################################################
class _FakeModelConfigService:
    def known_local_model_names(self) -> set[str]:
        return {"qwen3:8b"}

    def known_provider_model_names(self, provider: str) -> set[str]:
        return {"gpt-4.1", "gpt-4.1-mini"} if provider == "openai" else set()


###############################################################################
class _FakeAccessKeys:
    def __init__(self, active: bool) -> None:
        self.active = active

    def list_access_keys(self, provider: Any) -> list[Any]:
        return [SimpleNamespace(is_active=self.active)]


###############################################################################
@pytest.fixture
def agent_service(tmp_path: Path) -> AgentService:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'agents.db'}", future=True)
    Base.metadata.create_all(engine)
    return AgentService(
        build_repository_graph(engine=engine).agent_repository,
        model_config_service=cast(Any, _FakeModelConfigService()),
        access_key_service=cast(Any, _FakeAccessKeys(active=False)),
    )


###############################################################################
def test_default_agents_seed_once_from_legacy_configuration(
    agent_service: AgentService,
) -> None:
    snapshot = ModelConfigSnapshot(
        use_cloud_models=True,
        cloud_provider="openai",
        cloud_model="gpt-4.1-mini",
        clinical_model="gpt-4.1",
        text_extraction_model="gpt-4.1-mini",
        revision_model="gpt-4.1",
        timeline_model="gpt-4.1-mini",
        rag_settings={"retrieval_selected_count": 4, "chunk_size": 999},
    )
    repository = agent_service.repository
    assert ensure_default_agents(repository, snapshot) == ["dili", "revision"]
    assert ensure_default_agents(repository, snapshot) == []
    dili = agent_service.resolve_for_execution(kind="dili")
    assert isinstance(dili.configuration, DiliAgentConfig)
    assert dili.configuration.models.clinical.model_dump() == {
        "provider": "openai",
        "model": "gpt-4.1",
        "reasoning_level": None,
    }
    # Shared index settings never enter an agent.
    assert dili.configuration.retrieval.retrieval_selected_count == 4
    assert "chunk_size" not in dili.configuration.retrieval.model_dump()
    revision = agent_service.resolve_for_execution(kind="revision")
    assert revision.configuration.models.planning.model == "gpt-4.1"


###############################################################################
def test_agent_saves_are_versioned_and_execution_is_guarded(
    agent_service: AgentService,
) -> None:
    created = agent_service.create_agent(
        AgentCreateRequest(
            name="Precise", configuration=parse_agent_config(dili_payload())
        )
    )
    assert created.is_default is True and created.current_revision == 1
    edited = dili_payload(instructions={"assessment_focus": "Chronology first."})
    updated = agent_service.update_agent(
        created.id,
        AgentUpdateRequest(
            expected_revision=1,
            name="Precise",
            configuration=parse_agent_config(edited),
        ),
    )
    assert updated.current_revision == 2
    with pytest.raises(ServiceConflictError):
        agent_service.update_agent(
            created.id,
            AgentUpdateRequest(
                expected_revision=1,
                name="Precise",
                configuration=parse_agent_config(dili_payload()),
            ),
        )
    with pytest.raises(ServiceValidationError):
        agent_service.update_agent(
            created.id,
            AgentUpdateRequest(
                expected_revision=2,
                name="Precise",
                configuration=parse_agent_config(revision_payload()),
            ),
        )
    with pytest.raises(ServiceConflictError):
        agent_service.resolve_for_execution(
            kind="dili", agent_id=created.id, expected_revision=1
        )
    with pytest.raises(ServiceValidationError):
        agent_service.resolve_for_execution(kind="revision", agent_id=created.id)
    context = agent_service.resolve_for_execution(
        kind="dili", agent_id=created.id, expected_revision=2
    )
    assert context.revision_number == 2
    assert agent_service.resolve_revision(context.agent_revision_id) == context
    assert [
        item.revision_number
        for item in agent_service.list_revisions(created.id).revisions
    ] == [2, 1]


###############################################################################
def test_agent_validation_reports_models_tools_and_credentials(
    agent_service: AgentService,
) -> None:
    with pytest.raises(ServiceValidationError):
        agent_service.create_agent(
            AgentCreateRequest(
                name="Bad",
                configuration=parse_agent_config(
                    revision_payload(limits={"allowed_tools": ["format_disk"]})
                ),
            )
        )
    created = agent_service.create_agent(
        AgentCreateRequest(
            name="Cloud", configuration=parse_agent_config(dili_payload())
        )
    )
    result = agent_service.validate_agent(created.id)
    assert result.valid is False
    assert [issue.code for issue in result.issues] == ["provider_key_missing"]
