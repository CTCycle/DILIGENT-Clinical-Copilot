# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from common.exceptions import ServiceConflictError, ServiceValidationError
from domain.agents import (
    AgentConfig,
    AgentCreateRequest,
    AgentExecutionContext,
    AgentLifecycleRequest,
    AgentUpdateRequest,
    Capability,
    agent_config_hash,
    parse_agent_config,
)
from domain.model_configs import ModelConfigSnapshot, ReasoningLevel
from pydantic import ValidationError
from repositories.agent_seed import ensure_default_agent
from repositories.schemas.base import Base
from repositories.serialization.model_configs import ModelConfigSerializer
from repository_fixtures import build_repository_graph
from services.agents import AgentService
from services.llm.runtime_config import LLMRuntimeConfig
from services.retrieval.settings import build_effective_rag_settings
from sqlalchemy import create_engine


###############################################################################
def agent_payload(**overrides: Any) -> dict[str, Any]:
    """A complete three-capability configuration payload."""

    payload: dict[str, Any] = {
        "assessment": {
            "models": {
                "extraction": {"provider": "ollama", "model": "qwen3:8b"},
                "clinical": {"provider": "openai", "model": "gpt-4.1"},
            },
        },
        "revision": {"models": {"planning": {"provider": "openai", "model": "gpt-4.1"}}},
        "timeline": {"model": {"provider": "ollama", "model": "qwen3:8b"}},
    }
    for key, value in overrides.items():
        payload[key] = {**payload[key], **value}
    return payload


# -----------------------------------------------------------------------------
def assessment_payload(**overrides: Any) -> dict[str, Any]:
    return agent_payload(assessment=overrides)


###############################################################################
def test_agent_config_requires_all_three_capabilities_and_rejects_foreign_fields() -> None:
    config = parse_agent_config(agent_payload())
    assert config.enabled_capabilities() == ["assessment", "revision", "timeline"]
    incomplete = agent_payload()
    del incomplete["timeline"]
    with pytest.raises(ValidationError):
        parse_agent_config(incomplete)
    with pytest.raises(ValidationError):
        parse_agent_config(agent_payload(revision={"retrieval": {}}))
    with pytest.raises(ValidationError):
        parse_agent_config(agent_payload(assessment={"limits": {"max_tasks": 2}}))


###############################################################################
def test_agent_config_hash_is_canonical_and_content_sensitive() -> None:
    first = parse_agent_config(agent_payload())
    reordered = parse_agent_config(dict(reversed(list(agent_payload().items()))))
    changed = parse_agent_config(
        agent_payload(assessment={"instructions": {"assessment_focus": "Chronology."}})
    )
    assert agent_config_hash(first) == agent_config_hash(reordered)
    assert agent_config_hash(first) != agent_config_hash(changed)


###############################################################################
def test_retrieval_policy_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValidationError):
        parse_agent_config(
            agent_payload(
                assessment={
                    "retrieval": {
                        "retrieval_candidate_count": 2,
                        "retrieval_selected_count": 5,
                    }
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
def test_default_agent_seeds_once_from_legacy_configuration(
    agent_service: AgentService,
) -> None:
    payload = {
        "use_cloud_models": True,
        "cloud_provider": "openai",
        "cloud_model": "gpt-4.1-mini",
        "clinical_model": "gpt-4.1",
        "text_extraction_model": "gpt-4.1-mini",
        "revision_model": "gpt-4.1",
        "timeline_model": "gpt-4.1-mini",
        "rag_settings": {"retrieval_selected_count": 4, "chunk_size": 999},
    }
    repository = agent_service.repository
    assert ensure_default_agent(repository, payload) is True
    assert ensure_default_agent(repository, payload) is False
    assert repository.count_agents() == 1
    assessment = agent_service.resolve_for_execution(capability="assessment")
    assert assessment.configuration.models.clinical.model_dump() == {  # type: ignore[union-attr]
        "provider": "openai",
        "model": "gpt-4.1",
        "reasoning_level": None,
    }
    # Shared index settings never enter an agent.
    assert assessment.configuration.retrieval.retrieval_selected_count == 4  # type: ignore[union-attr]
    assert "chunk_size" not in assessment.configuration.retrieval.model_dump()  # type: ignore[union-attr]
    assert (
        agent_service.resolve_for_execution(capability="revision")
        .configuration.models.planning.model  # type: ignore[union-attr]
        == "gpt-4.1"
    )
    assert (
        agent_service.resolve_for_execution(capability="timeline")
        .configuration.model.model  # type: ignore[union-attr]
        == "gpt-4.1-mini"
    )


###############################################################################
def test_agent_saves_are_versioned_and_execution_is_guarded(
    agent_service: AgentService,
) -> None:
    created = agent_service.create_agent(
        AgentCreateRequest(
            name="Precise", configuration=parse_agent_config(agent_payload())
        )
    )
    assert created.is_default is True and created.current_revision == 1
    assert created.capabilities == ["assessment", "revision", "timeline"]
    edited = assessment_payload(instructions={"assessment_focus": "Chronology first."})
    updated = agent_service.update_agent(
        created.id,
        AgentUpdateRequest(
            expected_revision=1, name="Precise", configuration=parse_agent_config(edited)
        ),
    )
    assert updated.current_revision == 2
    with pytest.raises(ServiceConflictError):
        agent_service.update_agent(
            created.id,
            AgentUpdateRequest(
                expected_revision=1,
                name="Precise",
                configuration=parse_agent_config(agent_payload()),
            ),
        )
    with pytest.raises(ServiceConflictError):
        agent_service.resolve_for_execution(
            capability="assessment", agent_id=created.id, expected_revision=1
        )
    context = agent_service.resolve_for_execution(
        capability="assessment", agent_id=created.id, expected_revision=2
    )
    assert context.revision_number == 2
    assert (
        agent_service.resolve_revision(context.agent_revision_id, "assessment")
        == context
    )
    assert [
        item.revision_number
        for item in agent_service.list_revisions(created.id).revisions
    ] == [2, 1]


###############################################################################
def _disable(payload: dict[str, Any], *capabilities: Capability) -> dict[str, Any]:
    for capability in capabilities:
        payload[capability] = {**payload[capability], "enabled": False}
    return payload


# -----------------------------------------------------------------------------
def test_capability_compatibility_drives_selection_and_execution(
    agent_service: AgentService,
) -> None:
    created = agent_service.create_agent(
        AgentCreateRequest(
            name="Assessment only",
            configuration=parse_agent_config(
                _disable(agent_payload(), "revision", "timeline")
            ),
        )
    )
    assert created.capabilities == ["assessment"]

    def selectable(capability: Capability) -> list[str]:
        return [
            item.id
            for item in agent_service.list_agents(selectable_for=capability).agents
        ]

    assert selectable("assessment") == [created.id]
    assert selectable("revision") == selectable("timeline") == []
    agent_service.resolve_for_execution(capability="assessment", agent_id=created.id)
    for capability in ("revision", "timeline"):
        with pytest.raises(ServiceValidationError, match="capability enabled"):
            agent_service.resolve_for_execution(
                capability=capability, agent_id=created.id
            )


# -----------------------------------------------------------------------------
def test_disabling_a_capability_preserves_every_capability_configuration(
    agent_service: AgentService,
) -> None:
    payload = agent_payload(
        timeline={"instructions": {"timeline_focus": "Dates first."}, "ollama_seed": 7},
        revision={"limits": {"max_tasks": 3}},
    )
    created = agent_service.create_agent(
        AgentCreateRequest(name="Bundle", configuration=parse_agent_config(payload))
    )
    disabled = _disable(payload, "timeline")
    updated = agent_service.update_agent(
        created.id,
        AgentUpdateRequest(
            expected_revision=1, name="Bundle", configuration=parse_agent_config(disabled)
        ),
    )
    reloaded = agent_service.get_agent(updated.id).revision.configuration
    assert reloaded.enabled_capabilities() == ["assessment", "revision"]
    # The disabled capability keeps its values; the others are untouched.
    assert reloaded.timeline.enabled is False
    assert reloaded.timeline.instructions.timeline_focus == "Dates first."
    assert reloaded.timeline.ollama_seed == 7
    assert reloaded.revision.limits.max_tasks == 3
    assert reloaded.assessment.model_dump() == parse_agent_config(payload).assessment.model_dump()
    # Re-enabling restores it as it was.
    restored = agent_service.update_agent(
        created.id,
        AgentUpdateRequest(
            expected_revision=2, name="Bundle", configuration=parse_agent_config(payload)
        ),
    )
    assert restored.revision.configuration.timeline.instructions.timeline_focus == (
        "Dates first."
    )


# -----------------------------------------------------------------------------
def test_lifecycle_gates_editing_and_selection_without_dead_ends(
    agent_service: AgentService,
) -> None:
    default = agent_service.create_agent(
        AgentCreateRequest(
            name="Default", configuration=parse_agent_config(agent_payload())
        )
    )
    spare = agent_service.create_agent(
        AgentCreateRequest(
            name="Spare", configuration=parse_agent_config(agent_payload())
        )
    )
    assert default.is_default and not spare.is_default

    def update() -> Any:
        return agent_service.update_agent(
            spare.id,
            AgentUpdateRequest(
                expected_revision=agent_service.get_agent(spare.id).current_revision,
                name="Spare",
                configuration=parse_agent_config(agent_payload()),
            ),
        )

    for flags in (
        {"is_locked": True},
        {"is_locked": False, "is_active": False},
        {"is_active": True, "is_archived": True},
    ):
        state = agent_service.set_lifecycle(spare.id, AgentLifecycleRequest(**flags))
        assert state.editable is False
        with pytest.raises(ServiceConflictError):
            update()
        if not state.is_active or state.is_archived:
            with pytest.raises(ServiceValidationError):
                agent_service.resolve_for_execution(
                    capability="assessment", agent_id=spare.id
                )
        else:  # locked agents are viewable and still runnable
            agent_service.resolve_for_execution(
                capability="assessment", agent_id=spare.id
            )
    restored = agent_service.set_lifecycle(
        spare.id, AgentLifecycleRequest(is_archived=False)
    )
    assert restored.editable is True and update().current_revision == 2

    # The default is protected; other agents can then be retired and deleted.
    with pytest.raises(ServiceConflictError):
        agent_service.set_lifecycle(default.id, AgentLifecycleRequest(is_archived=True))
    with pytest.raises(ServiceConflictError):
        agent_service.delete_agent(default.id)
    agent_service.delete_agent(spare.id)
    assert [a.id for a in agent_service.list_agents(include_archived=True).agents] == [
        default.id
    ]


###############################################################################
def test_agent_validation_reports_models_tools_and_credentials(
    agent_service: AgentService,
) -> None:
    with pytest.raises(ServiceValidationError):
        agent_service.create_agent(
            AgentCreateRequest(
                name="Bad",
                configuration=parse_agent_config(
                    agent_payload(revision={"limits": {"allowed_tools": ["format_disk"]}})
                ),
            )
        )
    # Disabled capabilities are not validated.
    agent_service.create_agent(
        AgentCreateRequest(
            name="Revision off",
            configuration=parse_agent_config(
                _disable(
                    agent_payload(revision={"limits": {"allowed_tools": ["format_disk"]}}),
                    "revision",
                )
            ),
        )
    )
    created = agent_service.create_agent(
        AgentCreateRequest(
            name="Cloud", configuration=parse_agent_config(agent_payload())
        )
    )
    result = agent_service.validate_agent(created.id)
    assert result.valid is False
    assert {issue.code for issue in result.issues} == {"provider_key_missing"}
    assert {issue.field for issue in result.issues} == {
        "assessment.models.clinical",
        "revision.models.planning",
    }


###############################################################################
def _context(
    config_payload: dict[str, Any],
    revision_id: int,
    capability: Capability = "assessment",
) -> AgentExecutionContext:
    config: AgentConfig = parse_agent_config(config_payload)
    return AgentExecutionContext(
        agent_id=f"agent-{revision_id}",
        agent_name=f"Agent {revision_id}",
        agent_revision_id=revision_id,
        revision_number=1,
        capability=capability,
        config_hash=agent_config_hash(config),
        configuration=config.capability(capability),
    )


###############################################################################
@pytest.fixture
def persisted_snapshot(monkeypatch: pytest.MonkeyPatch) -> ModelConfigSnapshot:
    snapshot = ModelConfigSnapshot(rag_settings={"chunk_size": 777})
    monkeypatch.setattr(ModelConfigSerializer, "load_snapshot", lambda self: snapshot)
    return snapshot


###############################################################################
def test_agent_context_routes_roles_stages_reasoning_and_rag(
    persisted_snapshot: ModelConfigSnapshot,
) -> None:
    payload = assessment_payload(
        reasoning_level="low",
        retrieval={"retrieval_selected_count": 5, "retrieval_candidate_count": 20},
    )
    payload["assessment"]["models"]["clinical"]["reasoning_level"] = "high"
    payload["assessment"]["models"]["extraction_overrides"] = {
        "laboratory": {"provider": "openai", "model": "gpt-4.1-mini"}
    }
    context = _context(payload, 1)

    with LLMRuntimeConfig.override_for_run(
        LLMRuntimeConfig.agent_runtime_overrides(context)
    ):
        assert LLMRuntimeConfig.resolve_provider_and_model("parser") == (
            "ollama",
            "qwen3:8b",
        )
        assert LLMRuntimeConfig.resolve_provider_and_model(
            "parser", stage="laboratory"
        ) == ("openai", "gpt-4.1-mini")
        assert LLMRuntimeConfig.resolve_provider_and_model("clinical") == (
            "openai",
            "gpt-4.1",
        )
        assert (
            LLMRuntimeConfig.get_reasoning_level_for("clinical") == ReasoningLevel.HIGH
        )
        assert LLMRuntimeConfig.get_reasoning_level_for("parser") == ReasoningLevel.LOW
        rag = build_effective_rag_settings()
        # Agent preference wins; the shared index setting is preserved.
        assert rag.retrieval_selected_count == 5
        assert rag.chunk_size == 777
        active = LLMRuntimeConfig.active_agent()
        assert active["agent_revision_id"] == 1
        assert active["agent_capability"] == "assessment"


# -----------------------------------------------------------------------------
def test_each_capability_resolves_only_its_own_configuration(
    persisted_snapshot: ModelConfigSnapshot,
) -> None:
    payload = agent_payload(
        revision={"models": {"planning": {"provider": "anthropic", "model": "claude-x"}}},
        timeline={
            "model": {"provider": "openai", "model": "gpt-4.1-mini"},
            "instructions": {"timeline_focus": "Dates first."},
            "ollama_seed": 11,
        },
    )
    prompt = "Build the timeline."
    with LLMRuntimeConfig.override_for_run(
        LLMRuntimeConfig.agent_runtime_overrides(_context(payload, 1, "timeline"))
    ):
        assert LLMRuntimeConfig.resolve_provider_and_model("timeline") == (
            "openai",
            "gpt-4.1-mini",
        )
        assert LLMRuntimeConfig.get_ollama_seed() == 11
        assert "Dates first." in LLMRuntimeConfig.with_agent_guidance(
            prompt, "timeline_focus"
        )
    with LLMRuntimeConfig.override_for_run(
        LLMRuntimeConfig.agent_runtime_overrides(_context(payload, 1, "revision"))
    ):
        assert LLMRuntimeConfig.resolve_provider_and_model("revision") == (
            "anthropic",
            "claude-x",
        )
        assert LLMRuntimeConfig.get_ollama_seed() == 42
        assert (
            LLMRuntimeConfig.with_agent_guidance(prompt, "timeline_focus") == prompt
        )


###############################################################################
def test_frozen_agent_snapshot_replays_after_live_settings_change(
    persisted_snapshot: ModelConfigSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, digest = LLMRuntimeConfig.capture_run_snapshot(
        evidence_sources=["livertox"], agent_context=_context(agent_payload(), 7)
    )
    assert snapshot["agent"]["agent_revision_id"] == 7  # type: ignore[index]
    assert len(digest) == 64
    monkeypatch.setattr(
        ModelConfigSerializer,
        "load_snapshot",
        lambda self: replace(persisted_snapshot, rag_settings={"chunk_size": 1}),
    )
    with LLMRuntimeConfig.override_for_run(snapshot):
        assert LLMRuntimeConfig.resolve_provider_and_model("clinical") == (
            "openai",
            "gpt-4.1",
        )
        assert LLMRuntimeConfig.get_clinical_model() == "gpt-4.1"
        assert LLMRuntimeConfig.get_rag_settings()["chunk_size"] == 777


###############################################################################
def test_concurrent_agent_runs_do_not_share_configuration(
    persisted_snapshot: ModelConfigSnapshot,
) -> None:
    first = agent_payload()
    second = agent_payload()
    second["assessment"]["models"]["clinical"] = {
        "provider": "anthropic",
        "model": "claude-x",
    }
    barrier = threading.Barrier(2)
    observed: dict[str, tuple[str, str]] = {}

    def run(name: str, payload: dict[str, Any], revision_id: int) -> None:
        context = _context(payload, revision_id)
        with LLMRuntimeConfig.override_for_run(
            LLMRuntimeConfig.agent_runtime_overrides(context)
        ):
            barrier.wait(timeout=5)
            observed[name] = LLMRuntimeConfig.resolve_provider_and_model("clinical")
            barrier.wait(timeout=5)

    threads = [
        threading.Thread(target=run, args=("first", first, 1)),
        threading.Thread(target=run, args=("second", second, 2)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert observed == {
        "first": ("openai", "gpt-4.1"),
        "second": ("anthropic", "claude-x"),
    }
    assert LLMRuntimeConfig.has_role_routing() is False


###############################################################################
def test_role_resolution_requires_an_active_agent_runtime() -> None:
    with pytest.raises(ValueError, match="No agent runtime is active for 'clinical'"):
        LLMRuntimeConfig.resolve_provider_and_model("clinical")
    assert LLMRuntimeConfig.is_cloud_enabled() is False
    assert LLMRuntimeConfig.get_clinical_model() == ""


###############################################################################
def test_agent_guidance_is_bounded_and_absent_by_default(
    persisted_snapshot: ModelConfigSnapshot,
) -> None:
    prompt = "Extract medications. Return JSON."
    default_context = _context(agent_payload(), 1)
    with LLMRuntimeConfig.override_for_run(
        LLMRuntimeConfig.agent_runtime_overrides(default_context)
    ):
        # Empty instructions reproduce the pre-agent prompt exactly.
        assert LLMRuntimeConfig.with_agent_guidance(prompt, "extraction_guidance") == prompt

    guided = _context(
        assessment_payload(
            instructions={
                "assessment_focus": "Weigh   chronology.",
                "extraction_guidance": "Brand names are common.",
            }
        ),
        2,
    )
    with LLMRuntimeConfig.override_for_run(
        LLMRuntimeConfig.agent_runtime_overrides(guided)
    ):
        extraction = LLMRuntimeConfig.with_agent_guidance(prompt, "extraction_guidance")
        assessment = LLMRuntimeConfig.with_agent_guidance(
            prompt, "assessment_focus", "report_style"
        )
    assert extraction.startswith(prompt)
    assert "Brand names are common." in extraction
    assert "never overrides" in extraction
    assert "Weigh chronology." in assessment
    assert "Brand names" not in assessment


###############################################################################
def test_extraction_stage_scope_routes_stage_override(
    persisted_snapshot: ModelConfigSnapshot,
) -> None:
    payload = agent_payload()
    payload["assessment"]["models"]["extraction_overrides"] = {
        "diseases": {"provider": "openai", "model": "gpt-4.1-mini"}
    }
    with LLMRuntimeConfig.override_for_run(
        LLMRuntimeConfig.agent_runtime_overrides(_context(payload, 3))
    ):
        with LLMRuntimeConfig.stage_scope("diseases"):
            assert LLMRuntimeConfig.resolve_provider_and_model("parser") == (
                "openai",
                "gpt-4.1-mini",
            )
        with LLMRuntimeConfig.stage_scope("laboratory"):
            assert LLMRuntimeConfig.resolve_provider_and_model("parser") == (
                "ollama",
                "qwen3:8b",
            )
