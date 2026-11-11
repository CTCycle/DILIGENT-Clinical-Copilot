# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Seed default agents from the legacy singleton model configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import ValidationError

from common.utils.logger import logger
from domain.agents import (
    AGENT_CONFIG_SCHEMA_VERSION,
    AGENT_KINDS,
    AGENT_RAG_SETTING_KEYS,
    AgentKind,
    AgentModelRef,
    AgentRetrievalPolicy,
    DiliAgentConfig,
    DiliAgentModels,
    RevisionAgentConfig,
    RevisionAgentModels,
    agent_config_hash,
)
from domain.model_configs import ReasoningLevel
from repositories.agent_repository import AgentRepository

DEFAULT_AGENT_NAMES: dict[AgentKind, str] = {
    "dili": "Default DILI agent",
    "revision": "Default revision agent",
}


###############################################################################
def agent_configs_from_legacy_payload(
    payload: Mapping[str, object],
) -> dict[AgentKind, DiliAgentConfig | RevisionAgentConfig]:
    """Convert a legacy singleton model-configuration payload into agent configs.

    Legacy keys: ``use_cloud_models``/``cloud_provider`` (one provider for all
    roles), the four role models, ``reasoning_level``, ``ollama_seed`` and the
    per-agent retrieval keys inside ``rag_settings``.
    """

    def text(key: str) -> str:
        value = payload.get(key)
        return str(value).strip() if value is not None else ""

    use_cloud = payload.get("use_cloud_models", payload.get("use_cloud_services"))
    cloud_provider = text("cloud_provider") or text("llm_provider")
    provider = (cloud_provider.lower() if _as_bool(use_cloud) else "ollama") or "ollama"

    def ref(key: str) -> AgentModelRef:
        return AgentModelRef(provider=cast(Any, provider), model=text(key))

    raw_rag = payload.get("rag_settings")
    rag = {
        key: value
        for key, value in (raw_rag if isinstance(raw_rag, Mapping) else {}).items()
        if key in AGENT_RAG_SETTING_KEYS
    }
    try:
        retrieval = AgentRetrievalPolicy.model_validate(rag)
    except ValidationError:
        retrieval = AgentRetrievalPolicy()
    reasoning_level = _reasoning_level(payload.get("reasoning_level"))
    ollama_seed = _seed(payload.get("ollama_seed", 42))
    return {
        "dili": DiliAgentConfig(
            models=DiliAgentModels(
                extraction=ref("text_extraction_model"),
                clinical=ref("clinical_model"),
                timeline=ref("timeline_model"),
            ),
            reasoning_level=reasoning_level,
            ollama_seed=ollama_seed,
            retrieval=retrieval,
        ),
        "revision": RevisionAgentConfig(
            models=RevisionAgentModels(planning=ref("revision_model")),
            reasoning_level=reasoning_level,
            ollama_seed=ollama_seed,
        ),
    }


###############################################################################
def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


###############################################################################
def _reasoning_level(value: object) -> ReasoningLevel:
    try:
        return ReasoningLevel(str(value or "").strip().lower())
    except ValueError:
        return ReasoningLevel.OFF


###############################################################################
def _seed(value: object) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(str(value).strip()))
    except TypeError, ValueError:
        return None


###############################################################################
def ensure_default_agents(
    repository: AgentRepository, payload: Mapping[str, object]
) -> list[AgentKind]:
    """Seed one default agent per kind from a legacy model-configuration payload.

    Idempotent: kinds that already have a default agent are left untouched,
    and existing sessions are never modified.
    """

    created: list[AgentKind] = []
    configs = agent_configs_from_legacy_payload(payload)
    for kind in AGENT_KINDS:
        if repository.get_default(kind) is not None:
            continue
        config = configs[kind]
        repository.create_agent(
            kind=kind,
            name=DEFAULT_AGENT_NAMES[kind],
            description="Created from the existing model configuration.",
            configuration=config.model_dump(mode="json"),
            configuration_hash=agent_config_hash(config),
            schema_version=AGENT_CONFIG_SCHEMA_VERSION,
            is_default=True,
        )
        created.append(kind)
    if created:
        logger.info("Seeded default agents: %s", ", ".join(created))
    return created
