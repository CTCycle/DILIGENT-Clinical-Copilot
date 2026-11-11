# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Seed default agents from the legacy singleton model configuration."""

from __future__ import annotations

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
from domain.model_configs import ModelConfigSnapshot
from repositories.agent_repository import AgentRepository

DEFAULT_AGENT_NAMES: dict[AgentKind, str] = {
    "dili": "Default DILI agent",
    "revision": "Default revision agent",
}


###############################################################################
def agent_configs_from_snapshot(
    snapshot: ModelConfigSnapshot,
) -> dict[AgentKind, DiliAgentConfig | RevisionAgentConfig]:
    """Convert the legacy singleton model configuration into agent configs."""

    provider = (
        (snapshot.cloud_provider or "").strip().lower()
        if snapshot.use_cloud_models
        else "ollama"
    ) or "ollama"

    def ref(model: str | None) -> AgentModelRef:
        return AgentModelRef(provider=cast(Any, provider), model=(model or "").strip())

    rag = {
        key: value
        for key, value in (snapshot.rag_settings or {}).items()
        if key in AGENT_RAG_SETTING_KEYS
    }
    try:
        retrieval = AgentRetrievalPolicy.model_validate(rag)
    except ValidationError:
        retrieval = AgentRetrievalPolicy()
    return {
        "dili": DiliAgentConfig(
            models=DiliAgentModels(
                extraction=ref(snapshot.text_extraction_model),
                clinical=ref(snapshot.clinical_model),
                timeline=ref(snapshot.timeline_model),
            ),
            reasoning_level=snapshot.reasoning_level,
            ollama_seed=snapshot.ollama_seed,
            retrieval=retrieval,
        ),
        "revision": RevisionAgentConfig(
            models=RevisionAgentModels(planning=ref(snapshot.revision_model)),
            reasoning_level=snapshot.reasoning_level,
            ollama_seed=snapshot.ollama_seed,
        ),
    }


###############################################################################
def ensure_default_agents(
    repository: AgentRepository, snapshot: ModelConfigSnapshot
) -> list[AgentKind]:
    """Seed one default agent per kind from the legacy model configuration.

    Idempotent: kinds that already have a default agent are left untouched,
    and existing sessions are never modified.
    """

    created: list[AgentKind] = []
    configs = agent_configs_from_snapshot(snapshot)
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
