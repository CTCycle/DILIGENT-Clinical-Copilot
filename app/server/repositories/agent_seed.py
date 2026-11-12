# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Seed the default agent from the legacy singleton model configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import ValidationError

from common.utils.logger import logger
from domain.agents import (
    AGENT_CONFIG_SCHEMA_VERSION,
    AGENT_RAG_SETTING_KEYS,
    AgentConfig,
    AgentModelRef,
    AgentRetrievalPolicy,
    AssessmentConfig,
    AssessmentModels,
    RevisionConfig,
    RevisionModels,
    TimelineConfig,
    agent_config_hash,
)
from domain.model_configs import ReasoningLevel
from repositories.agent_repository import AgentRepository

DEFAULT_AGENT_NAME = "Default agent"


###############################################################################
def agent_config_from_legacy_payload(payload: Mapping[str, object]) -> AgentConfig:
    """Convert a legacy singleton model-configuration payload into one agent config.

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
    return AgentConfig(
        assessment=AssessmentConfig(
            models=AssessmentModels(
                extraction=ref("text_extraction_model"),
                clinical=ref("clinical_model"),
            ),
            reasoning_level=reasoning_level,
            ollama_seed=ollama_seed,
            retrieval=retrieval,
        ),
        revision=RevisionConfig(
            models=RevisionModels(planning=ref("revision_model")),
            reasoning_level=reasoning_level,
            ollama_seed=ollama_seed,
        ),
        timeline=TimelineConfig(
            model=ref("timeline_model"),
            reasoning_level=reasoning_level,
            ollama_seed=ollama_seed,
        ),
    )


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
def ensure_default_agent(
    repository: AgentRepository, payload: Mapping[str, object]
) -> bool:
    """Seed one default agent from a legacy model-configuration payload.

    Idempotent: nothing happens while any agent exists, and existing sessions
    are never modified. Returns True when an agent was created.
    """

    if repository.count_agents() > 0:
        return False
    config = agent_config_from_legacy_payload(payload)
    repository.create_agent(
        name=DEFAULT_AGENT_NAME,
        description="Created from the existing model configuration.",
        configuration=config.model_dump(mode="json"),
        configuration_hash=agent_config_hash(config),
        schema_version=AGENT_CONFIG_SCHEMA_VERSION,
        is_default=True,
    )
    logger.info("Seeded the default agent.")
    return True
