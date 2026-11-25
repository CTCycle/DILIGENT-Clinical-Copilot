# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Seed the default agent of a database that has none."""

from __future__ import annotations

from common.catalogs.model_choices import (
    get_clinical_model_choices,
    get_text_extraction_model_choices,
)
from common.utils.logger import logger
from domain.agents import (
    AGENT_CONFIG_SCHEMA_VERSION,
    AgentConfig,
    AgentModelRef,
    AssessmentConfig,
    AssessmentModels,
    RevisionConfig,
    RevisionModels,
    TimelineConfig,
    agent_config_hash,
)
from repositories.agent_repository import AgentRepository

DEFAULT_AGENT_NAME = "Default agent"


###############################################################################
def default_agent_config() -> AgentConfig:
    """Local Ollama defaults: the catalog's first extraction and clinical models."""

    extraction = AgentModelRef(
        provider="ollama", model=get_text_extraction_model_choices()[0]
    )
    clinical = AgentModelRef(provider="ollama", model=get_clinical_model_choices()[0])
    return AgentConfig(
        assessment=AssessmentConfig(
            models=AssessmentModels(extraction=extraction, clinical=clinical),
        ),
        revision=RevisionConfig(models=RevisionModels(planning=clinical)),
        timeline=TimelineConfig(model=extraction),
    )


###############################################################################
def ensure_default_agent(repository: AgentRepository, config: AgentConfig) -> bool:
    """Create the default agent from ``config`` when no agent exists.

    Idempotent: nothing happens while any agent exists. Returns True when an
    agent was created.
    """

    if repository.count_agents() > 0:
        return False
    repository.create_agent(
        name=DEFAULT_AGENT_NAME,
        description="",
        configuration=config.model_dump(mode="json"),
        configuration_hash=agent_config_hash(config),
        schema_version=AGENT_CONFIG_SCHEMA_VERSION,
        is_default=True,
    )
    logger.info("Seeded the default agent.")
    return True
