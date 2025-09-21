from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Pharmagent.app.constants import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_CLOUD_PROVIDER,
    DEFAULT_PARSING_MODEL,
)


###############################################################################
class Configuration:
    def __init__(self) -> None:
        self.configuration = {
            "model": "gpt-3.5-turbo",
        }

    # -------------------------------------------------------------------------
    def get_configuration(self) -> dict[str, Any]:
        return self.configuration


###############################################################################
@dataclass
class ClientRuntimeConfig:
    parsing_model: str = DEFAULT_PARSING_MODEL
    agent_model: str = DEFAULT_AGENT_MODEL
    llm_provider: str = DEFAULT_CLOUD_PROVIDER
    use_cloud_services: bool = False

    # ---------------------------------------------------------------------
    @classmethod
    def set_parsing_model(cls, model: str) -> str:
        value = model.strip()
        if value:
            cls.parsing_model = value
        return cls.parsing_model

    # ---------------------------------------------------------------------
    @classmethod
    def set_agent_model(cls, model: str) -> str:
        value = model.strip()
        if value:
            cls.agent_model = value
        return cls.agent_model

    # ---------------------------------------------------------------------
    @classmethod
    def set_llm_provider(cls, provider: str) -> str:
        value = provider.strip()
        if value:
            cls.llm_provider = value
        return cls.llm_provider

    # ---------------------------------------------------------------------
    @classmethod
    def set_use_cloud_services(cls, enabled: bool) -> bool:
        cls.use_cloud_services = bool(enabled)
        return cls.use_cloud_services

    # ---------------------------------------------------------------------
    @classmethod
    def get_parsing_model(cls) -> str:
        return cls.parsing_model

    # ---------------------------------------------------------------------
    @classmethod
    def get_agent_model(cls) -> str:
        return cls.agent_model

    # ---------------------------------------------------------------------
    @classmethod
    def get_llm_provider(cls) -> str:
        return cls.llm_provider

    # ---------------------------------------------------------------------
    @classmethod
    def is_cloud_enabled(cls) -> bool:
        return cls.use_cloud_services

    # ---------------------------------------------------------------------
    @classmethod
    def reset_defaults(cls) -> None:
        cls.parsing_model = DEFAULT_PARSING_MODEL
        cls.agent_model = DEFAULT_AGENT_MODEL
        cls.llm_provider = DEFAULT_CLOUD_PROVIDER
        cls.use_cloud_services = False
