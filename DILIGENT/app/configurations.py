from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from DILIGENT.app.constants import (
    CLOUD_MODEL_CHOICES,
    DEFAULT_AGENT_MODEL,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_CLOUD_PROVIDER,
    DEFAULT_ENHANCER_MODEL,
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
    enhancer_model: str = DEFAULT_ENHANCER_MODEL
    llm_provider: str = DEFAULT_CLOUD_PROVIDER
    cloud_model: str = DEFAULT_CLOUD_MODEL
    use_cloud_services: bool = False
    ollama_temperature: float = 0.7
    ollama_reasoning: bool = False
    revision: int = 0

    # ---------------------------------------------------------------------
    @classmethod
    def _touch_revision(cls) -> None:
        cls.revision += 1

    # ---------------------------------------------------------------------
    @classmethod
    def set_parsing_model(cls, model: str) -> str:
        value = model.strip()
        if value and value != cls.parsing_model:
            cls.parsing_model = value
            cls._touch_revision()
        return cls.parsing_model

    # ---------------------------------------------------------------------
    @classmethod
    def set_agent_model(cls, model: str) -> str:
        value = model.strip()
        if value and value != cls.agent_model:
            cls.agent_model = value
            cls._touch_revision()
        return cls.agent_model

    # ---------------------------------------------------------------------
    @classmethod
    def set_enhancer_model(cls, model: str) -> str:
        value = model.strip()
        if value and value != cls.enhancer_model:
            cls.enhancer_model = value
            cls._touch_revision()
        return cls.enhancer_model

    # ---------------------------------------------------------------------
    @classmethod
    def set_llm_provider(cls, provider: str) -> str:
        value = provider.strip()
        if value and value != cls.llm_provider:
            cls.llm_provider = value
            models = CLOUD_MODEL_CHOICES.get(cls.llm_provider, [])
            if cls.cloud_model not in models:
                cls.cloud_model = models[0] if models else ""
            cls._touch_revision()
        return cls.llm_provider

    # ---------------------------------------------------------------------
    @classmethod
    def set_cloud_model(cls, model: str) -> str:
        value = model.strip()
        if not value:
            if cls.cloud_model:
                cls.cloud_model = ""
                cls._touch_revision()
            return cls.cloud_model
        models = CLOUD_MODEL_CHOICES.get(cls.llm_provider, [])
        if value not in models:
            if models:
                if cls.cloud_model != models[0]:
                    cls.cloud_model = models[0]
                    cls._touch_revision()
            return cls.cloud_model
        if cls.cloud_model != value:
            cls.cloud_model = value
            cls._touch_revision()
        return cls.cloud_model

    # ---------------------------------------------------------------------
    @classmethod
    def set_use_cloud_services(cls, enabled: bool) -> bool:
        normalized = bool(enabled)
        if cls.use_cloud_services != normalized:
            cls.use_cloud_services = normalized
            cls._touch_revision()
        return cls.use_cloud_services

    # ---------------------------------------------------------------------
    @classmethod
    def set_ollama_temperature(cls, value: float | None) -> float:
        try:
            parsed = float(value) if value is not None else cls.ollama_temperature
        except (TypeError, ValueError):
            parsed = cls.ollama_temperature
        parsed = max(0.0, min(2.0, parsed))
        rounded = round(parsed, 2)
        if cls.ollama_temperature != rounded:
            cls.ollama_temperature = rounded
            cls._touch_revision()
        return cls.ollama_temperature

    # ---------------------------------------------------------------------
    @classmethod
    def set_ollama_reasoning(cls, enabled: bool) -> bool:
        normalized = bool(enabled)
        if cls.ollama_reasoning != normalized:
            cls.ollama_reasoning = normalized
            cls._touch_revision()
        return cls.ollama_reasoning

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
    def get_enhancer_model(cls) -> str:
        return cls.enhancer_model

    # ---------------------------------------------------------------------
    @classmethod
    def get_llm_provider(cls) -> str:
        return cls.llm_provider

    # ---------------------------------------------------------------------
    @classmethod
    def get_cloud_model(cls) -> str:
        return cls.cloud_model

    # ---------------------------------------------------------------------
    @classmethod
    def is_cloud_enabled(cls) -> bool:
        return cls.use_cloud_services

    # ---------------------------------------------------------------------
    @classmethod
    def get_ollama_temperature(cls) -> float:
        return cls.ollama_temperature

    # ---------------------------------------------------------------------
    @classmethod
    def is_ollama_reasoning_enabled(cls) -> bool:
        return cls.ollama_reasoning

    # ---------------------------------------------------------------------
    @classmethod
    def reset_defaults(cls) -> None:
        cls.parsing_model = DEFAULT_PARSING_MODEL
        cls.agent_model = DEFAULT_AGENT_MODEL
        cls.enhancer_model = DEFAULT_ENHANCER_MODEL
        cls.llm_provider = DEFAULT_CLOUD_PROVIDER
        cls.cloud_model = DEFAULT_CLOUD_MODEL
        cls.use_cloud_services = False
        cls.ollama_temperature = 0.7
        cls.ollama_reasoning = False
        cls.revision = 0

    # ---------------------------------------------------------------------
    @classmethod
    def get_revision(cls) -> int:
        return cls.revision

    # ---------------------------------------------------------------------
    @classmethod
    def resolve_provider_and_model(
        cls, purpose: Literal["agent", "parser", "enhancer"]
    ) -> tuple[str, str]:
        if cls.is_cloud_enabled():
            provider = cls.get_llm_provider()
            model = cls.get_cloud_model().strip()
            if not model:
                if purpose == "parser":
                    model = cls.get_parsing_model()
                elif purpose == "enhancer":
                    model = cls.get_enhancer_model()
                else:
                    model = cls.get_agent_model()
        else:
            provider = "ollama"
            if purpose == "parser":
                model = cls.get_parsing_model()
            elif purpose == "enhancer":
                model = cls.get_enhancer_model()
            else:
                model = cls.get_agent_model()
        return provider, model.strip()
