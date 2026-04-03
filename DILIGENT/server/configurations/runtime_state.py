from __future__ import annotations

from threading import RLock
from typing import Any, Literal

from DILIGENT.server.common.constants import CLOUD_MODEL_CHOICES
from DILIGENT.server.domain.settings.configuration import LLMRuntimeDefaults
from DILIGENT.server.domain.settings.runtime import LLMRuntimeState


class LLMRuntimeConfig:
    defaults: LLMRuntimeDefaults | None = None
    _state: LLMRuntimeState = LLMRuntimeState()
    _lock: RLock = RLock()

    @classmethod
    def _update_state(cls, **updates: Any) -> LLMRuntimeState:
        changed = False
        for field_name, value in updates.items():
            if getattr(cls._state, field_name) != value:
                setattr(cls._state, field_name, value)
                changed = True
        if changed:
            cls._state.revision += 1
        return cls._state

    @classmethod
    def configure(cls, defaults: LLMRuntimeDefaults) -> None:
        with cls._lock:
            cls.defaults = defaults
            cls.reset_defaults()

    @classmethod
    def _get_defaults(cls) -> LLMRuntimeDefaults:
        if cls.defaults is None:
            raise RuntimeError("Client runtime defaults are not configured.")
        return cls.defaults

    @classmethod
    def touch_revision(cls) -> None:
        with cls._lock:
            cls._state.revision += 1

    @classmethod
    def set_parsing_model(cls, model: str) -> str:
        value = model.strip()
        with cls._lock:
            if value and value != cls._state.parsing_model:
                cls._update_state(parsing_model=value)
            return cls._state.parsing_model

    @classmethod
    def set_clinical_model(cls, model: str) -> str:
        value = model.strip()
        with cls._lock:
            if value and value != cls._state.clinical_model:
                cls._update_state(clinical_model=value)
            return cls._state.clinical_model

    @classmethod
    def set_llm_provider(cls, provider: str) -> str:
        defaults = cls._get_defaults()
        value = provider.strip().lower()
        with cls._lock:
            if not value:
                return cls._state.llm_provider
            if value not in CLOUD_MODEL_CHOICES:
                value = defaults.llm_provider
            if cls._state.llm_provider != value:
                models = CLOUD_MODEL_CHOICES.get(value, [])
                cloud_model = cls._state.cloud_model
                if cloud_model not in models:
                    cloud_model = models[0] if models else ""
                cls._update_state(llm_provider=value, cloud_model=cloud_model)
            return cls._state.llm_provider

    @classmethod
    def set_cloud_model(cls, model: str) -> str:
        value = model.strip()
        with cls._lock:
            if not value:
                if cls._state.cloud_model:
                    cls._update_state(cloud_model="")
                return cls._state.cloud_model
            models = CLOUD_MODEL_CHOICES.get(cls._state.llm_provider, [])
            if value not in models:
                fallback = models[0] if models else ""
                if fallback and fallback != cls._state.cloud_model:
                    cls._update_state(cloud_model=fallback)
                return cls._state.cloud_model
            if cls._state.cloud_model != value:
                cls._update_state(cloud_model=value)
            return cls._state.cloud_model

    @classmethod
    def set_use_cloud_services(cls, enabled: bool) -> bool:
        normalized = bool(enabled)
        with cls._lock:
            if cls._state.use_cloud_services != normalized:
                cls._update_state(use_cloud_services=normalized)
            return cls._state.use_cloud_services

    @classmethod
    def set_ollama_temperature(cls, value: float | None) -> float:
        defaults = cls._get_defaults()
        with cls._lock:
            try:
                parsed = float(value) if value is not None else cls._state.ollama_temperature
            except (TypeError, ValueError):
                parsed = defaults.ollama_temperature
            rounded = round(max(0.0, min(2.0, parsed)), 2)
            if cls._state.ollama_temperature != rounded:
                cls._update_state(ollama_temperature=rounded)
            return cls._state.ollama_temperature

    @classmethod
    def set_cloud_temperature(cls, value: float | None) -> float:
        defaults = cls._get_defaults()
        with cls._lock:
            try:
                parsed = float(value) if value is not None else cls._state.cloud_temperature
            except (TypeError, ValueError):
                parsed = defaults.cloud_temperature
            rounded = round(max(0.0, min(2.0, parsed)), 2)
            if cls._state.cloud_temperature != rounded:
                cls._update_state(cloud_temperature=rounded)
            return cls._state.cloud_temperature

    @classmethod
    def set_ollama_reasoning(cls, enabled: bool) -> bool:
        normalized = bool(enabled)
        with cls._lock:
            if cls._state.ollama_reasoning != normalized:
                cls._update_state(ollama_reasoning=normalized)
            return cls._state.ollama_reasoning

    @classmethod
    def get_parsing_model(cls) -> str:
        with cls._lock:
            return cls._state.parsing_model

    @classmethod
    def get_clinical_model(cls) -> str:
        with cls._lock:
            return cls._state.clinical_model

    @classmethod
    def get_llm_provider(cls) -> str:
        with cls._lock:
            return cls._state.llm_provider

    @classmethod
    def get_cloud_model(cls) -> str:
        with cls._lock:
            return cls._state.cloud_model

    @classmethod
    def is_cloud_enabled(cls) -> bool:
        with cls._lock:
            return cls._state.use_cloud_services

    @classmethod
    def get_ollama_temperature(cls) -> float:
        with cls._lock:
            return cls._state.ollama_temperature

    @classmethod
    def is_ollama_reasoning_enabled(cls) -> bool:
        with cls._lock:
            return cls._state.ollama_reasoning

    @classmethod
    def get_cloud_temperature(cls) -> float:
        with cls._lock:
            return cls._state.cloud_temperature

    @classmethod
    def reset_defaults(cls) -> None:
        defaults = cls._get_defaults()
        with cls._lock:
            cls._state = LLMRuntimeState(
                parsing_model=defaults.parsing_model,
                clinical_model=defaults.clinical_model,
                llm_provider=defaults.llm_provider,
                cloud_model=defaults.cloud_model,
                use_cloud_services=defaults.use_cloud_services,
                ollama_temperature=round(max(0.0, min(2.0, defaults.ollama_temperature)), 2),
                cloud_temperature=round(max(0.0, min(2.0, defaults.cloud_temperature)), 2),
                ollama_reasoning=defaults.ollama_reasoning,
                revision=0,
            )

    @classmethod
    def get_revision(cls) -> int:
        with cls._lock:
            return cls._state.revision

    @classmethod
    def resolve_provider_and_model(
        cls,
        purpose: Literal["clinical", "parser"],
    ) -> tuple[str, str]:
        with cls._lock:
            if cls._state.use_cloud_services:
                provider = cls._state.llm_provider
                model = cls._state.cloud_model.strip()
                if not model:
                    model = (
                        cls._state.parsing_model
                        if purpose == "parser"
                        else cls._state.clinical_model
                    )
            else:
                provider = "ollama"
                model = cls._state.parsing_model if purpose == "parser" else cls._state.clinical_model
            return provider, model.strip()
