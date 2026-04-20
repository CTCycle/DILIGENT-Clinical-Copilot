from __future__ import annotations

from functools import lru_cache
from threading import RLock
from typing import Any, Literal

from DILIGENT.server.common.constants import CLOUD_MODEL_CHOICES
from DILIGENT.server.domain.settings.configuration import LLMRuntimeDefaults
from DILIGENT.server.domain.settings.runtime import LLMRuntimeState


###############################################################################
class _LLMRuntimeStore:
    def __init__(self) -> None:
        self.defaults: LLMRuntimeDefaults | None = None
        self.state = LLMRuntimeState()
        self.lock = RLock()


@lru_cache(maxsize=1)
def _runtime_store() -> _LLMRuntimeStore:
    return _LLMRuntimeStore()


###############################################################################
class LLMRuntimeConfig:
    @classmethod
    def _update_state(cls, **updates: Any) -> LLMRuntimeState:
        store = _runtime_store()
        changed = False
        for field_name, value in updates.items():
            if getattr(store.state, field_name) != value:
                setattr(store.state, field_name, value)
                changed = True
        if changed:
            store.state.revision += 1
        return store.state

    # -------------------------------------------------------------------------
    @classmethod
    def configure(cls, defaults: LLMRuntimeDefaults) -> None:
        store = _runtime_store()
        with store.lock:
            store.defaults = defaults
            cls.reset_defaults()

    # -------------------------------------------------------------------------
    @classmethod
    def _get_defaults(cls) -> LLMRuntimeDefaults:
        defaults = _runtime_store().defaults
        if defaults is None:
            raise RuntimeError("Client runtime defaults are not configured.")
        return defaults

    # -------------------------------------------------------------------------
    @classmethod
    def touch_revision(cls) -> None:
        store = _runtime_store()
        with store.lock:
            store.state.revision += 1

    # -------------------------------------------------------------------------
    @classmethod
    def set_text_extraction_model(cls, model: str) -> str:
        value = model.strip()
        store = _runtime_store()
        with store.lock:
            if value and value != store.state.text_extraction_model:
                cls._update_state(text_extraction_model=value)
            return store.state.text_extraction_model

    # -------------------------------------------------------------------------
    @classmethod
    def set_clinical_model(cls, model: str) -> str:
        value = model.strip()
        store = _runtime_store()
        with store.lock:
            if value and value != store.state.clinical_model:
                cls._update_state(clinical_model=value)
            return store.state.clinical_model

    # -------------------------------------------------------------------------
    @classmethod
    def set_llm_provider(cls, provider: str) -> str:
        defaults = cls._get_defaults()
        value = provider.strip().lower()
        store = _runtime_store()
        with store.lock:
            if not value:
                return store.state.llm_provider
            if value not in CLOUD_MODEL_CHOICES:
                value = defaults.llm_provider
            if store.state.llm_provider != value:
                models = CLOUD_MODEL_CHOICES.get(value, [])
                cloud_model = store.state.cloud_model
                if cloud_model not in models:
                    cloud_model = models[0] if models else ""
                cls._update_state(llm_provider=value, cloud_model=cloud_model)
            return store.state.llm_provider

    # -------------------------------------------------------------------------
    @classmethod
    def set_cloud_model(cls, model: str) -> str:
        value = model.strip()
        store = _runtime_store()
        with store.lock:
            if not value:
                if store.state.cloud_model:
                    cls._update_state(cloud_model="")
                return store.state.cloud_model
            models = CLOUD_MODEL_CHOICES.get(store.state.llm_provider, [])
            if value not in models:
                fallback = models[0] if models else ""
                if fallback and fallback != store.state.cloud_model:
                    cls._update_state(cloud_model=fallback)
                return store.state.cloud_model
            if store.state.cloud_model != value:
                cls._update_state(cloud_model=value)
            return store.state.cloud_model

    # -------------------------------------------------------------------------
    @classmethod
    def set_use_cloud_services(cls, enabled: bool) -> bool:
        normalized = bool(enabled)
        store = _runtime_store()
        with store.lock:
            if store.state.use_cloud_services != normalized:
                cls._update_state(use_cloud_services=normalized)
            return store.state.use_cloud_services

    # -------------------------------------------------------------------------
    @classmethod
    def set_ollama_temperature(cls, value: float | None) -> float:
        defaults = cls._get_defaults()
        store = _runtime_store()
        with store.lock:
            try:
                parsed = float(value) if value is not None else store.state.ollama_temperature
            except (TypeError, ValueError):
                parsed = defaults.ollama_temperature
            rounded = round(max(0.0, min(2.0, parsed)), 2)
            if store.state.ollama_temperature != rounded:
                cls._update_state(ollama_temperature=rounded)
            return store.state.ollama_temperature

    # -------------------------------------------------------------------------
    @classmethod
    def set_cloud_temperature(cls, value: float | None) -> float:
        defaults = cls._get_defaults()
        store = _runtime_store()
        with store.lock:
            try:
                parsed = float(value) if value is not None else store.state.cloud_temperature
            except (TypeError, ValueError):
                parsed = defaults.cloud_temperature
            rounded = round(max(0.0, min(2.0, parsed)), 2)
            if store.state.cloud_temperature != rounded:
                cls._update_state(cloud_temperature=rounded)
            return store.state.cloud_temperature

    # -------------------------------------------------------------------------
    @classmethod
    def set_ollama_reasoning(cls, enabled: bool) -> bool:
        normalized = bool(enabled)
        store = _runtime_store()
        with store.lock:
            if store.state.ollama_reasoning != normalized:
                cls._update_state(ollama_reasoning=normalized)
            return store.state.ollama_reasoning

    # -------------------------------------------------------------------------
    @classmethod
    def get_text_extraction_model(cls) -> str:
        store = _runtime_store()
        with store.lock:
            return store.state.text_extraction_model

    # -------------------------------------------------------------------------
    @classmethod
    def get_clinical_model(cls) -> str:
        store = _runtime_store()
        with store.lock:
            return store.state.clinical_model

    # -------------------------------------------------------------------------
    @classmethod
    def get_llm_provider(cls) -> str:
        store = _runtime_store()
        with store.lock:
            return store.state.llm_provider

    # -------------------------------------------------------------------------
    @classmethod
    def get_cloud_model(cls) -> str:
        store = _runtime_store()
        with store.lock:
            return store.state.cloud_model

    # -------------------------------------------------------------------------
    @classmethod
    def is_cloud_enabled(cls) -> bool:
        store = _runtime_store()
        with store.lock:
            return store.state.use_cloud_services

    # -------------------------------------------------------------------------
    @classmethod
    def get_ollama_temperature(cls) -> float:
        store = _runtime_store()
        with store.lock:
            return store.state.ollama_temperature

    # -------------------------------------------------------------------------
    @classmethod
    def is_ollama_reasoning_enabled(cls) -> bool:
        store = _runtime_store()
        with store.lock:
            return store.state.ollama_reasoning

    # -------------------------------------------------------------------------
    @classmethod
    def get_cloud_temperature(cls) -> float:
        store = _runtime_store()
        with store.lock:
            return store.state.cloud_temperature

    # -------------------------------------------------------------------------
    @classmethod
    def reset_defaults(cls) -> None:
        defaults = cls._get_defaults()
        store = _runtime_store()
        with store.lock:
            store.state = LLMRuntimeState(
                text_extraction_model=defaults.text_extraction_model,
                clinical_model=defaults.clinical_model,
                llm_provider=defaults.llm_provider,
                cloud_model=defaults.cloud_model,
                use_cloud_services=defaults.use_cloud_services,
                ollama_temperature=round(max(0.0, min(2.0, defaults.ollama_temperature)), 2),
                cloud_temperature=round(max(0.0, min(2.0, defaults.cloud_temperature)), 2),
                ollama_reasoning=defaults.ollama_reasoning,
                revision=0,
            )

    # -------------------------------------------------------------------------
    @classmethod
    def get_revision(cls) -> int:
        store = _runtime_store()
        with store.lock:
            return store.state.revision

    # -------------------------------------------------------------------------
    @classmethod
    def resolve_provider_and_model(
        cls,
        purpose: Literal["clinical", "parser"],
    ) -> tuple[str, str]:
        store = _runtime_store()
        with store.lock:
            if store.state.use_cloud_services:
                provider = store.state.llm_provider
                model = store.state.cloud_model.strip()
                if not model:
                    model = (
                        store.state.text_extraction_model
                        if purpose == "parser"
                        else store.state.clinical_model
                    )
            else:
                provider = "ollama"
                model = store.state.text_extraction_model if purpose == "parser" else store.state.clinical_model
            return provider, model.strip()
