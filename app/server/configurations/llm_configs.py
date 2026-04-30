from __future__ import annotations

from datetime import datetime
from typing import Literal

from common.constants import CLOUD_MODEL_CHOICES
from configurations.startup import server_settings
from domain.model_configs import ModelConfigSnapshot
from domain.settings.configuration import LLMRuntimeDefaults
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)


###############################################################################
class LLMRuntimeConfig:
    @staticmethod
    def _get_defaults() -> LLMRuntimeDefaults:
        return server_settings.llm_defaults

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_provider(value: str | None, fallback: str) -> str:
        normalized = (value or "").strip().lower()
        return normalized if normalized in CLOUD_MODEL_CHOICES else fallback

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_cloud_model(provider: str, value: str | None, fallback: str) -> str:
        allowed = CLOUD_MODEL_CHOICES.get(provider, [])
        normalized = (value or "").strip()
        if normalized and normalized in allowed:
            return normalized
        if fallback in allowed:
            return fallback
        if allowed:
            return allowed[0]
        return ""

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_local_model(value: str | None, fallback: str) -> str:
        normalized = (value or "").strip()
        return normalized or fallback

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_temperature(value: float | None, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = fallback
        return round(max(0.0, min(2.0, parsed)), 2)

    # -------------------------------------------------------------------------
    @classmethod
    def _load_snapshot(cls) -> ModelConfigSnapshot:
        defaults = cls._get_defaults()
        snapshot = ModelConfigSerializer().load_snapshot()
        provider = cls._normalize_provider(
            snapshot.cloud_provider, defaults.llm_provider
        )
        cloud_model = cls._normalize_cloud_model(
            provider, snapshot.cloud_model, defaults.cloud_model
        )
        return ModelConfigSnapshot(
            clinical_model=cls._normalize_local_model(
                snapshot.clinical_model, defaults.clinical_model
            ),
            text_extraction_model=cls._normalize_local_model(
                snapshot.text_extraction_model,
                defaults.text_extraction_model,
            ),
            use_cloud_models=bool(snapshot.use_cloud_models),
            cloud_provider=provider,
            cloud_model=cloud_model,
            ollama_temperature=cls._normalize_temperature(
                snapshot.ollama_temperature,
                defaults.ollama_temperature,
            ),
            cloud_temperature=cls._normalize_temperature(
                snapshot.cloud_temperature,
                defaults.cloud_temperature,
            ),
            ollama_reasoning=bool(snapshot.ollama_reasoning),
            updated_at=snapshot.updated_at,
        )

    # -------------------------------------------------------------------------
    @classmethod
    def get_text_extraction_model(cls) -> str:
        return (cls._load_snapshot().text_extraction_model or "").strip()

    # -------------------------------------------------------------------------
    @classmethod
    def get_clinical_model(cls) -> str:
        return (cls._load_snapshot().clinical_model or "").strip()

    # -------------------------------------------------------------------------
    @classmethod
    def get_llm_provider(cls) -> str:
        return (cls._load_snapshot().cloud_provider or "").strip()

    # -------------------------------------------------------------------------
    @classmethod
    def get_cloud_model(cls) -> str:
        return (cls._load_snapshot().cloud_model or "").strip()

    # -------------------------------------------------------------------------
    @classmethod
    def is_cloud_enabled(cls) -> bool:
        return bool(cls._load_snapshot().use_cloud_models)

    # -------------------------------------------------------------------------
    @classmethod
    def get_ollama_temperature(cls) -> float:
        return float(cls._load_snapshot().ollama_temperature)

    # -------------------------------------------------------------------------
    @classmethod
    def is_ollama_reasoning_enabled(cls) -> bool:
        return bool(cls._load_snapshot().ollama_reasoning)

    # -------------------------------------------------------------------------
    @classmethod
    def get_cloud_temperature(cls) -> float:
        return float(cls._load_snapshot().cloud_temperature)

    # -------------------------------------------------------------------------
    @classmethod
    def get_revision(cls) -> int:
        updated_at = cls._load_snapshot().updated_at
        if not isinstance(updated_at, datetime):
            return 0
        return int(updated_at.timestamp() * 1_000_000)

    # -------------------------------------------------------------------------
    @classmethod
    def resolve_provider_and_model(
        cls,
        purpose: Literal["clinical", "parser"],
    ) -> tuple[str, str]:
        snapshot = cls._load_snapshot()
        local_model = (
            (snapshot.text_extraction_model or "").strip()
            if purpose == "parser"
            else (snapshot.clinical_model or "").strip()
        )
        if snapshot.use_cloud_models:
            provider = (snapshot.cloud_provider or "").strip()
            cloud_model = (snapshot.cloud_model or "").strip()
            return provider, cloud_model or local_model
        return "ollama", local_model

