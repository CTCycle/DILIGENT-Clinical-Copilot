from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
import hashlib
import json
from typing import Literal

from configurations.startup import get_server_settings
from common.catalogs.model_choices import (
    get_clinical_model_choices,
    get_text_extraction_model_choices,
)
from domain.model_configs import ModelConfigSnapshot
from domain.settings.configuration import LLMRuntimeDefaults
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)
from services.llm.provider_registry import provider_registry

###############################################################################
class LLMRuntimeConfig:
    _runtime_override: ContextVar[dict[str, object] | None] = ContextVar(
        "llm_runtime_override",
        default=None,
    )

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_defaults() -> LLMRuntimeDefaults:
        return get_server_settings().llm_defaults

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_provider(value: str | None, fallback: str) -> str:
        normalized = (value or "").strip().lower()
        try:
            return provider_registry.get(normalized).provider_id
        except ValueError as exc:
            raise ValueError(f"Unsupported cloud provider: {normalized}") from exc

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_cloud_model(provider: str, value: str | None, fallback: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError("Cloud model is required")
        if not provider or not provider_registry.is_valid_model(provider, normalized):
            raise ValueError(
                f"Model '{normalized}' is not valid for provider '{provider}'"
            )
        return normalized

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_local_model(value: str | None, fallback: str) -> str:
        normalized = (value or "").strip()
        return normalized or fallback

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_temperature(value: float | None, fallback: float) -> float:
        if value is None:
            return round(max(0.0, min(2.0, fallback)), 2)
        try:
            parsed = float(value)
        except ValueError:
            parsed = fallback
        return round(max(0.0, min(2.0, parsed)), 2)

    # -------------------------------------------------------------------------
    @classmethod
    def _load_snapshot(cls) -> ModelConfigSnapshot:
        defaults = cls._get_defaults()
        snapshot = ModelConfigSerializer().load_snapshot()
        overrides = cls._runtime_override.get() or {}
        is_fresh = snapshot.updated_at is None and all(
            value is None
            for value in (
                snapshot.clinical_model,
                snapshot.text_extraction_model,
                snapshot.cloud_provider,
                snapshot.cloud_model,
            )
        )
        if is_fresh:
            base_provider = defaults.llm_provider
            base_cloud_model = defaults.cloud_model
            base_clinical_model = defaults.clinical_model
            base_text_extraction_model = defaults.text_extraction_model
        else:
            base_provider = snapshot.cloud_provider
            base_cloud_model = snapshot.cloud_model
            base_clinical_model = snapshot.clinical_model
            base_text_extraction_model = snapshot.text_extraction_model
            local_choices = set(get_clinical_model_choices()) | set(
                get_text_extraction_model_choices()
            )
            for role_name, model_name in (
                ("clinical", base_clinical_model),
                ("text_extraction", base_text_extraction_model),
            ):
                if model_name and model_name not in local_choices:
                    raise ValueError(
                        f"Model '{model_name}' is not supported for role '{role_name}'"
                    )
        provider = cls._normalize_provider(
            cls._coerce_optional_text(overrides.get("cloud_provider"))
            if "cloud_provider" in overrides
            else base_provider,
            defaults.llm_provider,
        )
        requested_cloud_model = (
            cls._coerce_optional_text(overrides.get("cloud_model"))
            if "cloud_model" in overrides
            else base_cloud_model
        )
        cloud_model = (
            cls._normalize_cloud_model(provider, requested_cloud_model, defaults.cloud_model)
            if requested_cloud_model or cls._coerce_bool(
                overrides.get("use_cloud_models", snapshot.use_cloud_models)
            )
            else ""
        )
        return ModelConfigSnapshot(
            clinical_model=cls._normalize_local_model(
                cls._coerce_optional_text(overrides.get("clinical_model"))
                if "clinical_model" in overrides
                else base_clinical_model,
                "" if not is_fresh else defaults.clinical_model,
            ),
            text_extraction_model=cls._normalize_local_model(
                cls._coerce_optional_text(overrides.get("text_extraction_model"))
                if "text_extraction_model" in overrides
                else base_text_extraction_model,
                "" if not is_fresh else defaults.text_extraction_model,
            ),
            use_cloud_models=(
                cls._coerce_bool(overrides.get("use_cloud_models"))
                if "use_cloud_models" in overrides
                else bool(snapshot.use_cloud_models)
            ),
            cloud_provider=provider,
            cloud_model=cloud_model,
            ollama_temperature=cls._normalize_temperature(
                cls._coerce_optional_float(overrides.get("ollama_temperature"))
                if "ollama_temperature" in overrides
                else snapshot.ollama_temperature,
                defaults.ollama_temperature,
            ),
            cloud_temperature=cls._normalize_temperature(
                cls._coerce_optional_float(overrides.get("cloud_temperature"))
                if "cloud_temperature" in overrides
                else snapshot.cloud_temperature,
                defaults.cloud_temperature,
            ),
            ollama_reasoning=(
                cls._coerce_bool(overrides.get("ollama_reasoning"))
                if "ollama_reasoning" in overrides
                else bool(snapshot.ollama_reasoning)
            ),
            ollama_seed=(
                cls._coerce_optional_int(overrides.get("ollama_seed"))
                if "ollama_seed" in overrides
                else snapshot.ollama_seed
            ),
            rag_settings=snapshot.rag_settings,
            updated_at=snapshot.updated_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_optional_text(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return max(0, int(str(value).strip()))
        except TypeError, ValueError:
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    # -------------------------------------------------------------------------
    @classmethod
    @contextmanager
    def override_for_run(cls, overrides: dict[str, object] | None):
        normalized = {str(key): value for key, value in (overrides or {}).items()}
        token: Token[dict[str, object] | None] | None = None
        if normalized:
            current = cls._runtime_override.get() or {}
            token = cls._runtime_override.set({**current, **normalized})
        try:
            yield
        finally:
            if token is not None:
                cls._runtime_override.reset(token)

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
    def get_ollama_seed(cls) -> int | None:
        return cls._load_snapshot().ollama_seed

    # -------------------------------------------------------------------------
    @classmethod
    def capture_run_snapshot(cls, *, use_rag: bool) -> tuple[dict[str, object], str]:
        """Capture the complete resolved runtime once for a clinical job."""
        parser_provider, parser_model = cls.resolve_provider_and_model("parser")
        clinical_provider, clinical_model = cls.resolve_provider_and_model("clinical")
        snapshot: dict[str, object] = {
            "use_cloud_services": cls.is_cloud_enabled(),
            "llm_provider": cls.get_llm_provider(),
            "cloud_model": cls.get_cloud_model(),
            "text_extraction_model": cls.get_text_extraction_model(),
            "clinical_model": cls.get_clinical_model(),
            "ollama_temperature": cls.get_ollama_temperature(),
            "cloud_temperature": cls.get_cloud_temperature(),
            "ollama_reasoning": cls.is_ollama_reasoning_enabled(),
            "ollama_seed": cls.get_ollama_seed(),
            "use_rag": bool(use_rag),
            "rag_settings": cls._load_snapshot().rag_settings or {},
            "parser_provider": parser_provider,
            "parser_model": parser_model,
            "clinical_provider": clinical_provider,
            "clinical_model_resolved": clinical_model,
        }
        canonical = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return snapshot, hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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
            local_choices = set(get_clinical_model_choices()) | set(
                get_text_extraction_model_choices()
            )
            if (
                local_model
                and local_model not in local_choices
                and provider_registry.is_valid_model(provider, local_model)
            ):
                return provider, local_model
            return provider, cloud_model
        return "ollama", local_model
