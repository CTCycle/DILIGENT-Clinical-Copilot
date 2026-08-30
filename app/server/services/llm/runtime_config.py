from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
from dataclasses import replace
import hashlib
import json
from typing import Literal, cast

from common.catalogs.model_choices import (
    get_clinical_model_choices,
    get_text_extraction_model_choices,
)
from domain.model_configs import ModelConfigSnapshot, ReasoningLevel
from domain.llm.providers import CloudProviderId
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)
from repositories.serialization.provider_model_catalog_cache import (
    ProviderModelCatalogCacheSerializer,
)
from services.llm import model_catalog
from services.llm.provider_registry import provider_registry
from services.llm.generation_policy import (
    GenerationPolicy,
    GenerationPurpose,
    resolve_generation_policy as resolve_policy,
)
from services.llm.model_capabilities import (
    EffectiveInferenceConfig,
    resolve_effective_inference_config as resolve_effective_config,
    resolve_model_capabilities,
)


###############################################################################
class LLMRuntimeConfig:
    _runtime_override: ContextVar[dict[str, object] | None] = ContextVar(
        "llm_runtime_override",
        default=None,
    )

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_provider(value: str | None) -> CloudProviderId:
        normalized = (value or "").strip().lower()
        try:
            return provider_registry.get(normalized).provider_id
        except ValueError as exc:
            raise ValueError(f"Unsupported cloud provider: {normalized}") from exc

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_cloud_model(provider: str, value: str | None) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError("Cloud model is required")
        if not provider or not provider_registry.is_valid_model(
            cast(CloudProviderId, provider), normalized
        ):
            raise ValueError(
                f"Model '{normalized}' is not valid for provider '{provider}'"
            )
        return normalized

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    @staticmethod
    def _local_model_choices() -> set[str]:
        choices = set(get_clinical_model_choices()) | set(
            get_text_extraction_model_choices()
        )
        try:
            record = model_catalog.load_catalog_record(
                ProviderModelCatalogCacheSerializer(), "ollama"
            )
        except Exception:
            return choices
        choices.update(
            str(item.get("id"))
            for item in (record.models if record else [])
            if str(item.get("id") or "").strip()
        )
        return choices

    # -------------------------------------------------------------------------
    @classmethod
    def _load_snapshot(cls) -> ModelConfigSnapshot:
        snapshot = ModelConfigSerializer().load_snapshot()
        overrides = cls._runtime_override.get() or {}
        if not overrides:
            return snapshot

        use_cloud_models = cls._coerce_bool(
            overrides.get("use_cloud_models", snapshot.use_cloud_models)
        )
        provider = cls._normalize_provider(
            cls._coerce_optional_text(overrides.get("cloud_provider"))
            if "cloud_provider" in overrides
            else snapshot.cloud_provider
        )
        requested_cloud_model = (
            cls._coerce_optional_text(overrides.get("cloud_model"))
            if "cloud_model" in overrides
            else snapshot.cloud_model
        )
        cloud_model = (
            cls._normalize_cloud_model(provider, requested_cloud_model)
            if use_cloud_models
            else requested_cloud_model
        )
        role_values: dict[str, str] = {}
        for field_name in (
            "clinical_model",
            "text_extraction_model",
            "revision_model",
            "timeline_model",
        ):
            value = (
                cls._coerce_optional_text(overrides[field_name])
                if field_name in overrides
                else getattr(snapshot, field_name)
            )
            if not value:
                raise ValueError(
                    f"Runtime configuration requires an explicit '{field_name}' assignment."
                )
            role_values[field_name] = value

        if not use_cloud_models:
            local_choices = cls._local_model_choices()
            for field_name, model_name in role_values.items():
                if model_name not in local_choices:
                    raise ValueError(
                        f"Model '{model_name}' is not supported for role '{field_name}'."
                    )

        return replace(
            snapshot,
            **role_values,
            use_cloud_models=use_cloud_models,
            cloud_provider=provider,
            cloud_model=cloud_model,
            reasoning_level=cls._resolve_reasoning_level(
                overrides=overrides,
                persisted_level=snapshot.reasoning_level,
            ),
            ollama_seed=(
                cls._coerce_optional_int(overrides.get("ollama_seed"))
                if "ollama_seed" in overrides
                else snapshot.ollama_seed
            ),
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
    def _resolve_reasoning_level(
        cls,
        *,
        overrides: dict[str, object],
        persisted_level: ReasoningLevel,
    ) -> ReasoningLevel:
        if "reasoning_level" in overrides:
            return cls._coerce_reasoning_level(overrides["reasoning_level"])
        return persisted_level

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_reasoning_level(value: object) -> ReasoningLevel:
        if isinstance(value, ReasoningLevel):
            return value
        if isinstance(value, str):
            try:
                return ReasoningLevel(value.strip().lower())
            except ValueError:
                pass
        return ReasoningLevel.OFF

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
    def get_revision_model(cls) -> str:
        return (cls._load_snapshot().revision_model or "").strip()

    # -------------------------------------------------------------------------
    @classmethod
    def get_timeline_model(cls) -> str:
        return (cls._load_snapshot().timeline_model or "").strip()

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
    def get_reasoning_level(cls) -> ReasoningLevel:
        return cls._load_snapshot().reasoning_level

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
        revision_provider, revision_model = cls.resolve_provider_and_model("revision")
        timeline_provider, timeline_model = cls.resolve_provider_and_model("timeline")
        parser_policy = cls.resolve_generation_policy(
            purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
            provider=parser_provider,
            model=parser_model,
        )
        clinical_policy = cls.resolve_generation_policy(
            purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
            provider=clinical_provider,
            model=clinical_model,
        )
        revision_policy = cls.resolve_generation_policy(
            purpose=GenerationPurpose.REVISION_PLANNING,
            provider=revision_provider,
            model=revision_model,
        )
        timeline_policy = cls.resolve_generation_policy(
            purpose=GenerationPurpose.TIMELINE_EXTRACTION,
            provider=timeline_provider,
            model=timeline_model,
            timeline_complexity="moderate",
        )
        parser_effective = cls.resolve_effective_inference_config(
            purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
            provider=parser_provider,
            model=parser_model,
        )
        clinical_effective = cls.resolve_effective_inference_config(
            purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
            provider=clinical_provider,
            model=clinical_model,
        )
        revision_effective = cls.resolve_effective_inference_config(
            purpose=GenerationPurpose.REVISION_PLANNING,
            provider=revision_provider,
            model=revision_model,
        )
        timeline_effective = cls.resolve_effective_inference_config(
            purpose=GenerationPurpose.TIMELINE_EXTRACTION,
            provider=timeline_provider,
            model=timeline_model,
            timeline_complexity="moderate",
        )
        snapshot: dict[str, object] = {
            "use_cloud_services": cls.is_cloud_enabled(),
            "llm_provider": cls.get_llm_provider(),
            "cloud_model": cls.get_cloud_model(),
            "text_extraction_model": cls.get_text_extraction_model(),
            "clinical_model": cls.get_clinical_model(),
            "reasoning_level": cls.get_reasoning_level().value,
            "ollama_seed": cls.get_ollama_seed(),
            "use_rag": bool(use_rag),
            "rag_settings": cls._load_snapshot().rag_settings or {},
            "parser_provider": parser_provider,
            "parser_model": parser_model,
            "clinical_provider": clinical_provider,
            "clinical_model_resolved": clinical_model,
            "revision_model": cls.get_revision_model(),
            "timeline_model": cls.get_timeline_model(),
            "revision_provider": revision_provider,
            "revision_model_resolved": revision_model,
            "timeline_provider": timeline_provider,
            "timeline_model_resolved": timeline_model,
            "sampling_policy_version": parser_policy.policy_version,
            "parser_sampling_policy": cls._policy_snapshot(
                parser_policy, parser_effective
            ),
            "clinical_sampling_policy": cls._policy_snapshot(
                clinical_policy, clinical_effective
            ),
            "revision_sampling_policy": cls._policy_snapshot(
                revision_policy, revision_effective
            ),
            "timeline_sampling_policy": cls._policy_snapshot(
                timeline_policy, timeline_effective
            ),
        }
        canonical = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return snapshot, hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @staticmethod
    def _policy_snapshot(
        policy: GenerationPolicy,
        effective: EffectiveInferenceConfig,
    ) -> dict[str, object]:
        return {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "temperature": policy.temperature
            if policy.temperature is not None
            else "provider_default",
            "match_kind": policy.match_kind.value,
            "provider": policy.provider,
            "model": policy.model,
            "purpose": policy.purpose.value,
            "user_reasoning_level": effective.user_reasoning_level.value,
            "requested_reasoning_level": effective.requested_reasoning_level.value,
            "effective_reasoning_level": effective.effective_reasoning_level.value,
            "reasoning_adjustment_reason": effective.reasoning_adjustment_reason,
            "capability_source": effective.capability_source,
            "model_context_limit": effective.model_context_limit,
            "effective_runtime_context_limit": effective.effective_runtime_context_limit,
            "input_budget": effective.input_budget,
            "visible_output_reserve": effective.visible_output_reserve,
            "reasoning_reserve": effective.reasoning_reserve,
            "output_token_limit": effective.output_token_limit,
            "context_safety_reserve": effective.context_safety_reserve,
            "context_selection_report": dict(effective.context_selection_report),
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_generation_policy(
        *,
        purpose: GenerationPurpose,
        provider: str,
        model: str,
        user_reasoning_level: ReasoningLevel | None = None,
        timeline_complexity: str = "moderate",
    ) -> GenerationPolicy:
        return resolve_policy(
            purpose=purpose,
            provider=provider,
            model=model,
            user_reasoning_level=(
                user_reasoning_level
                if user_reasoning_level is not None
                else LLMRuntimeConfig.get_reasoning_level()
            ),
            timeline_complexity=timeline_complexity,
        )

    # -------------------------------------------------------------------------
    @classmethod
    def resolve_effective_inference_config(
        cls,
        *,
        purpose: GenerationPurpose,
        provider: str,
        model: str,
        user_reasoning_level: ReasoningLevel | None = None,
        timeline_complexity: str = "moderate",
        runtime_context_limit: int | None = None,
        selected_input_tokens: int = 0,
    ) -> EffectiveInferenceConfig:
        policy = cls.resolve_generation_policy(
            purpose=purpose,
            provider=provider,
            model=model,
            user_reasoning_level=user_reasoning_level,
            timeline_complexity=timeline_complexity,
        )
        capabilities = resolve_model_capabilities(provider=provider, model=model)
        return resolve_effective_config(
            policy=policy,
            capabilities=capabilities,
            runtime_context_limit=runtime_context_limit,
            selected_input_tokens=selected_input_tokens,
        )

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
        purpose: Literal["clinical", "parser", "revision", "timeline"],
    ) -> tuple[str, str]:
        snapshot = cls._load_snapshot()
        role_models = {
            "parser": snapshot.text_extraction_model,
            "clinical": snapshot.clinical_model,
            "revision": snapshot.revision_model,
            "timeline": snapshot.timeline_model,
        }
        local_model = (role_models[purpose] or "").strip()
        if not local_model:
            raise ValueError(
                f"Persisted model configuration has no assignment for '{purpose}'."
            )
        if snapshot.use_cloud_models:
            provider = (snapshot.cloud_provider or "").strip()
            if not provider:
                raise ValueError("Persisted model configuration has no cloud provider.")
            return provider, local_model
        return "ollama", local_model
