# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections.abc import Mapping
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
from domain.agents import AgentExecutionContext, AgentModelRef, DiliAgentConfig
from domain.model_configs import ModelConfigSnapshot, ReasoningLevel
from domain.llm.providers import CloudModelDescriptor, CloudProviderId
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


RolePurpose = Literal["clinical", "parser", "revision", "timeline"]
ROLE_PURPOSES: tuple[RolePurpose, ...] = ("parser", "clinical", "revision", "timeline")
GENERATION_PURPOSE_ROLES: dict[GenerationPurpose, RolePurpose] = {
    GenerationPurpose.STRUCTURED_EXTRACTION: "parser",
    GenerationPurpose.CLINICAL_SYNTHESIS: "clinical",
    GenerationPurpose.FAITHFUL_REWRITE: "clinical",
    GenerationPurpose.REVISION_SCAN: "revision",
    GenerationPurpose.REVISION_PLANNING: "revision",
    GenerationPurpose.REVISION_TOOL_SELECTION: "revision",
    GenerationPurpose.REVISION_EDITING: "revision",
    GenerationPurpose.REVISION_QA: "revision",
    GenerationPurpose.TIMELINE_EXTRACTION: "timeline",
}


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
        overrides = cls._active_overrides()
        if not overrides:
            return snapshot
        role_models = cls._role_models(overrides)
        if role_models:
            return cls._snapshot_from_role_models(snapshot, overrides, role_models)

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
            rag_settings=(
                dict(cast(Mapping[str, object], overrides["rag_settings"]))
                if isinstance(overrides.get("rag_settings"), Mapping)
                else snapshot.rag_settings
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _role_models(overrides: Mapping[str, object]) -> dict[str, dict[str, object]]:
        value = overrides.get("role_models")
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key): dict(cast(Mapping[str, object], item))
            for key, item in value.items()
            if isinstance(item, Mapping)
        }

    # -------------------------------------------------------------------------
    @classmethod
    def _snapshot_from_role_models(
        cls,
        persisted: ModelConfigSnapshot,
        overrides: Mapping[str, object],
        role_models: dict[str, dict[str, object]],
    ) -> ModelConfigSnapshot:
        """Project agent per-role routing onto the legacy snapshot view.

        Agent configurations are validated by the agent service, so the
        legacy single-provider validation does not apply. ``use_cloud_models``
        here means "at least one role uses a cloud provider".
        """

        def model_for(role: str) -> str | None:
            item = role_models.get(role)
            return cls._coerce_optional_text(item.get("model")) if item else None

        cloud_roles = [
            item
            for role in ("clinical", *ROLE_PURPOSES)
            if (item := role_models.get(role))
            and str(item.get("provider") or "ollama") != "ollama"
        ]
        cloud = cloud_roles[0] if cloud_roles else None
        return replace(
            persisted,
            clinical_model=model_for("clinical"),
            text_extraction_model=model_for("parser"),
            revision_model=model_for("revision"),
            timeline_model=model_for("timeline"),
            use_cloud_models=cloud is not None,
            cloud_provider=(
                str(cloud.get("provider")) if cloud else persisted.cloud_provider
            ),
            cloud_model=(
                cls._coerce_optional_text(cloud.get("model"))
                if cloud
                else persisted.cloud_model
            ),
            reasoning_level=cls._resolve_reasoning_level(
                overrides=dict(overrides),
                persisted_level=persisted.reasoning_level,
            ),
            ollama_seed=(
                cls._coerce_optional_int(overrides.get("ollama_seed"))
                if "ollama_seed" in overrides
                else persisted.ollama_seed
            ),
            rag_settings=(
                dict(cast(Mapping[str, object], overrides["rag_settings"]))
                if isinstance(overrides.get("rag_settings"), Mapping)
                else persisted.rag_settings
            ),
        )

    # -------------------------------------------------------------------------
    @classmethod
    def has_role_routing(cls) -> bool:
        """True when an agent run routes each role to its own provider."""
        return bool(cls._role_models(cls._runtime_override.get() or {}))

    # -------------------------------------------------------------------------
    @classmethod
    def agent_runtime_overrides(
        cls, context: AgentExecutionContext
    ) -> dict[str, object]:
        """Build the runtime override that executes one agent revision.

        Shared RAG index settings are read from the persisted configuration
        at launch and frozen alongside the agent's retrieval preferences.
        """

        config = context.configuration

        def entry(ref: AgentModelRef) -> dict[str, object]:
            return {
                "provider": ref.provider,
                "model": ref.model,
                "reasoning_level": (
                    ref.reasoning_level.value if ref.reasoning_level else None
                ),
            }

        role_models: dict[str, dict[str, object]] = {}
        overrides: dict[str, object] = {
            "reasoning_level": config.reasoning_level.value,
            "ollama_seed": config.ollama_seed,
            "agent": {
                "agent_id": context.agent_id,
                "agent_name": context.agent_name,
                "agent_revision_id": context.agent_revision_id,
                "agent_revision": context.revision_number,
                "agent_kind": context.kind,
                "agent_config_hash": context.config_hash,
            },
            "agent_instructions": config.instructions.model_dump(),
        }
        if isinstance(config, DiliAgentConfig):
            models = config.models
            role_models["parser"] = entry(models.extraction)
            role_models["clinical"] = entry(models.clinical)
            role_models["timeline"] = entry(models.timeline)
            # A DILI run never revises; keep the legacy view complete.
            role_models["revision"] = entry(models.clinical)
            for stage, ref in models.extraction_overrides.items():
                role_models[f"parser.{stage}"] = entry(ref)
            if models.report_synthesis is not None:
                role_models["clinical.report_synthesis"] = entry(
                    models.report_synthesis
                )
            persisted = ModelConfigSerializer().load_snapshot().rag_settings or {}
            overrides["rag_settings"] = {
                **dict(persisted),
                **config.retrieval.rag_settings(),
            }
        else:
            planning = entry(config.models.planning)
            for role in ROLE_PURPOSES:
                role_models[role] = planning
            for stage, ref in config.models.stage_overrides.items():
                role_models[f"revision.{stage}"] = entry(ref)
            overrides["revision_limits"] = config.limits.model_dump()
        overrides["role_models"] = role_models
        return overrides

    # -------------------------------------------------------------------------
    @classmethod
    def active_agent(cls) -> dict[str, object]:
        value = (cls._runtime_override.get() or {}).get("agent")
        if not isinstance(value, Mapping):
            return {}
        return dict(cast(Mapping[str, object], value))

    # -------------------------------------------------------------------------
    @classmethod
    def active_agent_instructions(cls) -> dict[str, str]:
        value = (cls._runtime_override.get() or {}).get("agent_instructions")
        if not isinstance(value, Mapping):
            return {}
        return {str(key): str(item or "") for key, item in value.items()}

    # -------------------------------------------------------------------------
    @classmethod
    def _active_overrides(cls) -> dict[str, object]:
        """Return the active override, mapping run-snapshot key aliases.

        Frozen run snapshots record ``use_cloud_services``/``llm_provider``
        (the public runtime-settings vocabulary); the snapshot fields are
        ``use_cloud_models``/``cloud_provider``.
        """
        overrides = dict(cls._runtime_override.get() or {})
        for alias, field_name in (
            ("use_cloud_services", "use_cloud_models"),
            ("llm_provider", "cloud_provider"),
        ):
            if alias in overrides and field_name not in overrides:
                overrides[field_name] = overrides[alias]
        return overrides

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
    def override_for_run(cls, overrides: Mapping[str, object] | None):
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
    def active_model_overrides(cls) -> dict[str, str]:
        """Session-only model overrides recorded in the active run snapshot."""
        value = (cls._runtime_override.get() or {}).get("model_overrides")
        return dict(value) if isinstance(value, dict) else {}

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
    def get_rag_settings(cls) -> dict[str, object]:
        return dict(cls._load_snapshot().rag_settings or {})

    # -------------------------------------------------------------------------
    @classmethod
    def capture_run_snapshot(
        cls,
        *,
        evidence_sources: list[str],
        model_overrides: dict[str, str] | None = None,
        agent_context: AgentExecutionContext | None = None,
    ) -> tuple[dict[str, object], str]:
        """Capture the complete resolved runtime once for a job.

        With ``agent_context`` the agent revision's per-role routing,
        reasoning, seed and retrieval policy are frozen; replaying the
        snapshot through ``override_for_run`` restores them exactly.
        """
        agent_overrides = (
            cls.agent_runtime_overrides(agent_context) if agent_context else {}
        )
        with cls.override_for_run(agent_overrides), cls.override_for_run(
            model_overrides
        ):
            snapshot, _ = cls._capture_run_snapshot(
                evidence_sources=evidence_sources,
                model_overrides=dict(model_overrides or {}),
            )
        for key in ("role_models", "agent", "agent_instructions", "revision_limits"):
            if key in agent_overrides:
                snapshot[key] = agent_overrides[key]
        canonical = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return snapshot, hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @classmethod
    def _capture_run_snapshot(
        cls,
        *,
        evidence_sources: list[str],
        model_overrides: dict[str, str],
    ) -> tuple[dict[str, object], str]:
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
            "evidence_sources": list(evidence_sources),
            "model_overrides": model_overrides,
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
        capabilities = resolve_model_capabilities(
            provider=effective.provider,
            model=effective.model,
            descriptor=LLMRuntimeConfig.get_model_descriptor(
                effective.provider, effective.model
            ),
        )
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
            "model_capabilities": LLMRuntimeConfig._capability_snapshot(capabilities),
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _capability_snapshot(capabilities: object) -> dict[str, object]:
        values = dict(getattr(capabilities, "__dict__", {}))
        normalized: dict[str, object] = {}
        for key, value in values.items():
            if isinstance(value, tuple):
                normalized[key] = [
                    item.value if isinstance(item, ReasoningLevel) else item
                    for item in value
                ]
            elif isinstance(value, ReasoningLevel):
                normalized[key] = value.value
            else:
                normalized[key] = value
        return normalized

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
                else LLMRuntimeConfig._reasoning_level_for_generation(purpose)
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
        descriptor: CloudModelDescriptor | None = None,
    ) -> EffectiveInferenceConfig:
        policy = cls.resolve_generation_policy(
            purpose=purpose,
            provider=provider,
            model=model,
            user_reasoning_level=user_reasoning_level,
            timeline_complexity=timeline_complexity,
        )
        capabilities = resolve_model_capabilities(
            provider=provider,
            model=model,
            descriptor=descriptor or cls.get_model_descriptor(provider, model),
        )
        return resolve_effective_config(
            policy=policy,
            capabilities=capabilities,
            runtime_context_limit=runtime_context_limit,
            selected_input_tokens=selected_input_tokens,
        )

    # -------------------------------------------------------------------------
    @classmethod
    def get_model_descriptor(
        cls, provider: str, model: str
    ) -> CloudModelDescriptor | None:
        """Read the last successful catalog descriptor without making a request."""

        try:
            record = model_catalog.load_catalog_record(
                ProviderModelCatalogCacheSerializer(), provider  # type: ignore[arg-type]
            )
        except Exception:
            return None
        for descriptor in model_catalog.cloud_models_from_record(record):
            if descriptor.id == model:
                return descriptor
        return None

    # -------------------------------------------------------------------------
    @classmethod
    def get_revision(cls) -> int:
        overrides = cls._runtime_override.get()
        if overrides:
            canonical = json.dumps(
                overrides, sort_keys=True, default=str, separators=(",", ":")
            )
            return int(hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:15], 16)
        updated_at = cls._load_snapshot().updated_at
        if not isinstance(updated_at, datetime):
            return 0
        return int(updated_at.timestamp() * 1_000_000)

    # -------------------------------------------------------------------------
    @classmethod
    def resolve_provider_and_model(
        cls,
        purpose: RolePurpose,
        *,
        stage: str | None = None,
    ) -> tuple[str, str]:
        role_models = cls._role_models(cls._runtime_override.get() or {})
        if role_models:
            item = (role_models.get(f"{purpose}.{stage}") if stage else None) or (
                role_models.get(purpose)
            )
            model = cls._coerce_optional_text(item.get("model")) if item else None
            if not item or not model:
                raise ValueError(f"Agent configuration has no model for '{purpose}'.")
            return str(item.get("provider") or "ollama"), model
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

    # -------------------------------------------------------------------------
    @classmethod
    def is_cloud_purpose(cls, purpose: RolePurpose, *, stage: str | None = None) -> bool:
        provider, _ = cls.resolve_provider_and_model(purpose, stage=stage)
        return provider != "ollama"

    # -------------------------------------------------------------------------
    @classmethod
    def get_reasoning_level_for(cls, purpose: RolePurpose) -> ReasoningLevel:
        item = cls._role_models(cls._runtime_override.get() or {}).get(purpose)
        if item and item.get("reasoning_level"):
            return cls._coerce_reasoning_level(item["reasoning_level"])
        return cls.get_reasoning_level()

    # -------------------------------------------------------------------------
    @classmethod
    def _reasoning_level_for_generation(
        cls, purpose: GenerationPurpose
    ) -> ReasoningLevel:
        role = GENERATION_PURPOSE_ROLES.get(purpose)
        return cls.get_reasoning_level_for(role) if role else cls.get_reasoning_level()
