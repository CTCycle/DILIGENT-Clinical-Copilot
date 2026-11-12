# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
import hashlib
import json
from typing import Literal, cast

from common.prompts.agent_guidance import append_agent_guidance
from domain.agents import (
    AgentExecutionContext,
    AgentModelRef,
    AssessmentConfig,
    RevisionConfig,
    TimelineConfig,
)
from domain.model_configs import ReasoningLevel
from domain.llm.providers import CloudModelDescriptor
from domain.settings.configuration import SHARED_RAG_SETTING_KEYS
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)
from repositories.serialization.provider_model_catalog_cache import (
    ProviderModelCatalogCacheSerializer,
)
from services.llm import model_catalog
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
    # Active pipeline stage, used for an agent's per-stage model overrides.
    _stage: ContextVar[str | None] = ContextVar("llm_runtime_stage", default=None)

    # -------------------------------------------------------------------------
    @classmethod
    @contextmanager
    def stage_scope(cls, stage: str):
        token = cls._stage.set(stage)
        try:
            yield
        finally:
            cls._stage.reset(token)

    # -------------------------------------------------------------------------
    @classmethod
    def _overrides(cls) -> dict[str, object]:
        return dict(cls._runtime_override.get() or {})

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
    def _role_model(cls, role: RolePurpose) -> str:
        item = cls._role_models(cls._overrides()).get(role)
        return (cls._coerce_optional_text(item.get("model")) or "") if item else ""

    # -------------------------------------------------------------------------
    @classmethod
    def _cloud_role(cls) -> dict[str, object] | None:
        """First cloud-routed role of the active agent ("clinical" first)."""
        role_models = cls._role_models(cls._overrides())
        for role in ("clinical", *ROLE_PURPOSES):
            item = role_models.get(role)
            if item and str(item.get("provider") or "ollama") != "ollama":
                return item
        return None

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
                "agent_capability": context.capability,
                "agent_config_hash": context.config_hash,
            },
            "agent_instructions": config.instructions.model_dump(),
        }
        if isinstance(config, AssessmentConfig):
            models = config.models
            role_models["parser"] = entry(models.extraction)
            role_models["clinical"] = entry(models.clinical)
            # An assessment run neither revises nor builds timelines; keep
            # the role view complete with the clinical model.
            role_models["revision"] = entry(models.clinical)
            role_models["timeline"] = entry(models.clinical)
            for stage, ref in models.extraction_overrides.items():
                role_models[f"parser.{stage}"] = entry(ref)
            persisted = ModelConfigSerializer().load_snapshot().rag_settings or {}
            overrides["rag_settings"] = {
                **dict(persisted),
                **config.retrieval.rag_settings(),
            }
        elif isinstance(config, RevisionConfig):
            planning = entry(config.models.planning)
            for role in ROLE_PURPOSES:
                role_models[role] = planning
            for stage, ref in config.models.stage_overrides.items():
                role_models[f"revision.{stage}"] = entry(ref)
        elif isinstance(config, TimelineConfig):
            single = entry(config.model)
            for role in ROLE_PURPOSES:
                role_models[role] = single
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
    def with_agent_guidance(cls, prompt: str, *keys: str) -> str:
        """Append the active agent's bounded guidance fields to a user prompt."""
        return append_agent_guidance(prompt, cls.active_agent_instructions(), keys)

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_optional_text(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

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
    def get_text_extraction_model(cls) -> str:
        return cls._role_model("parser")

    # -------------------------------------------------------------------------
    @classmethod
    def get_clinical_model(cls) -> str:
        return cls._role_model("clinical")

    # -------------------------------------------------------------------------
    @classmethod
    def get_revision_model(cls) -> str:
        return cls._role_model("revision")

    # -------------------------------------------------------------------------
    @classmethod
    def get_timeline_model(cls) -> str:
        return cls._role_model("timeline")

    # -------------------------------------------------------------------------
    @classmethod
    def get_llm_provider(cls) -> str:
        item = cls._cloud_role()
        return str(item.get("provider") or "").strip() if item else ""

    # -------------------------------------------------------------------------
    @classmethod
    def get_cloud_model(cls) -> str:
        item = cls._cloud_role()
        return (cls._coerce_optional_text(item.get("model")) or "") if item else ""

    # -------------------------------------------------------------------------
    @classmethod
    def is_cloud_enabled(cls) -> bool:
        """True when at least one role of the active agent uses a cloud provider."""
        return cls._cloud_role() is not None

    # -------------------------------------------------------------------------
    @classmethod
    def get_reasoning_level(cls) -> ReasoningLevel:
        return cls._coerce_reasoning_level(cls._overrides().get("reasoning_level"))

    # -------------------------------------------------------------------------
    @classmethod
    def get_ollama_seed(cls) -> int | None:
        return cls._coerce_optional_int(cls._overrides().get("ollama_seed"))

    # -------------------------------------------------------------------------
    @classmethod
    def get_rag_settings(cls) -> dict[str, object]:
        """Persisted shared RAG settings overlaid by the active agent's settings."""
        value = cls._overrides().get("rag_settings")
        active = dict(cast(Mapping[str, object], value)) if isinstance(value, Mapping) else {}
        if set(SHARED_RAG_SETTING_KEYS) <= set(active):
            # Frozen at launch together with the shared settings.
            return active
        persisted = ModelConfigSerializer().load_snapshot().rag_settings or {}
        return {**persisted, **active}

    # -------------------------------------------------------------------------
    @classmethod
    def capture_run_snapshot(
        cls,
        *,
        evidence_sources: list[str],
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
        with cls.override_for_run(agent_overrides):
            snapshot, _ = cls._capture_run_snapshot(evidence_sources=evidence_sources)
        for key in ("role_models", "agent", "agent_instructions"):
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
            "rag_settings": cls.get_rag_settings(),
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
        updated_at = ModelConfigSerializer().load_snapshot().updated_at
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
        role_models = cls._role_models(cls._overrides())
        if not role_models:
            raise ValueError(f"No agent runtime is active for '{purpose}'.")
        stage = stage or cls._stage.get()
        item = (role_models.get(f"{purpose}.{stage}") if stage else None) or (
            role_models.get(purpose)
        )
        model = cls._coerce_optional_text(item.get("model")) if item else None
        if not item or not model:
            raise ValueError(f"Agent configuration has no model for '{purpose}'.")
        return str(item.get("provider") or "ollama"), model

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
