from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Literal, Mapping

from common.paths import CATALOGS_PATH
from domain.llm.providers import CloudModelDescriptor
from domain.model_configs import ReasoningLevel
from services.llm.generation_policy import GenerationPolicy

CapabilitySource = Literal["exact_model", "model_family", "provider", "live", "fallback"]
ReasoningParameter = Literal["none", "boolean", "level", "effort", "budget_tokens"]

###############################################################################
@dataclass(frozen=True)
class ModelCapabilities:
    input_token_limit: int | None
    output_token_limit: int | None
    supported_reasoning_levels: tuple[ReasoningLevel, ...]
    reasoning_parameter: ReasoningParameter
    supports_temperature: bool
    supports_json_mode: bool
    source: CapabilitySource

###############################################################################
@dataclass(frozen=True)
class EffectiveInferenceConfig:
    policy_version: str
    policy_id: str
    policy_match_source: str
    purpose: str
    provider: str
    model: str
    user_reasoning_level: ReasoningLevel
    requested_reasoning_level: ReasoningLevel
    effective_reasoning_level: ReasoningLevel
    reasoning_adjustment_reason: str | None
    reasoning_parameter: ReasoningParameter
    temperature: float | None
    model_context_limit: int | None
    effective_runtime_context_limit: int | None
    input_budget: int | None
    visible_output_reserve: int
    reasoning_reserve: int
    output_token_limit: int
    context_safety_reserve: int
    capability_source: CapabilitySource
    context_selection_report: Mapping[str, object]


_CATALOG_PATH = CATALOGS_PATH / "llm_model_capabilities.json"

###############################################################################
def _load_catalog() -> dict[str, object]:
    with _CATALOG_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("capability_version"), str
    ):
        raise ValueError("Invalid LLM model capability catalog")
    return payload


_CATALOG = _load_catalog()

###############################################################################
def _fallback_rule() -> dict[str, object]:
    fallback = _CATALOG.get("fallback")
    return fallback if isinstance(fallback, dict) else {}

###############################################################################
def _find_catalog_rule(provider: str, model: str) -> tuple[dict[str, object], CapabilitySource]:
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip()
    exact_models = _CATALOG.get("exact_models")
    if isinstance(exact_models, dict):
        exact = exact_models.get(f"{normalized_provider}:{normalized_model}")
        if isinstance(exact, dict):
            return exact, "exact_model"

    families = _CATALOG.get("families")
    if isinstance(families, dict):
        matching: list[tuple[int, dict[str, object]]] = []
        for key, raw_rule in families.items():
            if not isinstance(key, str) or not isinstance(raw_rule, dict):
                continue
            prefix = f"{normalized_provider}:"
            if not key.startswith(prefix):
                continue
            family = key.removeprefix(prefix)
            if normalized_model == family or normalized_model.startswith(f"{family}:") or normalized_model.startswith(family):
                matching.append((len(family), raw_rule))
        if matching:
            return max(matching, key=lambda item: item[0])[1], "model_family"

    providers = _CATALOG.get("providers")
    if isinstance(providers, dict):
        provider_rule = providers.get(normalized_provider)
        if isinstance(provider_rule, dict):
            return provider_rule, "provider"
    return _fallback_rule(), "fallback"

###############################################################################
def _coerce_optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None

###############################################################################
def _coerce_reasoning_levels(value: object) -> tuple[ReasoningLevel, ...]:
    if not isinstance(value, list | tuple):
        return (ReasoningLevel.OFF,)
    levels: list[ReasoningLevel] = []
    for item in value:
        try:
            level = item if isinstance(item, ReasoningLevel) else ReasoningLevel(str(item))
        except ValueError:
            continue
        if level not in levels:
            levels.append(level)
    return tuple(levels) or (ReasoningLevel.OFF,)

###############################################################################
def _coerce_reasoning_parameter(value: object) -> ReasoningParameter:
    allowed: tuple[ReasoningParameter, ...] = (
        "none",
        "boolean",
        "level",
        "effort",
        "budget_tokens",
    )
    normalized = str(value or "none")
    return normalized if normalized in allowed else "none"  # type: ignore[return-value]

###############################################################################
def resolve_model_capabilities(
    *,
    provider: str,
    model: str,
    descriptor: CloudModelDescriptor | None = None,
) -> ModelCapabilities:
    rule, source = _find_catalog_rule(provider, model)
    fallback = _fallback_rule()
    descriptor_has_metadata = descriptor is not None and any(
        value is not None
        for value in (
            descriptor.input_token_limit,
            descriptor.output_token_limit,
            descriptor.supports_thinking,
            descriptor.supports_temperature,
        )
    )

    input_token_limit = _coerce_optional_positive_int(
        descriptor.input_token_limit
        if descriptor_has_metadata and descriptor is not None and descriptor.input_token_limit is not None
        else rule.get("input_token_limit", fallback.get("input_token_limit"))
    )
    output_token_limit = _coerce_optional_positive_int(
        descriptor.output_token_limit
        if descriptor_has_metadata and descriptor is not None and descriptor.output_token_limit is not None
        else rule.get("output_token_limit", fallback.get("output_token_limit"))
    )
    levels = _coerce_reasoning_levels(
        rule.get("supported_reasoning_levels", fallback.get("supported_reasoning_levels"))
    )
    if descriptor_has_metadata and descriptor is not None and descriptor.supports_thinking is not None:
        levels = levels if descriptor.supports_thinking else (ReasoningLevel.OFF,)
    supports_temperature = bool(
        descriptor.supports_temperature
        if descriptor_has_metadata and descriptor is not None and descriptor.supports_temperature is not None
        else rule.get("supports_temperature", fallback.get("supports_temperature", False))
    )
    supports_json_mode = bool(
        rule.get("supports_json_mode", fallback.get("supports_json_mode", False))
    )
    return ModelCapabilities(
        input_token_limit=input_token_limit,
        output_token_limit=output_token_limit,
        supported_reasoning_levels=levels,
        reasoning_parameter=_coerce_reasoning_parameter(
            rule.get("reasoning_parameter", fallback.get("reasoning_parameter"))
        ),
        supports_temperature=supports_temperature,
        supports_json_mode=supports_json_mode,
        source="live" if descriptor_has_metadata else source,
    )

###############################################################################
def _select_supported_reasoning_level(
    requested: ReasoningLevel, supported: tuple[ReasoningLevel, ...]
) -> ReasoningLevel:
    if requested in supported:
        return requested
    rank = {
        ReasoningLevel.OFF: 0,
        ReasoningLevel.LOW: 1,
        ReasoningLevel.MEDIUM: 2,
        ReasoningLevel.HIGH: 3,
    }
    return min(supported, key=lambda level: (abs(rank[level] - rank[requested]), rank[level]))

###############################################################################
def resolve_effective_inference_config(
    *,
    policy: GenerationPolicy,
    capabilities: ModelCapabilities,
    runtime_context_limit: int | None = None,
    selected_input_tokens: int = 0,
) -> EffectiveInferenceConfig:
    effective_reasoning_level = _select_supported_reasoning_level(
        policy.requested_reasoning_level,
        capabilities.supported_reasoning_levels,
    )
    adjustment_reason = None
    if effective_reasoning_level is not policy.requested_reasoning_level:
        adjustment_reason = (
            f"Requested {policy.requested_reasoning_level.value} is not supported; "
            f"using {effective_reasoning_level.value}."
        )
    reasoning_reserve = (
        policy.reasoning_reserve
        if effective_reasoning_level is not ReasoningLevel.OFF
        else 0
    )
    context_limits = [
        limit
        for limit in (capabilities.input_token_limit, runtime_context_limit)
        if limit is not None and limit > 0
    ]
    effective_context_limit = min(context_limits) if context_limits else None
    reserved_tokens = (
        policy.visible_output_reserve
        + reasoning_reserve
        + policy.context_safety_reserve
    )
    input_budget = (
        max(0, effective_context_limit - reserved_tokens)
        if effective_context_limit is not None
        else None
    )
    output_token_limit = (
        min(policy.output_token_limit, capabilities.output_token_limit)
        if capabilities.output_token_limit is not None
        else policy.output_token_limit
    )
    effective_temperature = (
        policy.temperature
        if capabilities.supports_temperature
        and (
            effective_reasoning_level is ReasoningLevel.OFF
            or capabilities.reasoning_parameter == "none"
        )
        else None
    )
    report = MappingProxyType(
        {
            "capacity_known": effective_context_limit is not None,
            "selected_input_tokens": max(0, int(selected_input_tokens)),
            "input_budget": input_budget,
            "overflow_tokens": (
                max(0, int(selected_input_tokens) - input_budget)
                if input_budget is not None
                else 0
            ),
            "reserved_tokens": reserved_tokens,
        }
    )
    return EffectiveInferenceConfig(
        policy_version=policy.policy_version,
        policy_id=policy.policy_id,
        policy_match_source=policy.match_kind.value,
        purpose=policy.purpose.value,
        provider=policy.provider,
        model=policy.model,
        user_reasoning_level=policy.user_reasoning_level,
        requested_reasoning_level=policy.requested_reasoning_level,
        effective_reasoning_level=effective_reasoning_level,
        reasoning_adjustment_reason=adjustment_reason,
        reasoning_parameter=capabilities.reasoning_parameter,
        temperature=effective_temperature,
        model_context_limit=capabilities.input_token_limit,
        effective_runtime_context_limit=effective_context_limit,
        input_budget=input_budget,
        visible_output_reserve=policy.visible_output_reserve,
        reasoning_reserve=reasoning_reserve,
        output_token_limit=output_token_limit,
        context_safety_reserve=policy.context_safety_reserve,
        capability_source=capabilities.source,
        context_selection_report=report,
    )
