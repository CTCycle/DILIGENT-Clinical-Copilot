from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from common.paths import CATALOGS_PATH
from domain.llm.providers import (
    CloudModelDescriptor,
    ModelCapabilityMetadata,
    ReasoningToggleParameter,
    ToolCallMode,
)
from domain.model_configs import ReasoningLevel
from services.llm.generation_policy import GenerationPolicy

CapabilitySource = Literal[
    "exact_model",
    "model_family",
    "provider",
    "live",
    "documented",
    "probe",
    "fallback",
    "catalog",
]
ReasoningParameter = Literal[
    "none", "boolean", "level", "effort", "budget_tokens", "adaptive"
]
ReasoningToggle = ReasoningToggleParameter

###############################################################################
@dataclass(frozen=True)
class ModelCapabilities:
    input_token_limit: int | None
    output_token_limit: int | None
    supported_reasoning_levels: tuple[ReasoningLevel, ...]
    reasoning_parameter: ReasoningParameter
    reasoning_toggle_parameter: ReasoningToggle
    supports_temperature: bool
    supports_json_mode: bool
    supports_native_json_schema: bool
    source: CapabilitySource
    semantic_model_id: str | None = None
    provider_model_id: str | None = None
    aliases: tuple[str, ...] = ()
    endpoint_family: str | None = None
    supports_chat: bool | None = None
    supports_streaming: bool | None = None
    supports_tools: bool | None = None
    tool_call_mode: ToolCallMode = "unsupported"
    supports_tool_choice: bool | None = None
    supports_parallel_tool_calls: bool | None = None
    supports_top_p: bool | None = None
    supports_usage_metadata: bool | None = None
    supports_finish_reason: bool | None = None
    reasoning_counts_toward_output_limit: bool | None = None
    reasoning_unsupported_parameters: tuple[str, ...] = ()
    requires_reasoning_content_for_tool_calls: bool | None = None
    requires_assistant_content_for_tool_calls: bool | None = None

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
def _find_catalog_rule(
    provider: str, model: str
) -> tuple[dict[str, object], CapabilitySource]:
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
            if (
                normalized_model == family
                or normalized_model.startswith(f"{family}:")
                or normalized_model.startswith(family)
            ):
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
    except TypeError, ValueError:
        return None
    return parsed if parsed > 0 else None

###############################################################################
def _coerce_reasoning_levels(value: object) -> tuple[ReasoningLevel, ...]:
    if not isinstance(value, list | tuple):
        return (ReasoningLevel.OFF,)
    levels: list[ReasoningLevel] = []
    for item in value:
        try:
            level = (
                item if isinstance(item, ReasoningLevel) else ReasoningLevel(str(item))
            )
        except ValueError:
            continue
        if level not in levels:
            levels.append(level)
    return tuple(levels) or (ReasoningLevel.OFF,)

###############################################################################
def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values: tuple[object, ...] = (value,)
    elif isinstance(value, (list, tuple)):
        values = tuple(value)
    else:
        return ()
    return tuple(str(item) for item in values if str(item).strip())

###############################################################################
def _coerce_reasoning_parameter(value: object) -> ReasoningParameter:
    allowed: tuple[ReasoningParameter, ...] = (
        "none",
        "boolean",
        "level",
        "effort",
        "budget_tokens",
        "adaptive",
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
    metadata = descriptor.model_capabilities if descriptor is not None else None
    descriptor_has_metadata = descriptor is not None and (
        any(
            value is not None
            for value in (
                metadata.context_window_tokens if metadata else None,
                metadata.max_output_tokens if metadata else None,
                metadata.supports_reasoning if metadata else None,
                metadata.supports_temperature if metadata else None,
                metadata.supports_json_mode if metadata else None,
                metadata.supports_native_json_schema if metadata else None,
                metadata.supports_tools if metadata else None,
                metadata.supports_streaming if metadata else None,
                metadata.supports_tool_choice if metadata else None,
            )
        )
        or (metadata is not None and metadata.evidence != "fallback")
        or (descriptor is not None and descriptor.capabilities is not None)
    )

    def metadata_value(name: str) -> object:
        if metadata is None:
            return None
        return getattr(metadata, name)

    def rule_value(name: str, fallback_name: str | None = None) -> object:
        if name in rule:
            return rule[name]
        return fallback.get(fallback_name or name)

    input_token_limit = _coerce_optional_positive_int(
        metadata_value("context_window_tokens")
        if descriptor_has_metadata and metadata_value("context_window_tokens") is not None
        else rule_value("input_token_limit")
    )
    output_token_limit = _coerce_optional_positive_int(
        metadata_value("max_output_tokens")
        if descriptor_has_metadata and metadata_value("max_output_tokens") is not None
        else rule_value("output_token_limit")
    )
    levels = _coerce_reasoning_levels(
        metadata_value("reasoning_levels")
        if descriptor_has_metadata and metadata_value("reasoning_levels")
        else rule_value("supported_reasoning_levels")
    )
    if (
        descriptor_has_metadata
        and metadata_value("supports_reasoning") is not None
    ):
        levels = levels if metadata_value("supports_reasoning") else (ReasoningLevel.OFF,)
    supports_temperature = bool(
        metadata_value("supports_temperature")
        if descriptor_has_metadata and metadata_value("supports_temperature") is not None
        else rule_value("supports_temperature")
    )
    if descriptor_has_metadata and metadata_value("supports_json_mode") is not None:
        supports_json_mode = bool(metadata_value("supports_json_mode"))
    elif descriptor is not None and descriptor.capabilities is not None:
        supports_json_mode = bool(descriptor.capabilities.structured_output)
    else:
        supports_json_mode = bool(rule_value("supports_json_mode"))
    if descriptor_has_metadata and metadata_value("supports_native_json_schema") is not None:
        supports_native_json_schema = bool(metadata_value("supports_native_json_schema"))
    else:
        supports_native_json_schema = bool(rule_value("supports_native_json_schema"))
    reasoning_parameter = _coerce_reasoning_parameter(
        metadata_value("reasoning_parameter")
        if descriptor_has_metadata
        and metadata is not None
        and metadata_value("reasoning_parameter") != "none"
        else rule_value("reasoning_parameter")
    )
    reasoning_toggle_parameter = _coerce_reasoning_toggle(
        metadata_value("reasoning_toggle_parameter")
        if descriptor_has_metadata
        and metadata is not None
        and metadata_value("reasoning_toggle_parameter") != "provider_default"
        else rule_value("reasoning_toggle_parameter")
    )
    effective_source: CapabilitySource = (
        metadata.evidence
        if descriptor_has_metadata and metadata is not None and metadata.evidence != "fallback"
        else "live"
        if descriptor_has_metadata
        else source
    )
    merged_aliases = list(
        _coerce_string_tuple(
            metadata_value("aliases") if descriptor_has_metadata else ()
        )
    )
    merged_aliases.extend(_coerce_string_tuple(rule_value("aliases")))
    return ModelCapabilities(
        input_token_limit=input_token_limit,
        output_token_limit=output_token_limit,
        supported_reasoning_levels=levels,
        reasoning_parameter=reasoning_parameter,
        reasoning_toggle_parameter=reasoning_toggle_parameter,
        supports_temperature=supports_temperature,
        supports_json_mode=supports_json_mode,
        supports_native_json_schema=supports_native_json_schema,
        source=effective_source,
        semantic_model_id=(
            str(metadata_value("semantic_model_id"))
            if metadata_value("semantic_model_id")
            else str(rule_value("semantic_model_id"))
            if rule_value("semantic_model_id")
            else None
        ),
        provider_model_id=descriptor.id if descriptor is not None else model,
        aliases=tuple(
            dict.fromkeys(str(item) for item in merged_aliases if str(item).strip())
        ),
        endpoint_family=(
            str(metadata_value("endpoint_family"))
            if descriptor_has_metadata and metadata_value("endpoint_family")
            else str(rule_value("endpoint_family"))
            if rule_value("endpoint_family")
            else None
        ),
        supports_chat=(
            bool(metadata_value("supports_chat"))
            if descriptor_has_metadata and metadata_value("supports_chat") is not None
            else _optional_bool(rule_value("supports_chat"))
        ),
        supports_streaming=(
            bool(metadata_value("supports_streaming"))
            if descriptor_has_metadata and metadata_value("supports_streaming") is not None
            else _optional_bool(rule_value("supports_streaming"))
        ),
        supports_tools=(
            bool(metadata_value("supports_tools"))
            if descriptor_has_metadata and metadata_value("supports_tools") is not None
            else _optional_bool(rule_value("supports_tools"))
        ),
        tool_call_mode=(
            metadata.tool_call_mode
            if descriptor_has_metadata
            and metadata is not None
            and metadata.tool_call_mode != "unsupported"
            else _coerce_tool_call_mode(rule_value("tool_call_mode"))
        ),
        supports_tool_choice=(
            metadata.supports_tool_choice
            if descriptor_has_metadata
            and metadata is not None
            and metadata.supports_tool_choice is not None
            else _optional_bool(rule_value("supports_tool_choice"))
        ),
        supports_parallel_tool_calls=(
            metadata.supports_parallel_tool_calls
            if descriptor_has_metadata and metadata is not None and metadata.supports_parallel_tool_calls is not None
            else _optional_bool(rule_value("supports_parallel_tool_calls"))
        ),
        supports_top_p=(
            metadata.supports_top_p
            if descriptor_has_metadata and metadata is not None and metadata.supports_top_p is not None
            else _optional_bool(rule_value("supports_top_p"))
        ),
        supports_usage_metadata=(
            metadata.supports_usage_metadata
            if descriptor_has_metadata and metadata is not None and metadata.supports_usage_metadata is not None
            else _optional_bool(rule_value("supports_usage_metadata"))
        ),
        supports_finish_reason=(
            metadata.supports_finish_reason
            if descriptor_has_metadata and metadata is not None and metadata.supports_finish_reason is not None
            else _optional_bool(rule_value("supports_finish_reason"))
        ),
        reasoning_counts_toward_output_limit=(
            metadata.reasoning_counts_toward_output_limit
            if descriptor_has_metadata and metadata is not None and metadata.reasoning_counts_toward_output_limit is not None
            else _optional_bool(rule_value("reasoning_counts_toward_output_limit"))
        ),
        reasoning_unsupported_parameters=_coerce_string_tuple(
            metadata.reasoning_unsupported_parameters
            if descriptor_has_metadata
            and metadata is not None
            and metadata.reasoning_unsupported_parameters
            else rule_value("reasoning_unsupported_parameters")
        ),
        requires_reasoning_content_for_tool_calls=(
            metadata.requires_reasoning_content_for_tool_calls
            if descriptor_has_metadata
            and metadata is not None
            and metadata.requires_reasoning_content_for_tool_calls is not None
            else _optional_bool(rule_value("requires_reasoning_content_for_tool_calls"))
        ),
        requires_assistant_content_for_tool_calls=(
            metadata.requires_assistant_content_for_tool_calls
            if descriptor_has_metadata
            and metadata is not None
            and metadata.requires_assistant_content_for_tool_calls is not None
            else _optional_bool(rule_value("requires_assistant_content_for_tool_calls"))
        ),
    )

###############################################################################
def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "supported"}:
        return True
    if normalized in {"false", "no", "0", "unsupported"}:
        return False
    return None

###############################################################################
def _coerce_tool_call_mode(value: object) -> ToolCallMode:
    normalized = str(value or "unsupported").strip().lower()
    return normalized if normalized in {"native", "structured", "unsupported"} else "unsupported"  # type: ignore[return-value]

###############################################################################
def _coerce_reasoning_toggle(value: object) -> ReasoningToggle:
    normalized = str(value or "provider_default").strip().lower()
    allowed = {"none", "thinking", "reasoning", "provider_default"}
    return normalized if normalized in allowed else "provider_default"  # type: ignore[return-value]

###############################################################################
def capability_metadata(
    *, provider: str, model: str, descriptor: CloudModelDescriptor | None = None
) -> ModelCapabilityMetadata:
    """Return the transport-facing canonical metadata for a selected model."""

    resolved = resolve_model_capabilities(
        provider=provider,
        model=model,
        descriptor=descriptor,
    )
    return ModelCapabilityMetadata(
        semantic_model_id=resolved.semantic_model_id,
        aliases=resolved.aliases,
        endpoint_family=resolved.endpoint_family,
        context_window_tokens=resolved.input_token_limit,
        max_output_tokens=resolved.output_token_limit,
        supports_chat=resolved.supports_chat,
        supports_streaming=resolved.supports_streaming,
        supports_tools=resolved.supports_tools,
        tool_call_mode=resolved.tool_call_mode,
        supports_tool_choice=resolved.supports_tool_choice,
        supports_parallel_tool_calls=resolved.supports_parallel_tool_calls,
        supports_structured_output=resolved.supports_json_mode,
        supports_json_mode=resolved.supports_json_mode,
        supports_native_json_schema=resolved.supports_native_json_schema,
        supports_reasoning=bool(
            resolved.supported_reasoning_levels
            and resolved.supported_reasoning_levels != (ReasoningLevel.OFF,)
        ),
        reasoning_levels=tuple(level.value for level in resolved.supported_reasoning_levels),
        reasoning_parameter=resolved.reasoning_parameter,
        reasoning_toggle_parameter=resolved.reasoning_toggle_parameter,
        reasoning_counts_toward_output_limit=resolved.reasoning_counts_toward_output_limit,
        reasoning_unsupported_parameters=resolved.reasoning_unsupported_parameters,
        requires_reasoning_content_for_tool_calls=(
            resolved.requires_reasoning_content_for_tool_calls
        ),
        requires_assistant_content_for_tool_calls=(
            resolved.requires_assistant_content_for_tool_calls
        ),
        supports_temperature=resolved.supports_temperature,
        supports_top_p=resolved.supports_top_p,
        supports_usage_metadata=resolved.supports_usage_metadata,
        supports_finish_reason=resolved.supports_finish_reason,
        evidence=(
            resolved.source
            if resolved.source == "documented"
            or resolved.source == "probe"
            or resolved.source == "provider"
            or resolved.source == "fallback"
            else "catalog"
        ),
    )

###############################################################################
def catalog_capability_metadata(
    raw_item: object, *, endpoint_family: str | None = None
) -> ModelCapabilityMetadata:
    """Parse common provider catalog capability keys into the shared contract."""

    if not isinstance(raw_item, Mapping):
        return ModelCapabilityMetadata(endpoint_family=endpoint_family)
    item = dict(raw_item)
    nested = item.get("capabilities")
    capabilities = dict(nested) if isinstance(nested, Mapping) else {}

    def pick(*names: str) -> object:
        for name in names:
            if name in item and item[name] is not None:
                return item[name]
            if name in capabilities and capabilities[name] is not None:
                return capabilities[name]
        return None

    def optional_bool(*names: str) -> bool | None:
        value = pick(*names)
        return _optional_bool(value)

    values: dict[str, object] = {}
    string_fields = {
        "semantic_model_id": ("semantic_model_id", "canonical_model", "canonical_id"),
        "endpoint_family": ("endpoint_family", "endpoint"),
        "reasoning_parameter": ("reasoning_parameter",),
        "reasoning_toggle_parameter": (
            "reasoning_toggle_parameter",
            "reasoning_toggle",
            "thinking_parameter",
        ),
        "tool_call_mode": ("tool_call_mode",),
    }
    for target, names in string_fields.items():
        value = (
            endpoint_family
            if target == "endpoint_family" and endpoint_family is not None
            else pick(*names)
        )
        if value is not None and not isinstance(value, (dict, list, tuple)):
            values[target] = str(value)

    aliases = pick("aliases", "model_aliases")
    if isinstance(aliases, str):
        values["aliases"] = (aliases,)
    elif isinstance(aliases, (list, tuple)):
        values["aliases"] = tuple(str(item) for item in aliases if str(item).strip())

    reasoning_levels = pick("reasoning_levels", "supported_reasoning_levels")
    if isinstance(reasoning_levels, (list, tuple)):
        values["reasoning_levels"] = tuple(
            str(level) for level in reasoning_levels if str(level).strip()
        )

    integer_fields = {
        "context_window_tokens": (
            "context_window_tokens",
            "context_window",
            "input_token_limit",
            "max_input_tokens",
        ),
        "max_output_tokens": (
            "max_output_tokens",
            "output_token_limit",
        ),
    }
    for target, names in integer_fields.items():
        value = pick(*names)
        parsed = _coerce_optional_positive_int(value)
        if parsed is not None:
            values[target] = parsed

    bool_fields = {
        "supports_chat": ("supports_chat", "chat"),
        "supports_streaming": ("supports_streaming", "streaming"),
        "supports_tools": ("supports_tools", "tools", "tool_calling"),
        "supports_tool_choice": ("supports_tool_choice", "tool_choice"),
        "supports_parallel_tool_calls": (
            "supports_parallel_tool_calls",
            "parallel_tool_calls",
        ),
        "supports_structured_output": (
            "supports_structured_output",
            "structured_output",
            "structured_outputs",
        ),
        "supports_json_mode": ("supports_json_mode", "json_mode"),
        "supports_native_json_schema": (
            "supports_native_json_schema",
            "native_json_schema",
            "json_schema",
        ),
        "supports_reasoning": ("supports_reasoning", "reasoning", "thinking"),
        "supports_temperature": ("supports_temperature", "temperature"),
        "supports_top_p": ("supports_top_p", "top_p"),
        "supports_usage_metadata": ("supports_usage_metadata", "usage"),
        "supports_finish_reason": ("supports_finish_reason", "finish_reason"),
        "reasoning_counts_toward_output_limit": (
            "reasoning_counts_toward_output_limit",
            "reasoning_counts_toward_output",
        ),
        "requires_reasoning_content_for_tool_calls": (
            "requires_reasoning_content_for_tool_calls",
            "requires_reasoning_content",
        ),
        "requires_assistant_content_for_tool_calls": (
            "requires_assistant_content_for_tool_calls",
            "requires_assistant_content",
        ),
    }
    for target, names in bool_fields.items():
        parsed = optional_bool(*names)
        if parsed is not None:
            values[target] = parsed

    unsupported = pick(
        "reasoning_unsupported_parameters", "unsupported_reasoning_parameters"
    )
    if isinstance(unsupported, (list, tuple)):
        values["reasoning_unsupported_parameters"] = tuple(
            str(parameter) for parameter in unsupported if str(parameter).strip()
        )
    if "tool_call_mode" not in values and values.get("supports_tools") is True:
        values["tool_call_mode"] = "native"
    if values:
        values["evidence"] = "provider"
    return ModelCapabilityMetadata.model_validate(values)

###############################################################################
def enrich_model_descriptor(
    *, provider: str, descriptor: CloudModelDescriptor
) -> CloudModelDescriptor:
    """Merge live descriptor data with the canonical static model contract."""

    metadata = capability_metadata(
        provider=provider,
        model=descriptor.id,
        descriptor=descriptor,
    )
    return CloudModelDescriptor.model_validate(
        {
            **descriptor.model_dump(mode="python"),
            "model_capabilities": metadata.model_dump(mode="python"),
            "endpoint_family": descriptor.endpoint_family or metadata.endpoint_family,
            "input_token_limit": descriptor.input_token_limit or metadata.context_window_tokens,
            "output_token_limit": descriptor.output_token_limit or metadata.max_output_tokens,
            "supports_thinking": descriptor.supports_thinking
            if descriptor.supports_thinking is not None
            else metadata.supports_reasoning,
            "supports_temperature": descriptor.supports_temperature
            if descriptor.supports_temperature is not None
            else metadata.supports_temperature,
            "supports_json_mode": descriptor.supports_json_mode
            if descriptor.supports_json_mode is not None
            else metadata.supports_json_mode,
            "supports_native_json_schema": descriptor.supports_native_json_schema
            if descriptor.supports_native_json_schema is not None
            else metadata.supports_native_json_schema,
        }
    )

###############################################################################
def resolve_endpoint_family(*, provider: str, model: str) -> str | None:
    """Resolve a cataloged transport family without creating a live request."""

    return resolve_model_capabilities(provider=provider, model=model).endpoint_family

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
    return min(
        supported, key=lambda level: (abs(rank[level] - rank[requested]), rank[level])
    )

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
