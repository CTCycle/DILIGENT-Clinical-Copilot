from __future__ import annotations

from domain.llm.providers import CloudModelDescriptor
from domain.model_configs import ReasoningLevel
from services.llm.generation_policy import (
    GenerationPurpose,
    resolve_generation_policy,
)
from services.llm.model_capabilities import (
    resolve_effective_inference_config,
    resolve_model_capabilities,
)

###############################################################################
def test_capability_resolution_prefers_exact_then_family_then_provider() -> None:
    exact = resolve_model_capabilities(provider="ollama", model="gpt-oss:20b")
    family = resolve_model_capabilities(provider="ollama", model="gpt-oss:120b")
    provider = resolve_model_capabilities(provider="ollama", model="new-model:1b")

    assert exact.source == "exact_model"
    assert exact.supported_reasoning_levels == (
        ReasoningLevel.LOW,
        ReasoningLevel.MEDIUM,
        ReasoningLevel.HIGH,
    )
    assert family.source == "model_family"
    assert provider.source == "provider"

###############################################################################
def test_unknown_model_uses_conservative_fallback_without_assumed_capacity() -> None:
    capabilities = resolve_model_capabilities(
        provider="unknown-provider",
        model="unknown-model",
    )

    assert capabilities.source == "fallback"
    assert capabilities.input_token_limit is None
    assert capabilities.output_token_limit is None
    assert capabilities.supported_reasoning_levels == (ReasoningLevel.OFF,)
    assert capabilities.supports_temperature is False

###############################################################################
def test_live_descriptor_metadata_overrides_catalog_limits() -> None:
    capabilities = resolve_model_capabilities(
        provider="gemini",
        model="gemini-live-model",
        descriptor=CloudModelDescriptor(
            id="gemini-live-model",
            display_name="Gemini live",
            input_token_limit=32768,
            output_token_limit=4096,
            supports_thinking=False,
            supports_temperature=False,
        ),
    )

    assert capabilities.source == "live"
    assert capabilities.input_token_limit == 32768
    assert capabilities.output_token_limit == 4096
    assert capabilities.supported_reasoning_levels == (ReasoningLevel.OFF,)
    assert capabilities.supports_temperature is False

###############################################################################
def test_effective_config_preserves_request_and_reports_provider_coercion() -> None:
    policy = resolve_generation_policy(
        purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
        provider="ollama",
        model="gpt-oss:20b",
        user_reasoning_level=ReasoningLevel.OFF,
    )
    capabilities = resolve_model_capabilities(provider="ollama", model="gpt-oss:20b")

    effective = resolve_effective_inference_config(
        policy=policy,
        capabilities=capabilities,
        runtime_context_limit=8192,
        selected_input_tokens=1000,
    )

    assert effective.user_reasoning_level is ReasoningLevel.OFF
    assert effective.requested_reasoning_level is ReasoningLevel.OFF
    assert effective.effective_reasoning_level is ReasoningLevel.LOW
    assert effective.reasoning_adjustment_reason is not None
    assert effective.temperature is None
    assert effective.effective_runtime_context_limit == 8192
    assert effective.input_budget is not None
    assert effective.context_selection_report["selected_input_tokens"] == 1000

###############################################################################
def test_effective_config_never_steals_reserved_generation_capacity() -> None:
    policy = resolve_generation_policy(
        purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
        provider="openai",
        model="gpt-4.1-mini",
        user_reasoning_level=ReasoningLevel.HIGH,
    )
    capabilities = resolve_model_capabilities(provider="openai", model="gpt-4.1-mini")

    effective = resolve_effective_inference_config(
        policy=policy,
        capabilities=capabilities,
        runtime_context_limit=4096,
        selected_input_tokens=4096,
    )

    assert effective.input_budget == 0
    assert effective.context_selection_report["overflow_tokens"] == 4096
    assert effective.output_token_limit == policy.output_token_limit
