from __future__ import annotations

import pytest

from services.llm.generation_policy import (
    GenerationPolicy,
    GenerationPurpose,
    resolve_generation_policy,
    validate_catalog,
)

###############################################################################
@pytest.mark.parametrize(
    ("model", "reasoning", "expected"),
    [
        ("qwen3:8b", True, 0.6),
        ("qwen3:8b", False, 0.7),
        ("qwen3.5:2b", False, 1.0),
        ("qwen3.5:9b", True, 1.0),
        ("qwen3.5:9b", False, 0.7),
        ("deepseek-r1:14b", False, 0.6),
        ("phi4-reasoning:14b", False, 0.8),
    ],
)
def test_local_policy_matrix(model: str, reasoning: bool, expected: float) -> None:
    policy = resolve_generation_policy(
        purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
        provider="ollama",
        model=model,
        reasoning_enabled=reasoning,
    )
    assert policy.temperature == expected

###############################################################################
def test_restricted_and_unknown_models_use_defaults() -> None:
    for provider, model in (("anthropic", "claude-opus-4-6"), ("gemini", "gemini-3-pro"), ("ollama", "new-model:1b")):
        policy = resolve_generation_policy(
            purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
            provider=provider,
            model=model,
        )
        assert policy.temperature is None
        assert policy.uses_model_default

###############################################################################
def test_openai_and_deepseek_are_purpose_specific() -> None:
    assert resolve_generation_policy(
        purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
        provider="openai",
        model="gpt-4.1-mini",
    ).temperature == 0.0
    assert resolve_generation_policy(
        purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
        provider="openai",
        model="gpt-4.1-mini",
    ).temperature == 0.2
    assert resolve_generation_policy(
        purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
        provider="deepseek",
        model="deepseek-chat",
    ).temperature == 1.0
    assert resolve_generation_policy(
        purpose=GenerationPurpose.CONNECTIVITY_CHECK,
        provider="openai",
        model="gpt-4.1-mini",
    ).temperature is None

###############################################################################
def test_gpt5_and_gpt_oss_omit_temperature() -> None:
    for provider, model in (("openai", "gpt-5"), ("openai", "gpt-5.1"), ("ollama", "gpt-oss:20b")):
        policy = resolve_generation_policy(
            purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
            provider=provider,
            model=model,
        )
        assert policy.temperature is None

###############################################################################
def test_policy_is_immutable_and_catalog_validates() -> None:
    validate_catalog()
    policy = resolve_generation_policy(
        purpose=GenerationPurpose.JSON_REPAIR,
        provider="ollama",
        model="qwen3:8b",
        reasoning_enabled=True,
    )
    assert isinstance(policy, GenerationPolicy)
    with pytest.raises(AttributeError):
        policy.temperature = 0.9  # type: ignore[misc]
