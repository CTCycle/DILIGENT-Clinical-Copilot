from __future__ import annotations

import pytest

from services.llm.generation_policy import (
    GenerationPolicy,
    GenerationPurpose,
    resolve_generation_policy,
    validate_catalog,
)
from domain.model_configs import ReasoningLevel
from services.llm.model_capabilities import (
    resolve_effective_inference_config,
    resolve_model_capabilities,
)

###############################################################################
@pytest.mark.parametrize(
    ("model", "reasoning_level", "expected"),
    [
        ("qwen3:8b", ReasoningLevel.MEDIUM, 0.2),
        ("qwen3:8b", ReasoningLevel.OFF, 0.2),
        ("qwen3.5:2b", ReasoningLevel.OFF, 0.2),
        ("qwen3.5:9b", ReasoningLevel.HIGH, 0.2),
        ("qwen3.5:9b", ReasoningLevel.OFF, 0.2),
        ("deepseek-r1:14b", ReasoningLevel.OFF, 0.2),
        ("phi4-reasoning:14b", ReasoningLevel.OFF, 0.2),
    ],
)
def test_local_policy_matrix(
    model: str, reasoning_level: ReasoningLevel, expected: float
) -> None:
    policy = resolve_generation_policy(
        purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
        provider="ollama",
        model=model,
        user_reasoning_level=reasoning_level,
    )
    assert policy.temperature == expected

###############################################################################
def test_known_provider_defaults_use_base_temperature() -> None:
    for provider, model in (("anthropic", "claude-opus-4-6"), ("gemini", "gemini-3-pro"), ("ollama", "new-model:1b")):
        policy = resolve_generation_policy(
            purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
            provider=provider,
            model=model,
        )
        assert policy.temperature == 0.0
        assert not policy.uses_model_default

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
    ).temperature == 0.2
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
        assert policy.temperature == 0.2
        effective = resolve_effective_inference_config(
            policy=policy,
            capabilities=resolve_model_capabilities(provider=provider, model=model),
        )
        assert effective.temperature is None

###############################################################################
def test_policy_is_immutable_and_catalog_validates() -> None:
    validate_catalog()
    policy = resolve_generation_policy(
        purpose=GenerationPurpose.JSON_REPAIR,
        provider="ollama",
        model="qwen3:8b",
        user_reasoning_level=ReasoningLevel.HIGH,
    )
    assert isinstance(policy, GenerationPolicy)
    with pytest.raises(AttributeError):
        policy.temperature = 0.9  # type: ignore[misc]

###############################################################################
@pytest.mark.parametrize(
    ("purpose", "complexity", "expected"),
    [
        (GenerationPurpose.CLINICAL_SYNTHESIS, "moderate", ["off", "low", "medium", "high"]),
        (GenerationPurpose.STRUCTURED_EXTRACTION, "moderate", ["off", "low", "low", "low"]),
        (GenerationPurpose.FAITHFUL_REWRITE, "moderate", ["off", "low", "low", "low"]),
        (GenerationPurpose.REVISION_SCAN, "moderate", ["off", "low", "low", "medium"]),
        (GenerationPurpose.REVISION_EDITING, "moderate", ["off", "low", "low", "medium"]),
        (GenerationPurpose.JSON_REPAIR, "moderate", ["off", "off", "off", "off"]),
        (GenerationPurpose.CONNECTIVITY_CHECK, "moderate", ["off", "off", "off", "off"]),
        (GenerationPurpose.TIMELINE_EXTRACTION, "simple", ["off", "low", "low", "low"]),
        (GenerationPurpose.TIMELINE_EXTRACTION, "moderate", ["off", "low", "medium", "medium"]),
        (GenerationPurpose.TIMELINE_EXTRACTION, "complex", ["off", "low", "medium", "high"]),
    ],
)
def test_responsibility_reasoning_matrix(
    purpose: GenerationPurpose,
    complexity: str,
    expected: list[str],
) -> None:
    levels = [ReasoningLevel.OFF, ReasoningLevel.LOW, ReasoningLevel.MEDIUM, ReasoningLevel.HIGH]

    actual = [
        resolve_generation_policy(
            purpose=purpose,
            provider="openai",
            model="gpt-4.1-mini",
            user_reasoning_level=level,
            timeline_complexity=complexity,
        ).requested_reasoning_level.value
        for level in levels
    ]

    assert actual == expected

###############################################################################
def test_reasoning_target_is_monotonic_for_each_responsibility() -> None:
    levels = [ReasoningLevel.OFF, ReasoningLevel.LOW, ReasoningLevel.MEDIUM, ReasoningLevel.HIGH]
    rank = {level.value: index for index, level in enumerate(levels)}
    for purpose in GenerationPurpose:
        values = [
            rank[
                resolve_generation_policy(
                    purpose=purpose,
                    provider="openai",
                    model="gpt-4.1-mini",
                    user_reasoning_level=level,
                    timeline_complexity="complex",
                ).requested_reasoning_level.value
            ]
            for level in levels
        ]
        assert values == sorted(values)
