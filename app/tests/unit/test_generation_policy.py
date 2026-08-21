from __future__ import annotations

import pytest

from services.llm.generation_policy import (
    GenerationPolicy,
    GenerationPurpose,
    resolve_generation_policy,
    validate_catalog,
)
from domain.model_configs import ReasoningLevel

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
