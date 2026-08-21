from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Any

from common.catalogs.model_choices import get_clinical_model_choices, get_text_extraction_model_choices
from common.paths import CATALOGS_PATH
from domain.model_configs import ReasoningLevel

###############################################################################
class GenerationPurpose(StrEnum):
    STRUCTURED_EXTRACTION = "structured_extraction"
    CLINICAL_SYNTHESIS = "clinical_synthesis"
    FAITHFUL_REWRITE = "faithful_rewrite"
    REVISION_SCAN = "revision_scan"
    REVISION_PLANNING = "revision_planning"
    REVISION_TOOL_SELECTION = "revision_tool_selection"
    REVISION_EDITING = "revision_editing"
    REVISION_QA = "revision_qa"
    TIMELINE_EXTRACTION = "timeline_extraction"
    JSON_REPAIR = "json_repair"
    CONNECTIVITY_CHECK = "connectivity_check"


TimelineComplexity = str

###############################################################################
class PolicyMatchKind(StrEnum):
    EXACT_MODEL = "exact_model"
    MODEL_FAMILY = "model_family"
    PROVIDER = "provider"
    FALLBACK = "fallback"

###############################################################################
@dataclass(frozen=True)
class GenerationPolicy:
    policy_id: str
    policy_version: str
    provider: str
    model: str
    purpose: GenerationPurpose
    user_reasoning_level: ReasoningLevel
    requested_reasoning_level: ReasoningLevel
    temperature: float | None
    visible_output_reserve: int
    reasoning_reserve: int
    output_token_limit: int
    context_safety_reserve: int
    context_strategy: str
    uses_model_default: bool
    match_kind: PolicyMatchKind
    rationale: str


_CATALOG_PATH = CATALOGS_PATH / "llm_generation_policies.json"
_LOCAL_CATALOG_PATH = CATALOGS_PATH / "local_models.json"

###############################################################################
def _load_catalog() -> dict[str, Any]:
    with _CATALOG_PATH.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("policy_version"), str):
        raise ValueError("Invalid LLM generation policy catalog")
    return catalog


_CATALOG = _load_catalog()

###############################################################################
_REASONING_OFF_PURPOSES = frozenset(
    {
        GenerationPurpose.JSON_REPAIR,
        GenerationPurpose.CONNECTIVITY_CHECK,
    }
)
_REVISION_PURPOSES = frozenset(
    {
        GenerationPurpose.REVISION_SCAN,
        GenerationPurpose.REVISION_PLANNING,
        GenerationPurpose.REVISION_TOOL_SELECTION,
        GenerationPurpose.REVISION_EDITING,
        GenerationPurpose.REVISION_QA,
    }
)


def _values(
    rule: dict[str, Any], purpose: GenerationPurpose, reasoning_level: ReasoningLevel
) -> float | None:
    if "profile" in rule:
        profile = _CATALOG["profiles"][rule["profile"]]
        return profile[purpose.value]
    if "all" in rule:
        return rule["all"]
    if purpose.value in rule:
        return rule[purpose.value]
    mode = "reasoning_enabled" if reasoning_level is not ReasoningLevel.OFF else "reasoning_disabled"
    selected = rule.get(mode, rule.get("all"))
    if selected is None:
        return None
    if isinstance(selected, dict) and "all" in selected:
        return selected["all"]
    return selected.get(purpose.value)


def _reasoning_target(
    *,
    purpose: GenerationPurpose,
    user_reasoning_level: ReasoningLevel,
    timeline_complexity: TimelineComplexity,
) -> ReasoningLevel:
    if purpose in _REASONING_OFF_PURPOSES:
        return ReasoningLevel.OFF
    if user_reasoning_level is ReasoningLevel.OFF:
        return ReasoningLevel.OFF
    if purpose in {
        GenerationPurpose.STRUCTURED_EXTRACTION,
        GenerationPurpose.FAITHFUL_REWRITE,
    }:
        return ReasoningLevel.LOW
    if purpose in _REVISION_PURPOSES:
        return ReasoningLevel.MEDIUM if user_reasoning_level is ReasoningLevel.HIGH else ReasoningLevel.LOW
    if purpose is GenerationPurpose.TIMELINE_EXTRACTION:
        if timeline_complexity == "complex" and user_reasoning_level is ReasoningLevel.HIGH:
            return ReasoningLevel.HIGH
        if timeline_complexity in {"moderate", "complex"} and user_reasoning_level in {
            ReasoningLevel.MEDIUM,
            ReasoningLevel.HIGH,
        }:
            return ReasoningLevel.MEDIUM
        return ReasoningLevel.LOW
    return user_reasoning_level


def _requirements(
    purpose: GenerationPurpose, reasoning_level: ReasoningLevel
) -> tuple[int, int, int, int, str]:
    requirements = {
        GenerationPurpose.CLINICAL_SYNTHESIS: (4096, 2048, 512, 256, "clinical_relevance"),
        GenerationPurpose.STRUCTURED_EXTRACTION: (1536, 0, 256, 128, "current_source"),
        GenerationPurpose.FAITHFUL_REWRITE: (2048, 0, 256, 128, "source_faithful"),
        GenerationPurpose.REVISION_SCAN: (1024, 0, 256, 128, "revision_priority"),
        GenerationPurpose.REVISION_PLANNING: (1536, 256, 256, 128, "revision_priority"),
        GenerationPurpose.REVISION_TOOL_SELECTION: (768, 0, 128, 96, "revision_priority"),
        GenerationPurpose.REVISION_EDITING: (3072, 512, 384, 192, "revision_priority"),
        GenerationPurpose.REVISION_QA: (1536, 0, 256, 128, "revision_priority"),
        GenerationPurpose.TIMELINE_EXTRACTION: (2048, 1024, 384, 192, "chronological"),
        GenerationPurpose.JSON_REPAIR: (768, 0, 128, 64, "repair_only"),
        GenerationPurpose.CONNECTIVITY_CHECK: (64, 0, 32, 32, "connectivity_only"),
    }
    visible_output, base_reasoning, safety, padding, strategy = requirements[purpose]
    reasoning_reserve = base_reasoning
    if reasoning_level is ReasoningLevel.OFF:
        reasoning_reserve = 0
    elif reasoning_level is ReasoningLevel.LOW:
        reasoning_reserve = max(128, base_reasoning // 2)
    elif reasoning_level is ReasoningLevel.MEDIUM:
        reasoning_reserve = max(256, base_reasoning)
    else:
        reasoning_reserve = max(512, base_reasoning * 2)
    return visible_output, reasoning_reserve, visible_output, safety + padding, strategy

###############################################################################
def _policy(
    *, provider: str, model: str, purpose: GenerationPurpose,
    user_reasoning_level: ReasoningLevel, requested_reasoning_level: ReasoningLevel,
    temperature: float | None,
    match_kind: PolicyMatchKind, rationale: str,
) -> GenerationPolicy:
    visible_output_reserve, reasoning_reserve, output_token_limit, context_safety_reserve, context_strategy = _requirements(
        purpose, requested_reasoning_level
    )
    return GenerationPolicy(
        policy_id=f"{match_kind.value}:{provider}:{model}:{purpose.value}",
        policy_version=_CATALOG["policy_version"],
        provider=provider,
        model=model,
        purpose=purpose,
        user_reasoning_level=user_reasoning_level,
        requested_reasoning_level=requested_reasoning_level,
        temperature=temperature,
        visible_output_reserve=visible_output_reserve,
        reasoning_reserve=reasoning_reserve,
        output_token_limit=output_token_limit,
        context_safety_reserve=context_safety_reserve,
        context_strategy=context_strategy,
        uses_model_default=temperature is None,
        match_kind=match_kind,
        rationale=rationale,
    )

###############################################################################
def resolve_generation_policy(
    *,
    purpose: GenerationPurpose,
    provider: str,
    model: str,
    user_reasoning_level: ReasoningLevel = ReasoningLevel.OFF,
    timeline_complexity: TimelineComplexity = "moderate",
    reasoning_enabled: bool | None = None,
) -> GenerationPolicy:
    if reasoning_enabled is not None:
        user_reasoning_level = (
            ReasoningLevel.MEDIUM if reasoning_enabled else ReasoningLevel.OFF
        )
    requested_reasoning_level = _reasoning_target(
        purpose=purpose,
        user_reasoning_level=user_reasoning_level,
        timeline_complexity=timeline_complexity,
    )
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip()
    exact = _CATALOG.get("exact_models", {}).get(f"{normalized_provider}:{normalized_model}")
    if exact is not None:
        temperature = _values(exact, purpose, requested_reasoning_level)
        return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                       user_reasoning_level=user_reasoning_level,
                       requested_reasoning_level=requested_reasoning_level,
                       temperature=temperature, match_kind=PolicyMatchKind.EXACT_MODEL,
                       rationale="Verified exact-model compatibility rule")
    family = normalized_model.split(":", 1)[0] if ":" in normalized_model else normalized_model
    family_rule = _CATALOG.get("families", {}).get(f"{normalized_provider}:{family}")
    if family_rule is None and normalized_provider == "openai" and normalized_model.startswith("gpt-5"):
        family_rule = _CATALOG.get("families", {}).get("openai:gpt-5")
    if family_rule is not None:
        temperature = _values(family_rule, purpose, requested_reasoning_level)
        return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                       user_reasoning_level=user_reasoning_level,
                       requested_reasoning_level=requested_reasoning_level,
                       temperature=temperature, match_kind=PolicyMatchKind.MODEL_FAMILY,
                       rationale="Verified model-family compatibility rule")
    local_names = set(get_clinical_model_choices()) | set(get_text_extraction_model_choices())
    if normalized_provider == "ollama" and normalized_model in local_names:
        profile_name = _CATALOG["local_models"]["default_profile"]
        temperature = _CATALOG["profiles"][profile_name][purpose.value]
        return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                       user_reasoning_level=user_reasoning_level,
                       requested_reasoning_level=requested_reasoning_level,
                       temperature=temperature, match_kind=PolicyMatchKind.FALLBACK,
                       rationale="Catalogued local instruction-model product default")
    provider_rule = _CATALOG.get("providers", {}).get(normalized_provider)
    if provider_rule is not None:
        known_cloud_model = normalized_provider != "openai" or normalized_model.startswith(("gpt-", "o1", "o3", "o4"))
        if known_cloud_model:
            temperature = _values(provider_rule, purpose, requested_reasoning_level)
            return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                           user_reasoning_level=user_reasoning_level,
                           requested_reasoning_level=requested_reasoning_level,
                           temperature=temperature, match_kind=PolicyMatchKind.PROVIDER,
                           rationale="Provider compatibility rule")
    return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                   user_reasoning_level=user_reasoning_level,
                   requested_reasoning_level=requested_reasoning_level,
                   temperature=None, match_kind=PolicyMatchKind.FALLBACK,
                   rationale="Unknown model uses the provider/model default")

###############################################################################
def validate_catalog() -> None:
    required = {purpose.value for purpose in GenerationPurpose}
    for name, profile in _CATALOG["profiles"].items():
        if set(profile) != required:
            raise ValueError(f"Profile {name} does not define every generation purpose")
    with _LOCAL_CATALOG_PATH.open(encoding="utf-8") as handle:
        local_models = json.load(handle)["local_model_catalog"]
    if not all(isinstance(item.get("name"), str) for item in local_models):
        raise ValueError("Local model catalog contains an invalid model name")


validate_catalog()
