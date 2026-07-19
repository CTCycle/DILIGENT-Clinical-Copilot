from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Any

from common.catalogs.model_choices import get_clinical_model_choices, get_text_extraction_model_choices


class GenerationPurpose(StrEnum):
    STRUCTURED_EXTRACTION = "structured_extraction"
    CLINICAL_SYNTHESIS = "clinical_synthesis"
    JSON_REPAIR = "json_repair"
    CONNECTIVITY_CHECK = "connectivity_check"


class PolicyMatchKind(StrEnum):
    EXACT_MODEL = "exact_model"
    MODEL_FAMILY = "model_family"
    PROVIDER = "provider"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class GenerationPolicy:
    policy_id: str
    policy_version: str
    provider: str
    model: str
    purpose: GenerationPurpose
    temperature: float | None
    uses_model_default: bool
    match_kind: PolicyMatchKind
    rationale: str


_CATALOG_PATH = Path(__file__).resolve().parents[3] / "resources" / "catalogs" / "llm_generation_policies.json"
_LOCAL_CATALOG_PATH = Path(__file__).resolve().parents[3] / "resources" / "catalogs" / "local_models.json"


def _load_catalog() -> dict[str, Any]:
    with _CATALOG_PATH.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("policy_version"), str):
        raise ValueError("Invalid LLM generation policy catalog")
    return catalog


_CATALOG = _load_catalog()


def _values(rule: dict[str, Any], purpose: GenerationPurpose, reasoning_enabled: bool) -> float | None:
    if "profile" in rule:
        profile = _CATALOG["profiles"][rule["profile"]]
        return profile[purpose.value]
    if "all" in rule:
        return rule["all"]
    if purpose.value in rule:
        return rule[purpose.value]
    mode = "reasoning_enabled" if reasoning_enabled else "reasoning_disabled"
    selected = rule.get(mode, rule.get("all"))
    if selected is None:
        return None
    if isinstance(selected, dict) and "all" in selected:
        return selected["all"]
    return selected.get(purpose.value)


def _policy(
    *, provider: str, model: str, purpose: GenerationPurpose, temperature: float | None,
    match_kind: PolicyMatchKind, rationale: str,
) -> GenerationPolicy:
    return GenerationPolicy(
        policy_id=f"{match_kind.value}:{provider}:{model}:{purpose.value}",
        policy_version=_CATALOG["policy_version"],
        provider=provider,
        model=model,
        purpose=purpose,
        temperature=temperature,
        uses_model_default=temperature is None,
        match_kind=match_kind,
        rationale=rationale,
    )


def resolve_generation_policy(
    *, purpose: GenerationPurpose, provider: str, model: str, reasoning_enabled: bool = False,
) -> GenerationPolicy:
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip()
    exact = _CATALOG.get("exact_models", {}).get(f"{normalized_provider}:{normalized_model}")
    if exact is not None:
        temperature = _values(exact, purpose, reasoning_enabled)
        return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                       temperature=temperature, match_kind=PolicyMatchKind.EXACT_MODEL,
                       rationale="Verified exact-model compatibility rule")
    family = normalized_model.split(":", 1)[0] if ":" in normalized_model else normalized_model
    family_rule = _CATALOG.get("families", {}).get(f"{normalized_provider}:{family}")
    if family_rule is None and normalized_provider == "openai" and normalized_model.startswith("gpt-5"):
        family_rule = _CATALOG.get("families", {}).get("openai:gpt-5")
    if family_rule is not None:
        temperature = _values(family_rule, purpose, reasoning_enabled)
        return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                       temperature=temperature, match_kind=PolicyMatchKind.MODEL_FAMILY,
                       rationale="Verified model-family compatibility rule")
    local_names = set(get_clinical_model_choices()) | set(get_text_extraction_model_choices())
    if normalized_provider == "ollama" and normalized_model in local_names:
        profile_name = _CATALOG["local_models"]["default_profile"]
        temperature = _CATALOG["profiles"][profile_name][purpose.value]
        return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                       temperature=temperature, match_kind=PolicyMatchKind.FALLBACK,
                       rationale="Catalogued local instruction-model product default")
    provider_rule = _CATALOG.get("providers", {}).get(normalized_provider)
    if provider_rule is not None:
        known_cloud_model = normalized_provider != "openai" or normalized_model.startswith(("gpt-", "o1", "o3", "o4"))
        if known_cloud_model:
            temperature = _values(provider_rule, purpose, reasoning_enabled)
            return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                           temperature=temperature, match_kind=PolicyMatchKind.PROVIDER,
                           rationale="Provider compatibility rule")
    return _policy(provider=normalized_provider, model=normalized_model, purpose=purpose,
                   temperature=None, match_kind=PolicyMatchKind.FALLBACK,
                   rationale="Unknown model uses the provider/model default")


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
