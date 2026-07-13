from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.llm.providers import CloudProviderDefinition, ProviderCapabilities
from services.llm.provider_registry import ProviderRegistry, provider_registry

###############################################################################
def test_registry_exposes_exact_runtime_provider_set() -> None:
    assert {item.provider_id for item in provider_registry.all()} == {
        "openai",
        "gemini",
        "deepseek",
        "anthropic",
        "opencode_zen",
        "opencode_go",
    }

###############################################################################
@pytest.mark.parametrize(
    "alias", ["claude", "google", "opencode", "zen", "go", "open-code", "deep-seek"]
)
def test_runtime_aliases_are_rejected(alias: str) -> None:
    with pytest.raises(ValueError, match="Unsupported cloud provider"):
        provider_registry.get(alias)

###############################################################################
def test_opencode_runtime_providers_share_credential_scope() -> None:
    assert provider_registry.get("opencode_zen").credential_scope == "opencode"
    assert provider_registry.get("opencode_go").credential_scope == "opencode"

###############################################################################
def test_duplicate_provider_is_rejected() -> None:
    item = provider_registry.get("openai")
    with pytest.raises(ValueError, match="duplicate"):
        ProviderRegistry((item, item))

###############################################################################
def test_unknown_transport_is_rejected() -> None:
    payload = provider_registry.get("openai").model_dump()
    payload["transport_strategy"] = "unknown"
    with pytest.raises(ValidationError):
        CloudProviderDefinition.model_validate(payload)

###############################################################################
def test_invalid_default_model_is_rejected() -> None:
    with pytest.raises(ValidationError, match="default_model"):
        CloudProviderDefinition(
            provider_id="deepseek",
            display_name="DeepSeek",
            credential_scope="deepseek",
            discovery_strategy="static",
            models=("deepseek-v4-flash",),
            default_model="not-present",
            capabilities=ProviderCapabilities(
                chat=True,
                structured_output=True,
                reasoning=True,
                model_listing=False,
                embeddings=False,
                vision=False,
            ),
            transport_strategy="openai_chat_completions",
        )
