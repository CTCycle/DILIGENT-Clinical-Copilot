from __future__ import annotations

from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.model_config import ModelConfigService

###############################################################################
def test_model_config_service_ensure_defaults_applies_snapshot(monkeypatch) -> None:
    sentinel_snapshot = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        ModelConfigService,
        "ensure_defaults",
        lambda self: observed.setdefault("snapshot", sentinel_snapshot),
    )

    ModelConfigService().ensure_defaults()

    assert observed["snapshot"] is sentinel_snapshot

###############################################################################
def test_cloud_runtime_uses_cloud_model_when_role_models_are_local() -> None:
    with LLMRuntimeConfig.override_for_run(
        {
            "use_cloud_models": True,
            "cloud_provider": "openai",
            "cloud_model": "gpt-4.1-mini",
            "clinical_model": "gpt-oss:20b",
            "text_extraction_model": "qwen3:8b",
        }
    ):
        assert LLMRuntimeConfig.resolve_provider_and_model("clinical") == (
            "openai",
            "gpt-4.1-mini",
        )
        assert LLMRuntimeConfig.resolve_provider_and_model("parser") == (
            "openai",
            "gpt-4.1-mini",
        )

###############################################################################
def test_cloud_runtime_preserves_valid_cloud_role_override() -> None:
    with LLMRuntimeConfig.override_for_run(
        {
            "use_cloud_models": True,
            "cloud_provider": "openai",
            "cloud_model": "gpt-4.1-mini",
            "clinical_model": "gpt-4.1",
            "text_extraction_model": "gpt-4.1-mini",
        }
    ):
        assert LLMRuntimeConfig.resolve_provider_and_model("clinical") == (
            "openai",
            "gpt-4.1",
        )
        assert LLMRuntimeConfig.resolve_provider_and_model("parser") == (
            "openai",
            "gpt-4.1-mini",
        )
