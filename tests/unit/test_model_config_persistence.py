from __future__ import annotations

from datetime import datetime

from DILIGENT.server.api.model_config import ModelConfigEndpoint
from DILIGENT.server.api.session import endpoint as clinical_endpoint
from DILIGENT.server.configurations import LLMRuntimeConfig
from DILIGENT.server.repositories.serialization.model_configs import ModelConfigSnapshot


class InMemorySerializer:
    def __init__(self, snapshot: ModelConfigSnapshot) -> None:
        self.snapshot = snapshot

    def load_snapshot(self) -> ModelConfigSnapshot:
        return self.snapshot

    def save_snapshot(self, **updates):  # type: ignore[no-untyped-def]
        data = self.snapshot.__dict__.copy()
        data.update(updates)
        self.snapshot = ModelConfigSnapshot(**data)
        return self.snapshot


def test_model_config_roundtrip_preserves_cloud_selection() -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-oss:20b",
            text_extraction_model="qwen3:1.7b",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-5.4-mini",
            updated_at=datetime.now(),
        )
    )
    endpoint = ModelConfigEndpoint(router=clinical_endpoint.router, serializer=serializer)

    updated = endpoint.serializer.save_snapshot(
        use_cloud_models=True,
        cloud_provider="openai",
        cloud_model="gpt-5.4-mini",
        clinical_model="gpt-oss:20b",
        text_extraction_model="qwen3:1.7b",
    )
    endpoint.apply_runtime_snapshot(updated)
    snapshot = endpoint.serializer.load_snapshot()

    assert snapshot.use_cloud_models is True
    assert snapshot.cloud_provider == "openai"
    assert snapshot.cloud_model == "gpt-5.4-mini"
    assert snapshot.clinical_model == "gpt-oss:20b"
    assert snapshot.text_extraction_model == "qwen3:1.7b"


def test_clinical_runtime_overrides_are_request_scoped() -> None:
    initial = {
        "use_cloud": LLMRuntimeConfig.is_cloud_enabled(),
        "provider": LLMRuntimeConfig.get_llm_provider(),
        "cloud_model": LLMRuntimeConfig.get_cloud_model(),
    }
    with clinical_endpoint.runtime_override_context(
        use_cloud_services=True,
        llm_provider="openai",
        cloud_model="gpt-5.4-mini",
        parsing_model=None,
        clinical_model=None,
        ollama_temperature=None,
        ollama_reasoning=None,
    ):
        assert LLMRuntimeConfig.is_cloud_enabled() is True
        assert LLMRuntimeConfig.get_llm_provider() == "openai"
    assert LLMRuntimeConfig.is_cloud_enabled() == initial["use_cloud"]
    assert LLMRuntimeConfig.get_llm_provider() == initial["provider"]
    assert LLMRuntimeConfig.get_cloud_model() == initial["cloud_model"]
