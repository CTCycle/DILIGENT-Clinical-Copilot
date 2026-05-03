from __future__ import annotations

from datetime import datetime
from typing import Any

from configurations.llm_configs import LLMRuntimeConfig
from domain.model_configs import ModelConfigSnapshot
from services.llm.model_config import ModelConfigService
from services.runtime.jobs import get_job_manager
from services.session.factory import build_clinical_session_service


class InMemorySerializer:
    def __init__(self, snapshot: ModelConfigSnapshot) -> None:
        self.snapshot = snapshot

    def load_snapshot(self) -> ModelConfigSnapshot:
        return self.snapshot

    def save_snapshot(self, **updates: Any) -> ModelConfigSnapshot:
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
            ollama_temperature=0.7,
            cloud_temperature=0.7,
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    service.serializer.save_snapshot(
        use_cloud_models=True,
        cloud_provider="openai",
        cloud_model="gpt-5.4-mini",
        clinical_model="gpt-oss:20b",
        text_extraction_model="qwen3:1.7b",
    )
    snapshot = service.serializer.load_snapshot()

    assert snapshot.use_cloud_models is True
    assert snapshot.cloud_provider == "openai"
    assert snapshot.cloud_model == "gpt-5.4-mini"
    assert snapshot.clinical_model == "gpt-oss:20b"
    assert snapshot.text_extraction_model == "qwen3:1.7b"


def test_clinical_service_reads_runtime_from_persisted_config() -> None:
    clinical_service = build_clinical_session_service(get_job_manager())
    clinical_service.apply_persisted_runtime_configuration()
    parser_provider, parser_model = LLMRuntimeConfig.resolve_provider_and_model(
        "parser"
    )
    assert parser_provider
    assert parser_model

