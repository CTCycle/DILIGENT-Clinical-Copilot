from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from domain.model_configs import ModelConfigUpdateRequest
import services.llm.model_config as model_config_module
from configurations.llm_configs import LLMRuntimeConfig
from domain.model_configs import ModelConfigSnapshot
from services.llm.model_config import ModelConfigService
from services.llm.ollama_client import OllamaError
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
            cloud_model="gpt-4.1-mini",
            ollama_temperature=0.7,
            cloud_temperature=0.7,
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    service.serializer.save_snapshot(
        use_cloud_models=True,
        cloud_provider="openai",
        cloud_model="gpt-4.1-mini",
        clinical_model="gpt-oss:20b",
        text_extraction_model="qwen3:1.7b",
    )
    snapshot = service.serializer.load_snapshot()

    assert snapshot.use_cloud_models is True
    assert snapshot.cloud_provider == "openai"
    assert snapshot.cloud_model == "gpt-4.1-mini"
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


def test_model_config_service_accepts_cloud_models_for_role_assignments() -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-oss:20b",
            text_extraction_model="qwen3:14b",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            ollama_temperature=0.7,
            cloud_temperature=0.7,
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    payload = ModelConfigUpdateRequest(
        use_cloud_services=True,
        llm_provider="openai",
        cloud_model="gpt-4.1-mini",
        clinical_model="gpt-4.1-mini",
        text_extraction_model="gpt-4.1",
    )

    response = asyncio.run(service.update_state(payload))

    assert response.clinical_model == "gpt-4.1-mini"
    assert response.text_extraction_model == "gpt-4.1"
    assert serializer.snapshot.clinical_model == "gpt-4.1-mini"
    assert serializer.snapshot.text_extraction_model == "gpt-4.1"


def test_model_config_service_normalizes_stale_local_roles_in_cloud_mode() -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gemma4:31b",
            text_extraction_model="qwen3.5:9b",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            ollama_temperature=0.7,
            cloud_temperature=0.7,
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    payload = ModelConfigUpdateRequest(
        use_cloud_services=True,
        llm_provider="openai",
        cloud_model="gpt-4.1-mini",
        clinical_model="gemma4:31b",
        text_extraction_model="qwen3.5:9b",
    )

    response = asyncio.run(service.update_state(payload))

    assert response.clinical_model == "gpt-4.1-mini"
    assert response.text_extraction_model == "gpt-4.1-mini"
    assert serializer.snapshot.clinical_model == "gpt-4.1-mini"
    assert serializer.snapshot.text_extraction_model == "gpt-4.1-mini"


def test_model_config_service_throttles_repeated_ollama_warnings(monkeypatch) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-oss:20b",
            text_extraction_model="qwen3:14b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            ollama_temperature=0.7,
            cloud_temperature=0.7,
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    class FailingOllamaClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def list_models(self):
            raise OllamaError("Failed to list Ollama models: All connection attempts failed")

    warnings: list[str] = []
    times = iter([10.0, 20.0, 135.0])

    monkeypatch.setattr(model_config_module, "OllamaClient", FailingOllamaClient)
    monkeypatch.setattr(model_config_module, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        model_config_module.logger,
        "warning",
        lambda message, exc: warnings.append(f"{message}::{exc}"),
    )

    asyncio.run(service.list_available_ollama_models())
    asyncio.run(service.list_available_ollama_models())
    asyncio.run(service.list_available_ollama_models())

    assert len(warnings) == 2

