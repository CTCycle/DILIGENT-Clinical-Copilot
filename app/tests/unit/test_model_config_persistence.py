from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from common.exceptions import ServiceValidationError
from domain.model_configs import ModelConfigUpdateRequest
import services.llm.model_config as model_config_module
from services.llm.runtime_config import LLMRuntimeConfig
from domain.model_configs import ConnectivityCheckRequest, ModelConfigSnapshot
from services.llm.cloud import LLMError
from services.llm.model_config import ModelConfigService
from domain.llm.providers import CloudModelDescriptor, CloudProviderDescriptor
from repositories.serialization.model_configs import ModelConfigSerializer
from repositories.schemas.base import Base
from services.llm.ollama_client import OllamaError
from services.runtime.jobs import get_job_manager
from services.session.factory import build_clinical_session_service

###############################################################################
class InMemorySerializer:

    # -------------------------------------------------------------------------
    def __init__(self, snapshot: ModelConfigSnapshot) -> None:
        self.snapshot = snapshot

    # -------------------------------------------------------------------------
    def load_snapshot(self) -> ModelConfigSnapshot:
        return self.snapshot

    # -------------------------------------------------------------------------
    def save_snapshot(self, **updates: Any) -> ModelConfigSnapshot:
        data = self.snapshot.__dict__.copy()
        data.update(updates)
        self.snapshot = ModelConfigSnapshot(**data)
        return self.snapshot

###############################################################################
def test_model_config_serializer_has_no_clean_break_migration() -> None:
    assert not hasattr(ModelConfigSerializer, "migrate_cloud_selection_clean_break")

###############################################################################
def test_model_config_serializer_refreshes_updated_at_on_save(tmp_path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'model-config.db'}")
    Base.metadata.create_all(engine)
    serializer = ModelConfigSerializer(engine=engine)
    serializer.save_snapshot(clinical_model="qwen3.5:2b")
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE application_configuration "
                "SET updated_at = '2000-01-01 00:00:00' WHERE id = 1"
            )
        )

    snapshot = serializer.save_snapshot(text_extraction_model="qwen3.5:2b")

    assert snapshot.updated_at is not None
    assert snapshot.updated_at.year > 2000

###############################################################################
def test_model_config_service_initializes_fresh_snapshot_from_canonical_defaults() -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model=None,
            text_extraction_model=None,
            use_cloud_models=False,
            cloud_provider=None,
            cloud_model=None,
            updated_at=None,
        )
    )
    snapshot = ModelConfigService(serializer=serializer).ensure_defaults()
    assert snapshot.clinical_model
    assert snapshot.text_extraction_model
    assert snapshot.cloud_provider

###############################################################################
@pytest.mark.parametrize(
    ("cloud_provider", "cloud_model"),
    [("legacy-openai", "gpt-4.1-mini")],
)
def test_model_config_service_rejects_invalid_persisted_cloud_selection(
    cloud_provider: str, cloud_model: str
) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="qwen3.5:2b",
            text_extraction_model="qwen3.5:2b",
            use_cloud_models=False,
            cloud_provider=cloud_provider,
            cloud_model=cloud_model,
            updated_at=datetime.now(),
        )
    )
    with pytest.raises(ServiceValidationError):
        ModelConfigService(serializer=serializer).ensure_defaults()

###############################################################################
def test_model_config_service_allows_persisted_deepseek_model_before_refresh() -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="deepseek-v4-flash",
            text_extraction_model="deepseek-v4-flash",
            use_cloud_models=True,
            cloud_provider="deepseek",
            cloud_model="deepseek-v4-flash",
            updated_at=datetime.now(),
        )
    )

    snapshot = ModelConfigService(serializer=serializer).ensure_defaults()

    assert snapshot.cloud_provider == "deepseek"
    assert snapshot.cloud_model == "deepseek-v4-flash"

###############################################################################
def test_model_config_state_survives_provider_catalog_drift(monkeypatch) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-4.1-mini",
            text_extraction_model="gpt-4.1-mini",
            use_cloud_models=True,
            cloud_provider="deepseek",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(UTC),
        )
    )
    service = ModelConfigService(serializer=serializer)
    monkeypatch.setitem(
        ModelConfigService._provider_catalog_cache,
        "deepseek",
        (
            datetime.now(UTC),
            [CloudModelDescriptor(id="deepseek-chat", display_name="DeepSeek Chat")],
        ),
    )

    async def fake_discover_provider_descriptors() -> list[CloudProviderDescriptor]:
        return ModelConfigService.build_provider_descriptors()

    monkeypatch.setattr(
        service,
        "discover_provider_descriptors",
        fake_discover_provider_descriptors,
    )

    response = asyncio.run(
        service.get_state(include_local_availability=False)
    )

    assert response.llm_provider == "deepseek"
    assert response.cloud_model == "gpt-4.1-mini"

###############################################################################
def test_model_config_catalog_keeps_configured_model_when_refresh_fails(
    monkeypatch,
) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-4.1-mini",
            text_extraction_model="gpt-4.1-mini",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(UTC),
        )
    )

    ###############################################################################
    class FailingCloudClient:

        # -------------------------------------------------------------------------
        def __init__(self, **_: Any) -> None:
            pass

        # -------------------------------------------------------------------------
        async def __aenter__(self) -> "FailingCloudClient":
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, *_: Any) -> None:
            return None

        # -------------------------------------------------------------------------
        async def list_model_descriptors(self) -> list[CloudModelDescriptor]:
            raise LLMError("provider catalog unavailable")

    monkeypatch.setattr(model_config_module, "CloudLLMClient", FailingCloudClient)
    descriptors = asyncio.run(
        ModelConfigService(serializer=serializer).discover_provider_descriptors()
    )
    openai = next(item for item in descriptors if item.id == "openai")

    assert openai.catalog_status == "unavailable"
    assert [model.id for model in openai.models] == ["gpt-4.1-mini"]
    assert openai.catalog_message == (
        "Provider catalog unavailable; showing the configured model."
    )

###############################################################################
def test_model_config_service_rejects_switching_cloud_model_roles_to_local_mode(
    monkeypatch,
) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="deepseek-v4-flash",
            text_extraction_model="deepseek-v4-flash",
            use_cloud_models=True,
            cloud_provider="deepseek",
            cloud_model="deepseek-v4-flash",
            ollama_reasoning=True,
            ollama_seed=42,
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    async def fake_list_available_ollama_models() -> set[str]:
        return {"qwen3.5:2b"}

    monkeypatch.setattr(
        service,
        "list_available_ollama_models",
        fake_list_available_ollama_models,
    )

    with pytest.raises(ServiceValidationError, match="not supported for role 'clinical'"):
        asyncio.run(service.update_state(ModelConfigUpdateRequest(use_cloud_services=False)))

    assert serializer.snapshot.use_cloud_models is True

###############################################################################
def test_model_config_service_rejects_invalid_persisted_local_model() -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="legacy-local-model",
            text_extraction_model="qwen3.5:2b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(),
        )
    )
    with pytest.raises(ServiceValidationError):
        ModelConfigService(serializer=serializer).ensure_defaults()

###############################################################################
def test_model_config_roundtrip_preserves_cloud_selection() -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-oss:20b",
            text_extraction_model="qwen3:1.7b",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
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

###############################################################################
def test_clinical_service_reads_runtime_from_persisted_config() -> None:
    clinical_service = build_clinical_session_service(get_job_manager())
    clinical_service.apply_persisted_runtime_configuration()
    parser_provider, parser_model = LLMRuntimeConfig.resolve_provider_and_model(
        "parser"
    )
    assert parser_provider
    assert parser_model

###############################################################################
def test_model_config_service_accepts_cloud_models_for_role_assignments() -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-oss:20b",
            text_extraction_model="qwen3:14b",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    payload = ModelConfigUpdateRequest(
        use_cloud_services=True,
        llm_provider="openai",
        cloud_model="gpt-4.1-mini",
        clinical_model="gpt-4.1-mini",
        text_extraction_model="gpt-4.1-mini",
    )

    response = asyncio.run(service.update_state(payload))

    assert response.clinical_model == "gpt-4.1-mini"
    assert response.text_extraction_model == "gpt-4.1-mini"
    assert serializer.snapshot.clinical_model == "gpt-4.1-mini"
    assert serializer.snapshot.text_extraction_model == "gpt-4.1-mini"

###############################################################################
def test_model_config_cloud_save_does_not_refresh_remote_catalogs_or_ollama(
    monkeypatch,
) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-4.1-mini",
            text_extraction_model="gpt-4.1-mini",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    async def unexpected_remote_call() -> object:
        raise AssertionError("saving must not refresh remote availability")

    monkeypatch.setattr(service, "list_available_ollama_models", unexpected_remote_call)
    monkeypatch.setattr(service, "discover_provider_descriptors", unexpected_remote_call)

    response = asyncio.run(
        service.update_state(
            ModelConfigUpdateRequest(
                use_cloud_services=True,
                llm_provider="openai",
                cloud_model="gpt-4.1-mini",
                clinical_model="gpt-4.1-mini",
                text_extraction_model="gpt-4.1-mini",
            )
        )
    )

    assert response.llm_provider == "openai"
    openai = next(provider for provider in response.cloud_providers if provider.id == "openai")
    assert [model.id for model in openai.models] == ["gpt-4.1-mini"]

###############################################################################
def test_model_config_service_rejects_stale_local_roles_in_cloud_mode() -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gemma4:31b",
            text_extraction_model="qwen3.5:9b",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
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

    with pytest.raises(ServiceValidationError, match="Select a model explicitly"):
        asyncio.run(service.update_state(payload))

###############################################################################
def test_model_config_service_rejects_uninstalled_local_models(monkeypatch) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="qwen3.5:2b",
            text_extraction_model="qwen3.5:2b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    async def fake_list_available_ollama_models() -> set[str]:
        return {"qwen3.5:2b"}

    monkeypatch.setattr(
        service,
        "list_available_ollama_models",
        fake_list_available_ollama_models,
    )

    payload = ModelConfigUpdateRequest(
        use_cloud_services=False,
        clinical_model="gpt-oss:20b",
        text_extraction_model="qwen3.5:2b",
    )

    with pytest.raises(ServiceValidationError, match="Install local Ollama model"):
        asyncio.run(service.update_state(payload))

###############################################################################
def test_model_config_service_prioritizes_recommended_installed_local_models(
    monkeypatch,
) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="qwen3.5:2b",
            text_extraction_model="qwen3.5:9b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    async def fake_list_available_ollama_models() -> set[str]:
        return {"qwen3.5:2b", "qwen3.5:9b", "gpt-oss:20b"}

    monkeypatch.setattr(
        service,
        "list_available_ollama_models",
        fake_list_available_ollama_models,
    )

    response = asyncio.run(service.get_state(include_local_availability=True))

    assert response.local_models[0].name == "qwen3.5:2b"
    assert response.local_models[0].recommended_for_local_extraction is True
    assert response.local_models[1].name == "qwen3.5:9b"

###############################################################################
def test_model_config_service_throttles_repeated_ollama_warnings(monkeypatch) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-oss:20b",
            text_extraction_model="qwen3:14b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    ###############################################################################
    class FailingOllamaClient:

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, exc_type, exc, tb):
            return False

        # -------------------------------------------------------------------------
        async def list_models(self):
            raise OllamaError(
                "Failed to list Ollama models: All connection attempts failed"
            )

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

###############################################################################
def test_connectivity_check_uses_requested_provider_and_model(monkeypatch) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-oss:20b",
            text_extraction_model="qwen3.5:9b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(),
        )
    )
    calls: list[dict[str, Any]] = []

    ###############################################################################
    class FakeCloudLLMClient:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs: Any) -> None:
            calls.append({"init": kwargs})

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, exc_type, exc, tb):
            return False

        # -------------------------------------------------------------------------
        async def chat(self, **kwargs: Any) -> str:
            calls.append({"chat": kwargs})
            return "OK"

    monkeypatch.setattr(model_config_module, "CloudLLMClient", FakeCloudLLMClient)

    response = asyncio.run(
        ModelConfigService(serializer=serializer).check_connectivity(
            ConnectivityCheckRequest(provider="openai", model="gpt-4.1-mini")
        )
    )

    assert response.ok is True
    assert response.provider == "openai"
    assert response.model == "gpt-4.1-mini"
    assert response.response_preview == "OK"
    assert calls[0]["init"]["provider"] == "openai"
    assert calls[1]["chat"]["model"] == "gpt-4.1-mini"

###############################################################################
def test_connectivity_check_reports_llm_error(monkeypatch) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="gpt-oss:20b",
            text_extraction_model="qwen3.5:9b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model=None,
            updated_at=datetime.now(),
        )
    )

    ###############################################################################
    class FailingCloudLLMClient:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs: Any) -> None:
            raise LLMError("No active OpenAI access key configured")

    monkeypatch.setattr(model_config_module, "CloudLLMClient", FailingCloudLLMClient)

    response = asyncio.run(
        ModelConfigService(serializer=serializer).check_connectivity(
            ConnectivityCheckRequest(provider="openai", model="gpt-4.1")
        )
    )

    assert response.ok is False
    assert response.provider == "openai"
    assert response.model == "gpt-4.1"
    assert response.error == "No active OpenAI access key configured"

###############################################################################
def test_provider_catalog_uses_last_successful_models_when_refresh_fails(monkeypatch) -> None:
    ModelConfigService._provider_catalog_cache.clear()
    failures: set[str] = set()

    ###############################################################################
    class FakeCloudLLMClient:

        # -------------------------------------------------------------------------
        def __init__(self, *, provider: str, **kwargs: Any) -> None:
            _ = kwargs
            self.provider = provider
            if provider in failures:
                raise LLMError("Provider request failed")

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, exc_type, exc, tb):
            return False

        # -------------------------------------------------------------------------
        async def list_model_descriptors(self) -> list[CloudModelDescriptor]:
            return [CloudModelDescriptor(id=f"{self.provider}-model", display_name="Model")]

    monkeypatch.setattr(model_config_module, "CloudLLMClient", FakeCloudLLMClient)
    service = ModelConfigService()

    first = asyncio.run(service.discover_provider_descriptors())
    assert next(item for item in first if item.id == "deepseek").catalog_status == "available"

    failures.add("deepseek")
    second = asyncio.run(service.discover_provider_descriptors())
    deepseek = next(item for item in second if item.id == "deepseek")
    assert deepseek.catalog_status == "cached"
    assert [item.id for item in deepseek.models] == ["deepseek-model"]
    assert deepseek.catalog_message == "Showing models from the last successful provider refresh."
    ModelConfigService._provider_catalog_cache.clear()

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

###############################################################################
def test_cloud_runtime_accepts_persisted_cloud_role_models(monkeypatch) -> None:
    snapshot = ModelConfigSnapshot(
        use_cloud_models=True,
        cloud_provider="openai",
        cloud_model="gpt-4.1-mini",
        clinical_model="gpt-4.1-mini",
        text_extraction_model="gpt-4.1-mini",
    )
    monkeypatch.setattr(ModelConfigSerializer, "load_snapshot", lambda self: snapshot)

    assert LLMRuntimeConfig.resolve_provider_and_model("clinical") == (
        "openai",
        "gpt-4.1-mini",
    )
    assert LLMRuntimeConfig.resolve_provider_and_model("parser") == (
        "openai",
        "gpt-4.1-mini",
    )
