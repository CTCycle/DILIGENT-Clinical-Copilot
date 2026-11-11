# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
import services.llm.model_config as model_config_module
from domain.llm.providers import CloudModelDescriptor, CloudProviderDescriptor
from domain.model_configs import (
    ConnectivityCheckRequest,
    ModelConfigSnapshot,
    ModelConfigUpdateRequest,
)
from domain.settings.configuration import SHARED_RAG_SETTING_KEYS
from pydantic import ValidationError
from repositories.schemas.base import Base
from repositories.serialization.model_configs import (
    LEGACY_EXECUTION_KEYS,
    ModelConfigSerializer,
)
from repositories.serialization.provider_model_catalog_cache import (
    ProviderModelCatalogCacheRecord,
    ProviderModelCatalogCacheSerializer,
)
from services.llm.cloud import LLMError
from services.llm.model_config import ModelConfigService
from services.llm.ollama_client import OllamaError
from sqlalchemy import create_engine, text


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
        base_snapshot = updates.pop("base_snapshot", None)
        data = (base_snapshot or self.snapshot).__dict__.copy()
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
    serializer.ensure_defaults({})
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE application_configuration "
                "SET updated_at = '2000-01-01 00:00:00' WHERE id = 1"
            )
        )

    snapshot = serializer.save_snapshot(rag_settings={"chunk_size": 256})

    assert snapshot.updated_at is not None
    assert snapshot.updated_at.year > 2000
    assert snapshot.rag_settings is not None
    assert snapshot.rag_settings["chunk_size"] == 256


###############################################################################
def test_model_config_update_preserves_other_settings_and_prunes_legacy_keys(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'settings-merge.db'}")
    Base.metadata.create_all(engine)
    serializer = ModelConfigSerializer(engine=engine)
    serializer.application_configuration.save(
        {
            "clinical_model": "clinical-model",
            "use_cloud_models": False,
            "reasoning_level": "high",
            "rag_settings": {"chunk_size": 300, "use_reranking": False},
            "jobs": {"polling_interval": 2.5},
            "runtime": {"livertox_archive": "custom.tar.gz"},
            "ingestion": {"drug_name_min_length": 4},
            "database": {"backend": "postgresql"},
        }
    )

    updated_snapshot = serializer.save_snapshot(
        rag_settings={"chunk_size": 400, "use_reranking": True}
    )
    saved = serializer.application_configuration.load()

    assert updated_snapshot.rag_settings is not None
    assert updated_snapshot.rag_settings["chunk_size"] == 400
    assert saved is not None
    assert not LEGACY_EXECUTION_KEYS & set(saved)
    assert set(saved["rag_settings"]) == set(SHARED_RAG_SETTING_KEYS)
    assert saved["jobs"] == {"polling_interval": 2.5}
    assert saved["runtime"]["livertox_archive"] == "custom.tar.gz"
    assert saved["ingestion"] == {"drug_name_min_length": 4}
    assert "database" not in saved


###############################################################################
@pytest.mark.parametrize(
    "payload",
    [
        {"use_cloud_services": True},
        {"clinical_model": "gpt-4.1"},
        {"reasoning_level": "high"},
        {"rag_settings": {"use_reranking": False}},
        {"rag_settings": {"retrieval_selected_count": 3}},
    ],
)
def test_model_config_update_rejects_agent_owned_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ModelConfigUpdateRequest.model_validate(payload)


###############################################################################
def test_shared_rag_save_does_not_probe_ollama_or_remote_catalogs(
    monkeypatch,
) -> None:
    serializer = InMemorySerializer(_snapshot(rag_settings={"chunk_size": 512}))
    service = ModelConfigService(serializer=serializer)

    async def unexpected_remote_call() -> object:
        raise AssertionError("saving must not refresh remote availability")

    monkeypatch.setattr(service, "list_available_ollama_models", unexpected_remote_call)
    monkeypatch.setattr(
        service, "discover_provider_descriptors", unexpected_remote_call
    )

    response = asyncio.run(
        service.update_state(
            ModelConfigUpdateRequest.model_validate(
                {"rag_settings": {"chunk_size": 1024}}
            )
        )
    )

    assert response.rag_settings.chunk_size == 1024
    assert serializer.snapshot.rag_settings is not None
    assert serializer.snapshot.rag_settings["chunk_size"] == 1024
    assert set(response.model_dump()) == {"rag_settings", "updated_at"}


###############################################################################
def test_application_configuration_concurrent_patches_do_not_drop_updates(
    tmp_path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from repositories.serialization.application_configuration import (
        ApplicationConfigurationSerializer,
    )

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'settings-concurrent.db'}")
    Base.metadata.create_all(engine)
    serializer = ApplicationConfigurationSerializer(engine=engine)
    serializer.save({"initial": True})

    def update(index: int) -> None:
        serializer.update_payload(lambda payload: payload.update({f"value_{index}": index}))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(update, range(24)))

    saved = serializer.load()
    assert saved is not None
    assert saved["initial"] is True
    assert {saved[f"value_{index}"] for index in range(24)} == set(range(24))


###############################################################################
def test_model_config_state_returns_only_shared_rag_settings(monkeypatch) -> None:
    serializer = InMemorySerializer(
        _snapshot(rag_settings={"chunk_size": 321, "vector_index_type": "HNSW"})
    )
    service = ModelConfigService(serializer=serializer)

    async def fake_list_local_model_cards(**_: Any) -> list[Any]:
        return []

    async def fake_discover_provider_descriptors() -> list[CloudProviderDescriptor]:
        return ModelConfigService.build_provider_descriptors()

    monkeypatch.setattr(service, "list_local_model_cards", fake_list_local_model_cards)
    monkeypatch.setattr(
        service, "discover_provider_descriptors", fake_discover_provider_descriptors
    )

    response = asyncio.run(service.get_state())

    assert response.rag_settings.chunk_size == 321
    assert response.rag_settings.vector_index_type == "HNSW"
    assert set(response.rag_settings.model_dump()) == set(SHARED_RAG_SETTING_KEYS)
    assert not {"use_cloud_services", "llm_provider", "clinical_model"} & set(
        response.model_dump()
    )


###############################################################################
def test_get_state_does_not_write_persisted_configuration(
    monkeypatch, tmp_path
) -> None:
    snapshot = _snapshot()


    ###############################################################################
    class ReadOnlySerializer(InMemorySerializer):

        # -------------------------------------------------------------------------
        def save_snapshot(self, **_: Any) -> ModelConfigSnapshot:
            raise AssertionError("GET model configuration must not persist state")

    service = _catalog_test_service(tmp_path, snapshot)
    service.serializer = ReadOnlySerializer(snapshot)

    async def fake_list_local_model_cards(**_: Any) -> list[Any]:
        return []

    async def fake_discover_provider_descriptors() -> list[CloudProviderDescriptor]:
        return []

    monkeypatch.setattr(service, "list_local_model_cards", fake_list_local_model_cards)
    monkeypatch.setattr(
        service, "discover_provider_descriptors", fake_discover_provider_descriptors
    )

    response = asyncio.run(service.get_state())

    assert response.updated_at == snapshot.updated_at


###############################################################################
def test_malformed_cached_catalog_entry_is_skipped() -> None:
    record = ProviderModelCatalogCacheRecord(
        provider_id="openai",
        configuration_fingerprint="test",
        models=[
            {"id": "valid-model", "display_name": "Valid model"},
            {"display_name": "Malformed model"},
        ],
        last_success_at=None,
        last_attempt_at=datetime.now(UTC),
        last_attempt_status="success",
        last_error=None,
    )

    models = model_config_module.model_catalog.cloud_models_from_record(record)

    assert [model.id for model in models] == ["valid-model"]


###############################################################################
def test_model_config_catalog_reports_failed_refresh(monkeypatch) -> None:
    serializer = InMemorySerializer(_snapshot())


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
        async def list_model_descriptors(self, **_: Any) -> list[CloudModelDescriptor]:
            raise LLMError("provider catalog unavailable")

    monkeypatch.setattr(model_config_module, "CloudLLMClient", FailingCloudClient)
    service = ModelConfigService(serializer=serializer)
    monkeypatch.setattr(
        model_config_module.model_catalog,
        "catalog_configuration_fingerprint",
        lambda _provider: "openai-test-fingerprint",
    )
    operation = asyncio.run(service.load_catalog("openai", force_refresh=True))
    openai = next(
        item for item in operation.state.cloud_providers if item.id == "openai"
    )

    assert operation.outcome == "failed"
    assert openai.catalog_status == "unavailable"
    assert openai.models == []
    assert openai.catalog_message == "provider catalog unavailable"


###############################################################################
def test_cold_local_catalog_loads_ollama_once(monkeypatch) -> None:
    serializer = InMemorySerializer(_snapshot())
    service = ModelConfigService(serializer=serializer)
    calls = 0


    ###############################################################################
    class FakeOllamaClient:

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, exc_type, exc, tb):
            return False

        # -------------------------------------------------------------------------
        async def list_models(self):
            nonlocal calls
            calls += 1
            return ["qwen3.5:2b"]

    monkeypatch.setattr(model_config_module, "OllamaClient", FakeOllamaClient)
    monkeypatch.setattr(
        model_config_module.model_catalog,
        "catalog_configuration_fingerprint",
        lambda _provider: "ollama-test-fingerprint",
    )
    first = asyncio.run(service.load_catalog("ollama"))
    second = asyncio.run(service.load_catalog("ollama"))

    assert first.outcome == "refreshed"
    assert second.outcome == "cached"
    assert calls == 1


###############################################################################
def test_model_config_service_prioritizes_recommended_installed_local_models(
    monkeypatch,
) -> None:
    serializer = InMemorySerializer(_snapshot())
    service = ModelConfigService(serializer=serializer)

    async def fake_list_available_ollama_models() -> set[str]:
        return {"qwen3.5:2b", "qwen3.5:9b", "gpt-oss:20b"}

    monkeypatch.setattr(
        service,
        "list_available_ollama_models",
        fake_list_available_ollama_models,
    )

    response = asyncio.run(service.get_state())

    assert response.local_models[0].name == "qwen3.5:2b"
    assert response.local_models[0].recommended_for_local_extraction is True
    assert response.local_models[1].name == "qwen3.5:9b"


###############################################################################
def test_failed_ollama_catalog_load_is_persisted_without_retry(monkeypatch) -> None:
    serializer = InMemorySerializer(_snapshot())
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

    monkeypatch.setattr(model_config_module, "OllamaClient", FailingOllamaClient)
    monkeypatch.setattr(
        model_config_module.model_catalog,
        "catalog_configuration_fingerprint",
        lambda _provider: "ollama-failure-fingerprint",
    )
    first = asyncio.run(service.load_catalog("ollama", force_refresh=True))
    second = asyncio.run(service.load_catalog("ollama"))

    assert first.outcome == "failed"
    assert second.outcome == "cached"


###############################################################################
def test_connectivity_check_uses_requested_provider_and_model(monkeypatch) -> None:
    serializer = InMemorySerializer(_snapshot())
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
    serializer = InMemorySerializer(_snapshot())


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
def test_provider_catalog_uses_last_successful_models_when_refresh_fails(
    monkeypatch,
) -> None:
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
        async def list_model_descriptors(self, **_: Any) -> list[CloudModelDescriptor]:
            if self.provider in failures:
                raise LLMError("Provider request failed")
            return [
                CloudModelDescriptor(id=f"{self.provider}-model", display_name="Model")
            ]

    monkeypatch.setattr(model_config_module, "CloudLLMClient", FakeCloudLLMClient)
    service = ModelConfigService()
    monkeypatch.setattr(
        model_config_module.model_catalog,
        "catalog_configuration_fingerprint",
        lambda provider: f"{provider}-test-fingerprint",
    )

    first = asyncio.run(service.load_catalog("deepseek", force_refresh=True))
    assert first.outcome == "refreshed"

    failures.add("deepseek")
    second = asyncio.run(service.load_catalog("deepseek", force_refresh=True))
    deepseek = next(
        item for item in second.state.cloud_providers if item.id == "deepseek"
    )
    assert second.outcome == "failed"
    assert deepseek.catalog_status == "cached"
    assert [item.id for item in deepseek.models] == ["deepseek-model"]
    assert "Latest refresh failed" in (deepseek.catalog_message or "")


###############################################################################
def test_empty_ollama_catalog_is_saved_as_a_valid_empty_result(
    monkeypatch, tmp_path
) -> None:
    service = _catalog_test_service(
        tmp_path,
        _snapshot(),
    )


    ###############################################################################
    class EmptyOllamaClient:

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, exc_type, exc, tb):
            return False

        # -------------------------------------------------------------------------
        async def list_models(self):
            return []

    monkeypatch.setattr(model_config_module, "OllamaClient", EmptyOllamaClient)
    monkeypatch.setattr(
        model_config_module.model_catalog,
        "catalog_configuration_fingerprint",
        lambda _provider: "empty-ollama-fingerprint",
    )

    result = asyncio.run(service.load_catalog("ollama", force_refresh=True))
    cached = asyncio.run(service.load_catalog("ollama"))

    assert result.outcome == "refreshed"
    assert cached.outcome == "cached"
    assert cached.state.local_catalog.status == "available"
    assert not {
        model.name for model in cached.state.local_models if model.available_in_ollama
    }


###############################################################################
def _catalog_test_service(
    tmp_path, snapshot: ModelConfigSnapshot
) -> ModelConfigService:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog-cache.db'}")
    Base.metadata.create_all(engine)
    return ModelConfigService(
        serializer=InMemorySerializer(snapshot),
        catalog_cache=ProviderModelCatalogCacheSerializer(engine=engine),
    )


###############################################################################
def test_catalog_provider_switching_keeps_provider_specific_lists(
    monkeypatch, tmp_path
) -> None:
    service = _catalog_test_service(
        tmp_path,
        _snapshot(),
    )


    ###############################################################################
    class FakeCloudClient:

        # -------------------------------------------------------------------------
        def __init__(self, *, provider: str, **_: Any) -> None:
            self.provider = provider

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, *_: Any) -> None:
            return None

        # -------------------------------------------------------------------------
        async def list_model_descriptors(self, **_: Any) -> list[CloudModelDescriptor]:
            return [
                CloudModelDescriptor(
                    id=f"{self.provider}-model", display_name=self.provider
                )
            ]

    monkeypatch.setattr(model_config_module, "CloudLLMClient", FakeCloudClient)
    monkeypatch.setattr(
        model_config_module.model_catalog,
        "catalog_configuration_fingerprint",
        lambda provider: f"{provider}-fingerprint",
    )

    asyncio.run(service.load_catalog("openai", force_refresh=True))
    asyncio.run(service.load_catalog("deepseek", force_refresh=True))
    state = asyncio.run(service.get_state())

    openai = next(item for item in state.cloud_providers if item.id == "openai")
    deepseek = next(item for item in state.cloud_providers if item.id == "deepseek")
    assert [item.id for item in openai.models] == ["openai-model"]
    assert [item.id for item in deepseek.models] == ["deepseek-model"]


###############################################################################
def test_concurrent_catalog_loads_share_one_provider_fetch(
    monkeypatch, tmp_path
) -> None:
    service = _catalog_test_service(
        tmp_path,
        _snapshot(),
    )
    calls = 0


    ###############################################################################
    class FakeCloudClient:

        # -------------------------------------------------------------------------
        def __init__(self, **_: Any) -> None:
            pass

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, *_: Any) -> None:
            return None

        # -------------------------------------------------------------------------
        async def list_model_descriptors(self, **_: Any) -> list[CloudModelDescriptor]:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return [CloudModelDescriptor(id="openai-model", display_name="OpenAI")]

    monkeypatch.setattr(model_config_module, "CloudLLMClient", FakeCloudClient)
    monkeypatch.setattr(
        model_config_module.model_catalog,
        "catalog_configuration_fingerprint",
        lambda _provider: "same-fingerprint",
    )

    async def run_concurrent():
        return await asyncio.gather(
            service.load_catalog("openai", force_refresh=True),
            service.load_catalog("openai", force_refresh=True),
        )

    first, second = asyncio.run(run_concurrent())

    assert first.outcome == "refreshed"
    assert second.outcome == "refreshed"
    assert calls == 1


###############################################################################
def test_catalog_fingerprint_change_invalidates_saved_models(
    tmp_path, monkeypatch
) -> None:
    service = _catalog_test_service(
        tmp_path,
        _snapshot(),
    )
    service.catalog_cache.save_success(
        provider_id="openai",
        configuration_fingerprint="old-fingerprint",
        models=[{"id": "old-model", "display_name": "Old"}],
    )
    monkeypatch.setattr(
        model_config_module.model_catalog,
        "catalog_configuration_fingerprint",
        lambda _provider: "new-fingerprint",
    )

    state = asyncio.run(service.get_state())
    openai = next(item for item in state.cloud_providers if item.id == "openai")

    assert openai.catalog_status == "not_loaded"
    assert openai.models == []


###############################################################################
def _snapshot(rag_settings: dict[str, object] | None = None) -> ModelConfigSnapshot:
    return ModelConfigSnapshot(rag_settings=rag_settings, updated_at=datetime.now(UTC))
