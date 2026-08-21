from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import create_engine, text

import services.llm.model_config as model_config_module
import services.llm.runtime_config as runtime_config_module
from common.exceptions import ServiceValidationError
from domain.llm.providers import CloudModelDescriptor, CloudProviderDescriptor
from domain.model_configs import (
    ConnectivityCheckRequest,
    ModelConfigSnapshot,
    ModelConfigUpdateRequest,
)
from repositories.schemas.base import Base
from repositories.serialization.model_configs import ModelConfigSerializer
from repositories.serialization.provider_model_catalog_cache import (
    ProviderModelCatalogCacheRecord,
    ProviderModelCatalogCacheSerializer,
)
from services.llm.cloud import LLMError
from services.llm.model_config import ModelConfigService
from services.llm.ollama_client import OllamaError
from services.llm.runtime_config import LLMRuntimeConfig
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
        base_snapshot = updates.pop("base_snapshot", None)
        data = (base_snapshot or self.snapshot).__dict__.copy()
        data.update(updates)
        self.snapshot = ModelConfigSnapshot(**data)
        return self.snapshot

###############################################################################
def test_model_config_serializer_has_no_clean_break_migration() -> None:
    assert not hasattr(ModelConfigSerializer, "migrate_cloud_selection_clean_break")

###############################################################################
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, "off"),
        ({"reasoning_level": "invalid"}, "off"),
        ({"reasoning_level": "high"}, "high"),
    ],
)
def test_model_config_serializer_reads_only_reasoning_level(
    payload: dict[str, object], expected: str
) -> None:
    snapshot = ModelConfigSerializer.snapshot_from_payload(payload, updated_at=None)

    assert snapshot.reasoning_level.value == expected

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
def test_model_config_serializer_persists_independent_revision_and_timeline_roles(tmp_path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'model-config-roles.db'}")
    Base.metadata.create_all(engine)
    serializer = ModelConfigSerializer(engine=engine)

    snapshot = serializer.save_snapshot(
        clinical_model="clinical-model",
        text_extraction_model="parser-model",
        revision_model="revision-model",
        timeline_model="timeline-model",
        use_cloud_models=True,
        cloud_provider="openai",
        cloud_model="gpt-4.1-mini",
    )
    reloaded = serializer.load_snapshot()

    assert snapshot.revision_model == "revision-model"
    assert snapshot.timeline_model == "timeline-model"
    assert reloaded.revision_model == "revision-model"
    assert reloaded.timeline_model == "timeline-model"

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
    async def fake_discover_provider_descriptors(
        _snapshot: ModelConfigSnapshot,
    ) -> list[CloudProviderDescriptor]:
        return ModelConfigService.build_provider_descriptors()

    monkeypatch.setattr(
        service,
        "discover_provider_descriptors",
        fake_discover_provider_descriptors,
    )

    response = asyncio.run(
        service.get_state()
    )

    assert response.llm_provider == "deepseek"
    assert response.cloud_model == "gpt-4.1-mini"

###############################################################################
def test_model_config_state_returns_persisted_rag_settings(monkeypatch) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="qwen3.5:2b",
            text_extraction_model="qwen3.5:2b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            rag_settings={
                "use_hybrid_search": False,
                "retrieval_candidate_count": 12,
                "retrieval_selected_count": 4,
                "reranker_model": "persisted-reranker",
            },
            updated_at=datetime.now(UTC),
        )
    )
    service = ModelConfigService(serializer=serializer)

    async def fake_list_local_model_cards(**_: Any) -> list[Any]:
        return []

    async def fake_discover_provider_descriptors(
        _snapshot: ModelConfigSnapshot,
    ) -> list[CloudProviderDescriptor]:
        return ModelConfigService.build_provider_descriptors()

    monkeypatch.setattr(service, "list_local_model_cards", fake_list_local_model_cards)
    monkeypatch.setattr(
        service, "discover_provider_descriptors", fake_discover_provider_descriptors
    )

    response = asyncio.run(service.get_state())

    assert response.rag_settings.use_hybrid_search is False
    assert response.rag_settings.retrieval_candidate_count == 12
    assert response.rag_settings.retrieval_selected_count == 4
    assert response.rag_settings.reranker_model == "persisted-reranker"

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
    openai = next(item for item in operation.state.cloud_providers if item.id == "openai")

    assert operation.outcome == "failed"
    assert openai.catalog_status == "unavailable"
    assert [model.id for model in openai.models] == ["gpt-4.1-mini"]
    assert openai.catalog_message == "provider catalog unavailable"

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
            reasoning_level="medium",
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
def test_model_config_service_allows_installed_dynamic_local_model(
    monkeypatch,
) -> None:
    dynamic_model = "huihui_ai/gpt-oss-abliterated:20b"
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="deepseek-v4-flash",
            text_extraction_model="deepseek-v4-flash",
            revision_model="deepseek-v4-pro",
            timeline_model="deepseek-v4-flash",
            use_cloud_models=True,
            cloud_provider="deepseek",
            cloud_model="deepseek-v4-flash",
            updated_at=datetime.now(UTC),
        )
    )
    service = ModelConfigService(serializer=serializer)
    cached_catalog = ProviderModelCatalogCacheRecord(
        provider_id="ollama",
        configuration_fingerprint="ollama-test-fingerprint",
        models=[{"id": dynamic_model, "display_name": dynamic_model}],
        last_success_at=datetime.now(UTC),
        last_attempt_at=datetime.now(UTC),
        last_attempt_status="success",
        last_error=None,
    )

    monkeypatch.setattr(
        model_config_module.model_catalog,
        "load_catalog_record",
        lambda _cache, _provider: cached_catalog,
    )

    async def fake_list_available_ollama_models() -> set[str]:
        return {dynamic_model}

    monkeypatch.setattr(
        service,
        "list_available_ollama_models",
        fake_list_available_ollama_models,
    )

    asyncio.run(
        service.update_state(
            ModelConfigUpdateRequest(
                use_cloud_services=False,
                clinical_model=dynamic_model,
                text_extraction_model=dynamic_model,
                revision_model=dynamic_model,
                timeline_model=dynamic_model,
            )
        )
    )

    assert serializer.snapshot.use_cloud_models is False
    assert serializer.snapshot.clinical_model == dynamic_model
    assert serializer.snapshot.revision_model == dynamic_model
    assert serializer.snapshot.timeline_model == dynamic_model

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
    assert response.cloud_model == "gpt-4.1-mini"
    assert not hasattr(response, "cloud_providers")

###############################################################################
@pytest.mark.parametrize(
    "patch",
    [
        {"reasoning_level": "medium"},
        {"rag_settings": {"use_hybrid_search": False}},
    ],
)
def test_local_option_saves_do_not_probe_ollama(
    monkeypatch, patch: dict[str, object]
) -> None:
    serializer = InMemorySerializer(
        ModelConfigSnapshot(
            clinical_model="qwen3.5:2b",
            text_extraction_model="qwen3.5:2b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            rag_settings={"use_hybrid_search": True},
            updated_at=datetime.now(),
        )
    )
    service = ModelConfigService(serializer=serializer)

    async def unexpected_probe() -> set[str]:
        raise AssertionError("option persistence must not probe Ollama")

    monkeypatch.setattr(service, "list_available_ollama_models", unexpected_probe)

    response = asyncio.run(
        service.update_state(ModelConfigUpdateRequest.model_validate(patch))
    )

    assert response.updated_at is not None

###############################################################################
def test_local_model_save_reuses_cached_availability(monkeypatch) -> None:
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
    async def cached_availability() -> set[str]:
        return {"qwen3.5:2b"}

    monkeypatch.setattr(service, "list_available_ollama_models", cached_availability)
    response = asyncio.run(
        service.update_state(
            ModelConfigUpdateRequest(
                use_cloud_services=False,
                clinical_model="qwen3.5:2b",
                text_extraction_model="qwen3.5:2b",
            )
        )
    )

    assert response.use_cloud_services is False
    assert serializer.snapshot.clinical_model == "qwen3.5:2b"

###############################################################################
def test_cold_local_catalog_loads_ollama_once(monkeypatch) -> None:
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

    response = asyncio.run(service.get_state())

    assert response.local_models[0].name == "qwen3.5:2b"
    assert response.local_models[0].recommended_for_local_extraction is True
    assert response.local_models[1].name == "qwen3.5:9b"

###############################################################################
def test_failed_ollama_catalog_load_is_persisted_without_retry(monkeypatch) -> None:
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
            return [CloudModelDescriptor(id=f"{self.provider}-model", display_name="Model")]

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
    deepseek = next(item for item in second.state.cloud_providers if item.id == "deepseek")
    assert second.outcome == "failed"
    assert deepseek.catalog_status == "cached"
    assert [item.id for item in deepseek.models] == ["deepseek-model"]
    assert "Latest refresh failed" in (deepseek.catalog_message or "")

###############################################################################
def test_empty_ollama_catalog_is_saved_as_a_valid_empty_result(monkeypatch, tmp_path) -> None:
    service = _catalog_test_service(
        tmp_path,
        ModelConfigSnapshot(
            clinical_model="gpt-oss:20b",
            text_extraction_model="qwen3:14b",
            use_cloud_models=False,
            cloud_provider="openai",
            cloud_model=None,
            updated_at=datetime.now(),
        ),
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
    assert not {model.name for model in cached.state.local_models if model.available_in_ollama}

###############################################################################
def _catalog_test_service(tmp_path, snapshot: ModelConfigSnapshot) -> ModelConfigService:
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
        ModelConfigSnapshot(
            clinical_model="gpt-4.1-mini",
            text_extraction_model="gpt-4.1-mini",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(UTC),
        ),
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
def test_concurrent_catalog_loads_share_one_provider_fetch(monkeypatch, tmp_path) -> None:
    service = _catalog_test_service(
        tmp_path,
        ModelConfigSnapshot(
            clinical_model="gpt-4.1-mini",
            text_extraction_model="gpt-4.1-mini",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(UTC),
        ),
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
def test_catalog_fingerprint_change_invalidates_saved_models(tmp_path, monkeypatch) -> None:
    service = _catalog_test_service(
        tmp_path,
        ModelConfigSnapshot(
            clinical_model="gpt-4.1-mini",
            text_extraction_model="gpt-4.1-mini",
            use_cloud_models=True,
            cloud_provider="openai",
            cloud_model="gpt-4.1-mini",
            updated_at=datetime.now(UTC),
        ),
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
def test_cloud_runtime_uses_cloud_model_when_role_models_are_local() -> None:
    with LLMRuntimeConfig.override_for_run(
        {
            "use_cloud_models": True,
            "cloud_provider": "openai",
            "cloud_model": "gpt-4.1-mini",
            "clinical_model": "gpt-oss:20b",
            "text_extraction_model": "qwen3:8b",
            "revision_model": "gpt-oss:20b",
            "timeline_model": "qwen3:8b",
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
def test_local_runtime_accepts_cached_dynamic_ollama_model(monkeypatch) -> None:
    dynamic_model = "huihui_ai/gpt-oss-abliterated:20b"
    snapshot = ModelConfigSnapshot(
        use_cloud_models=False,
        cloud_provider="openai",
        cloud_model="gpt-4.1-mini",
        clinical_model=dynamic_model,
        text_extraction_model=dynamic_model,
        revision_model=dynamic_model,
        timeline_model=dynamic_model,
    )
    cached_catalog = ProviderModelCatalogCacheRecord(
        provider_id="ollama",
        configuration_fingerprint="ollama-runtime-test-fingerprint",
        models=[{"id": dynamic_model, "display_name": dynamic_model}],
        last_success_at=datetime.now(UTC),
        last_attempt_at=datetime.now(UTC),
        last_attempt_status="success",
        last_error=None,
    )
    monkeypatch.setattr(ModelConfigSerializer, "load_snapshot", lambda self: snapshot)
    monkeypatch.setattr(
        runtime_config_module.model_catalog,
        "load_catalog_record",
        lambda _cache, _provider: cached_catalog,
    )

    assert LLMRuntimeConfig.resolve_provider_and_model("clinical") == (
        "ollama",
        dynamic_model,
    )
    assert LLMRuntimeConfig.resolve_provider_and_model("timeline") == (
        "ollama",
        dynamic_model,
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
            "revision_model": "gpt-4.1",
            "timeline_model": "gpt-4.1-mini",
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
        revision_model="gpt-4.1",
        timeline_model="gpt-4.1-mini",
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
