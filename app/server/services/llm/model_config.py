# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Protocol, cast

from pydantic import TypeAdapter, ValidationError

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from common.exceptions import ServiceValidationError
from common.paths import VECTOR_DB_PATH
from common.utils.catalog_loader import CatalogLoader
from common.utils.logger import logger
from domain.llm.providers import (
    CloudModelDescriptor,
    CloudProviderDescriptor,
)
from domain.model_configs import (
    CatalogProviderId,
    ConnectivityCheckRequest,
    ConnectivityCheckResponse,
    EmbeddingIndexStatus,
    EmbeddingRuntimeStatus,
    EmbeddingStatusResponse,
    LocalCatalogMetadata,
    LocalModelCard,
    ModelCatalogOperationResponse,
    ModelConfigPersistResponse,
    ModelConfigSnapshot,
    ModelConfigStateResponse,
    ModelConfigUpdateRequest,
    SharedRagSettingsResponse,
)
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)
from repositories.serialization.provider_model_catalog_cache import (
    ProviderModelCatalogCacheSerializer,
)
from services.llm import model_catalog
from services.llm.cloud import CloudLLMClient, LLMError
from services.llm.generation_policy import GenerationPurpose
from services.llm.ollama_client import OllamaClient, OllamaError
from services.llm.provider_registry import provider_registry
from services.llm.model_capabilities import enrich_model_descriptor
from services.retrieval.embedding_runtime import get_embedding_runtime
from services.retrieval.settings import (
    build_effective_rag_settings,
    normalize_rag_settings_patch,
    rag_settings_payload,
)

_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, object])


def _manifest_count(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if value is None:
        return 0
    if isinstance(value, (str, int, float)):
        return int(value)
    return 0


###############################################################################
class ModelConfigSnapshotStore(Protocol):

    # -------------------------------------------------------------------------
    def load_snapshot(self) -> ModelConfigSnapshot: ...

    # -------------------------------------------------------------------------
    def save_snapshot(
        self, *, rag_settings: dict[str, object] | object = ...
    ) -> ModelConfigSnapshot: ...


###############################################################################
class ModelConfigService:
    _FAST_LOCAL_EXTRACTION_MODELS = ("qwen3.5:2b", "qwen3.5:9b")

    # -------------------------------------------------------------------------
    def __init__(
        self,
        serializer: ModelConfigSnapshotStore | None = None,
        catalog_cache: ProviderModelCatalogCacheSerializer | None = None,
    ) -> None:
        self.serializer = serializer or ModelConfigSerializer()
        self.catalog_cache = catalog_cache or ProviderModelCatalogCacheSerializer()
        self.local_model_catalog = cast(
            tuple[tuple[str, str, str], ...],
            CatalogLoader.get_catalog_records(
                "local_models.json",
                "local_model_catalog",
                ("name", "family", "description"),
            ),
        )
        self.local_model_names = {name for name, _, _ in self.local_model_catalog}
        self._catalog_tasks: dict[
            CatalogProviderId, asyncio.Task[ModelCatalogOperationResponse]
        ] = {}

    # -------------------------------------------------------------------------
    async def get_state(self) -> ModelConfigStateResponse:
        snapshot = await asyncio.to_thread(self.load_current_snapshot)
        local_models = await self.list_local_model_cards()
        local_catalog = await asyncio.to_thread(
            model_catalog.local_catalog_metadata, self.catalog_cache
        )
        return self.build_response(
            snapshot=snapshot,
            local_models=local_models,
            cloud_providers=await self.discover_provider_descriptors(),
            local_catalog=local_catalog,
        )

    # -------------------------------------------------------------------------
    async def update_state(
        self, payload: ModelConfigUpdateRequest
    ) -> ModelConfigPersistResponse:
        snapshot = await asyncio.to_thread(self.load_current_snapshot)
        if payload.rag_settings is not None:
            rag_settings = normalize_rag_settings_patch(
                payload.rag_settings.model_dump(exclude_unset=True),
                persisted_settings=snapshot.rag_settings,
            )
            snapshot = await asyncio.to_thread(
                self.serializer.save_snapshot, rag_settings=rag_settings
            )
        return self.build_persist_response(snapshot)

    # -------------------------------------------------------------------------
    async def get_embedding_status(self) -> EmbeddingStatusResponse:
        runtime, index = await asyncio.gather(
            asyncio.to_thread(self.build_embedding_runtime_status),
            asyncio.to_thread(self.build_embedding_index_status),
        )
        return EmbeddingStatusResponse(
            embedding_runtime=runtime,
            embedding_index=index,
        )

    # -------------------------------------------------------------------------
    async def check_connectivity(
        self, payload: ConnectivityCheckRequest
    ) -> ConnectivityCheckResponse:
        model = payload.model.strip()
        try:
            async with CloudLLMClient(
                provider=payload.provider,
                default_model=model,
                timeout_s=20.0,
                max_retries=0,
            ) as client:
                response = await client.chat(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Reply with exactly: OK",
                        },
                        {
                            "role": "user",
                            "content": "Provider connectivity check.",
                        },
                    ],
                    purpose=GenerationPurpose.CONNECTIVITY_CHECK,
                )
        except LLMError as exc:
            return ConnectivityCheckResponse(
                provider=payload.provider,
                model=model,
                ok=False,
                error=str(exc),
            )

        preview = self._normalize_connectivity_preview(response)
        return ConnectivityCheckResponse(
            provider=payload.provider,
            model=model,
            ok=True,
            response_preview=preview,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_connectivity_preview(response: object) -> str:
        preview = response if isinstance(response, str) else str(response)
        preview = " ".join(preview.split())
        return preview[:200]

    # -------------------------------------------------------------------------
    def known_provider_model_names(self, provider: str) -> set[str]:
        definition = provider_registry.get(provider)
        if definition.models:
            return set(definition.models)
        record = self.catalog_cache.get(
            definition.provider_id,
            model_catalog.catalog_configuration_fingerprint(definition.provider_id),
        )
        return {
            str(item.get("id"))
            for item in (record.models if record else [])
            if str(item.get("id") or "").strip()
        }

    # -------------------------------------------------------------------------
    def load_current_snapshot(self) -> ModelConfigSnapshot:
        """Load the persisted shared configuration without changing it."""
        try:
            return self.serializer.load_snapshot()
        except ValueError as exc:
            raise ServiceValidationError(str(exc)) from exc

    # -------------------------------------------------------------------------
    def known_local_model_names(self) -> set[str]:
        names = set(self.local_model_names)
        record = model_catalog.load_catalog_record(self.catalog_cache, "ollama")
        names.update(
            str(item.get("id"))
            for item in (record.models if record else [])
            if str(item.get("id") or "").strip()
        )
        return names

    # -------------------------------------------------------------------------
    @staticmethod
    def describe_local_model(model_name: str) -> str:
        family = model_name.split(":", maxsplit=1)[0].strip() or model_name
        return f"Installed local Ollama model from the {family} family."

    # -------------------------------------------------------------------------
    @classmethod
    def local_recommendation_rank(cls, model_name: str) -> int | None:
        try:
            return cls._FAST_LOCAL_EXTRACTION_MODELS.index(model_name)
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    @classmethod
    def build_local_model_card(
        cls,
        *,
        name: str,
        family: str,
        description: str,
        available_in_ollama: bool,
    ) -> LocalModelCard:
        recommendation_rank = cls.local_recommendation_rank(name)
        return LocalModelCard(
            name=name,
            family=family,
            description=description,
            available_in_ollama=available_in_ollama,
            recommended_for_local_extraction=recommendation_rank is not None,
            recommended_rank=recommendation_rank,
        )

    # -------------------------------------------------------------------------
    @classmethod
    def sort_local_model_cards(
        cls,
        cards: list[LocalModelCard],
    ) -> list[LocalModelCard]:
        return sorted(
            cards,
            key=lambda card: (
                0 if card.available_in_ollama else 1,
                card.recommended_rank
                if card.recommended_rank is not None
                else len(cls._FAST_LOCAL_EXTRACTION_MODELS),
                card.name.casefold(),
            ),
        )

    # -------------------------------------------------------------------------
    async def list_available_ollama_models(self) -> set[str]:
        record = await asyncio.to_thread(
            model_catalog.load_catalog_record, self.catalog_cache, "ollama"
        )
        return {
            str(item.get("id"))
            for item in (record.models if record else [])
            if str(item.get("id") or "").strip()
        }

    # -------------------------------------------------------------------------
    async def list_local_model_cards(self) -> list[LocalModelCard]:
        available_models = await self.list_available_ollama_models()

        cards = [
            self.build_local_model_card(
                name=name,
                family=family,
                description=description,
                available_in_ollama=name in available_models,
            )
            for name, family, description in self.local_model_catalog
        ]

        extra_models = sorted(
            available_models - self.local_model_names,
            key=str.casefold,
        )
        cards.extend(
            self.build_local_model_card(
                name=model_name,
                family="custom",
                description=self.describe_local_model(model_name),
                available_in_ollama=model_name in available_models,
            )
            for model_name in extra_models
        )
        return self.sort_local_model_cards(cards)

    # -------------------------------------------------------------------------
    def build_persist_response(
        self,
        snapshot: ModelConfigSnapshot,
    ) -> ModelConfigPersistResponse:
        return ModelConfigPersistResponse(
            rag_settings=self.shared_rag_settings(snapshot),
            updated_at=snapshot.updated_at,
        )

    # -------------------------------------------------------------------------
    def build_response(
        self,
        *,
        snapshot: ModelConfigSnapshot,
        local_models: list[LocalModelCard],
        cloud_providers: list[CloudProviderDescriptor] | None = None,
        local_catalog: LocalCatalogMetadata | None = None,
    ) -> ModelConfigStateResponse:
        return ModelConfigStateResponse(
            local_models=local_models,
            cloud_providers=(
                cloud_providers
                if cloud_providers is not None
                else self.build_provider_descriptors()
            ),
            local_catalog=local_catalog
            or model_catalog.local_catalog_metadata(self.catalog_cache),
            rag_settings=self.shared_rag_settings(snapshot),
            embedding_runtime=self.build_embedding_runtime_status(),
            embedding_index=self.build_embedding_index_status(),
            updated_at=snapshot.updated_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def shared_rag_settings(snapshot: ModelConfigSnapshot) -> SharedRagSettingsResponse:
        return rag_settings_payload(
            build_effective_rag_settings(persisted_settings=snapshot.rag_settings)
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def build_embedding_runtime_status() -> EmbeddingRuntimeStatus:
        status = get_embedding_runtime().status()
        return EmbeddingRuntimeStatus(
            model_display_name="Granite Embedding 97M Multilingual R2",
            model_revision=CANONICAL_EMBEDDING_CONFIG.revision,
            device=str(status["execution_provider"]),
            cache_status=str(status["cache_status"]),
            loaded=bool(status["loaded"]),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def build_embedding_index_status() -> EmbeddingIndexStatus:
        manifest_path = VECTOR_DB_PATH / "rag_index_manifest.json"
        if not manifest_path.is_file():
            return EmbeddingIndexStatus(status="reindex_required")
        try:
            payload = _JSON_OBJECT_ADAPTER.validate_json(manifest_path.read_bytes())
        except (OSError, ValidationError):
            return EmbeddingIndexStatus(status="corrupt")
        source = payload.get("source")
        source_data = (
            _JSON_OBJECT_ADAPTER.validate_python(source)
            if isinstance(source, dict)
            else {}
        )
        is_ready = (
            payload.get("manifest_version") == 2
            and payload.get("status") == "ready"
            and bool(payload.get("embedding_fingerprint"))
        )
        built_at_value = payload.get("built_at")
        built_at = None
        if isinstance(built_at_value, str):
            try:
                built_at = datetime.fromisoformat(built_at_value)
            except ValueError:
                built_at = None
        return EmbeddingIndexStatus(
            status="ready" if is_ready else "reindex_required",
            fingerprint=(
                str(payload["embedding_fingerprint"])
                if payload.get("embedding_fingerprint")
                else None
            ),
            document_count=_manifest_count(source_data, "document_count"),
            chunk_count=_manifest_count(source_data, "chunk_count"),
            built_at=built_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def build_provider_descriptors() -> list[CloudProviderDescriptor]:
        return [
            CloudProviderDescriptor(
                id=item.provider_id,
                display_name=item.display_name,
                credential_scope=item.credential_scope,
                capabilities=item.capabilities,
                catalog_status="available" if item.models else "not_loaded",
                models=[
                    enrich_model_descriptor(
                        provider=item.provider_id,
                        descriptor=CloudModelDescriptor(id=model, display_name=model),
                    )
                    for model in item.models
                ],
            )
            for item in provider_registry.all()
        ]

    # -------------------------------------------------------------------------
    async def discover_provider_descriptors(self) -> list[CloudProviderDescriptor]:
        descriptors: list[CloudProviderDescriptor] = []
        for item in provider_registry.all():
            if item.models:
                models = [
                    enrich_model_descriptor(
                        provider=item.provider_id,
                        descriptor=CloudModelDescriptor(id=model, display_name=model),
                    )
                    for model in item.models
                ]
                status = "available"
                catalog_updated_at = None
                message = None
            else:
                record = await asyncio.to_thread(
                    model_catalog.load_catalog_record,
                    self.catalog_cache,
                    item.provider_id,
                )
                models = [
                    enrich_model_descriptor(
                        provider=item.provider_id,
                        descriptor=descriptor,
                    )
                    for descriptor in model_catalog.cloud_models_from_record(record)
                ]
                if record is None:
                    status = "not_loaded"
                    catalog_updated_at = None
                    message = "Refresh this provider to load its model catalog."
                elif record.last_attempt_status == "success":
                    status = "available"
                    catalog_updated_at = record.last_success_at
                    message = "Provider catalog loaded from the saved cache."
                elif models:
                    status = "cached"
                    catalog_updated_at = record.last_success_at
                    message = (
                        "Showing models from the last successful provider refresh. "
                        f"Latest refresh failed: {record.last_error or 'provider unavailable'}"
                    )
                else:
                    status = (
                        "authentication_required"
                        if record.last_attempt_status == "authentication_required"
                        else "unavailable"
                    )
                    catalog_updated_at = None
                    message = (
                        record.last_error
                        or "The provider catalog could not be refreshed. Try again shortly."
                    )
            descriptors.append(
                CloudProviderDescriptor(
                    id=item.provider_id,
                    display_name=item.display_name,
                    credential_scope=item.credential_scope,
                    capabilities=item.capabilities,
                    catalog_status=status,
                    catalog_updated_at=catalog_updated_at,
                    catalog_message=message,
                    models=models,
                )
            )
        return descriptors

    # -------------------------------------------------------------------------
    async def load_catalog(
        self, provider: CatalogProviderId, *, force_refresh: bool = False
    ) -> ModelCatalogOperationResponse:
        if provider != "ollama":
            provider_registry.get(provider)
        cached_record = (
            await asyncio.to_thread(
                model_catalog.load_catalog_record, self.catalog_cache, provider
            )
            if not force_refresh
            else None
        )
        if not force_refresh and cached_record is not None:
            return ModelCatalogOperationResponse(
                catalog_provider=provider,
                outcome="cached",
                state=await self.get_state(),
            )
        task = self._catalog_tasks.get(provider)
        if task is None:
            task = asyncio.create_task(self._fetch_catalog(provider))
            self._catalog_tasks[provider] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and self._catalog_tasks.get(provider) is task:
                self._catalog_tasks.pop(provider, None)

    # -------------------------------------------------------------------------
    async def _fetch_catalog(
        self, provider: CatalogProviderId
    ) -> ModelCatalogOperationResponse:
        fingerprint = await asyncio.to_thread(
            model_catalog.catalog_configuration_fingerprint, provider
        )
        try:
            if provider == "ollama":
                async with OllamaClient() as client:
                    names = await client.list_models()
                models = [
                    {"id": name.strip(), "display_name": name.strip()}
                    for name in names
                    if name.strip()
                ]
            else:
                async with CloudLLMClient(
                    provider=provider, timeout_s=15.0, max_retries=0
                ) as client:
                    descriptors = await client.list_model_descriptors(
                        force_refresh=True
                    )
                if not descriptors:
                    raise LLMError(
                        "Provider returned no models available for clinical text generation."
                    )
                models = [item.model_dump(mode="json") for item in descriptors]
            await asyncio.to_thread(
                self.catalog_cache.save_success,
                provider_id=provider,
                configuration_fingerprint=fingerprint,
                models=models,
            )
            return ModelCatalogOperationResponse(
                catalog_provider=provider,
                outcome="refreshed",
                state=await self.get_state(),
            )
        except (LLMError, OllamaError) as exc:
            message = model_catalog.sanitize_catalog_error(str(exc))
            status = (
                "authentication_required"
                if "access key" in message.lower()
                else "unavailable"
            )
            await asyncio.to_thread(
                self.catalog_cache.save_failure,
                provider_id=provider,
                configuration_fingerprint=fingerprint,
                status=status,
                error=message,
            )
            return ModelCatalogOperationResponse(
                catalog_provider=provider,
                outcome="failed",
                error=message,
                state=await self.get_state(),
            )
        except Exception:
            logger.exception(
                "Unexpected error while refreshing %s model catalog", provider
            )
            message = "The provider model catalog could not be refreshed."
            await asyncio.to_thread(
                self.catalog_cache.save_failure,
                provider_id=provider,
                configuration_fingerprint=fingerprint,
                status="unavailable",
                error=message,
            )
            return ModelCatalogOperationResponse(
                catalog_provider=provider,
                outcome="failed",
                error=message,
                state=await self.get_state(),
            )
