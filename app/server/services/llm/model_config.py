from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol, cast

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from common.exceptions import ServiceValidationError
from common.paths import VECTOR_DB_PATH
from common.utils.catalog_loader import CatalogLoader
from common.utils.logger import logger
from domain.llm.providers import (
    CloudModelDescriptor,
    CloudProviderDescriptor,
    CloudProviderId,
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
    ReasoningLevel,
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
from services.retrieval.embedding_runtime import get_embedding_runtime
from services.retrieval.settings import (
    build_effective_rag_settings,
    normalize_rag_settings_patch,
    rag_settings_payload,
)


###############################################################################
class ModelConfigSnapshotStore(Protocol):
    # -------------------------------------------------------------------------
    def load_snapshot(self) -> ModelConfigSnapshot: ...

    # -------------------------------------------------------------------------
    def save_snapshot(
        self,
        *,
        base_snapshot: ModelConfigSnapshot | None = None,
        clinical_model: str | None | object = ...,
        text_extraction_model: str | None | object = ...,
        revision_model: str | None | object = ...,
        timeline_model: str | None | object = ...,
        use_cloud_models: bool | object = ...,
        cloud_provider: str | None | object = ...,
        cloud_model: str | None | object = ...,
        reasoning_level: ReasoningLevel | object = ...,
        ollama_seed: int | None | object = ...,
        rag_settings: dict[str, object] | object = ...,
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
        snapshot = self.load_current_snapshot()
        local_models = await self.list_local_model_cards(
            selected_models=(
                snapshot.clinical_model,
                snapshot.text_extraction_model,
                snapshot.revision_model,
                snapshot.timeline_model,
            ),
        )
        local_catalog = model_catalog.local_catalog_metadata(self.catalog_cache)
        return self.build_response(
            snapshot=snapshot,
            local_models=local_models,
            cloud_providers=await self.discover_provider_descriptors(snapshot),
            local_catalog=local_catalog,
        )

    # -------------------------------------------------------------------------
    async def update_state(
        self, payload: ModelConfigUpdateRequest
    ) -> ModelConfigPersistResponse:
        snapshot = self.load_current_snapshot()
        fields_set = payload.model_fields_set
        local_roles_updated = self._local_roles_updated(fields_set)
        target_use_cloud_models = (
            bool(payload.use_cloud_services)
            if "use_cloud_services" in fields_set
            else bool(snapshot.use_cloud_models)
        )
        requires_local_model_availability = not target_use_cloud_models and (
            local_roles_updated
            or ("use_cloud_services" in fields_set and not target_use_cloud_models)
        )
        available_local_model_names = (
            await self.list_available_ollama_models()
            if requires_local_model_availability
            else set()
        )
        local_model_names = await self._build_local_model_names(
            snapshot=snapshot,
            refresh_from_ollama=not target_use_cloud_models and local_roles_updated,
        )
        updates = self._build_updates(
            payload=payload,
            snapshot=snapshot,
            fields_set=fields_set,
            local_model_names=local_model_names,
            available_local_model_names=available_local_model_names,
        )

        if updates:
            self.validate_current_snapshot(replace(snapshot, **updates))
            snapshot = self.serializer.save_snapshot(
                base_snapshot=snapshot,
                **updates,
            )

        return self.build_persist_response(snapshot)

    # -------------------------------------------------------------------------
    async def get_embedding_status(self) -> EmbeddingStatusResponse:
        return EmbeddingStatusResponse(
            embedding_runtime=self.build_embedding_runtime_status(),
            embedding_index=self.build_embedding_index_status(),
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
    def _local_roles_updated(fields_set: set[str]) -> bool:
        return bool(
            fields_set
            & {
                "clinical_model",
                "text_extraction_model",
                "revision_model",
                "timeline_model",
            }
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_connectivity_preview(response: object) -> str:
        preview = response if isinstance(response, str) else str(response)
        preview = " ".join(preview.split())
        return preview[:200]

    # -------------------------------------------------------------------------
    async def _build_local_model_names(
        self,
        *,
        snapshot: ModelConfigSnapshot,
        refresh_from_ollama: bool,
    ) -> set[str]:
        local_model_names = self.known_local_model_names()
        for model_name in (
            snapshot.clinical_model,
            snapshot.text_extraction_model,
            snapshot.revision_model,
            snapshot.timeline_model,
        ):
            if model_name:
                local_model_names.add(model_name)
        if not refresh_from_ollama:
            return local_model_names
        available_models = await self.list_available_ollama_models()
        return local_model_names | available_models

    # -------------------------------------------------------------------------
    def _build_updates(
        self,
        *,
        payload: ModelConfigUpdateRequest,
        snapshot: ModelConfigSnapshot,
        fields_set: set[str],
        local_model_names: set[str],
        available_local_model_names: set[str],
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        target_use_cloud_models = (
            bool(payload.use_cloud_services)
            if "use_cloud_services" in fields_set
            else bool(snapshot.use_cloud_models)
        )
        provider = self.resolve_provider(
            payload.llm_provider
            if "llm_provider" in fields_set
            else snapshot.cloud_provider
        )
        self._collect_local_model_updates(
            payload=payload,
            fields_set=fields_set,
            local_model_names=local_model_names,
            available_local_model_names=available_local_model_names,
            use_cloud_models=target_use_cloud_models,
            cloud_provider=provider,
            active_cloud_model=(
                self.normalize_optional_text(payload.cloud_model)
                if "cloud_model" in fields_set
                else self.normalize_optional_text(snapshot.cloud_model)
            ),
            updates=updates,
        )
        self._collect_cloud_model_updates(
            payload=payload,
            fields_set=fields_set,
            provider=provider,
            updates=updates,
        )
        self._collect_runtime_option_updates(
            payload=payload,
            fields_set=fields_set,
            updates=updates,
        )
        self._collect_rag_settings_updates(
            payload=payload,
            fields_set=fields_set,
            snapshot=snapshot,
            updates=updates,
        )
        return updates

    # -------------------------------------------------------------------------
    def _collect_local_model_updates(
        self,
        *,
        payload: ModelConfigUpdateRequest,
        fields_set: set[str],
        local_model_names: set[str],
        available_local_model_names: set[str],
        use_cloud_models: bool,
        cloud_provider: str,
        active_cloud_model: str | None,
        updates: dict[str, Any],
    ) -> None:
        for field_name, role_name in (
            ("clinical_model", "clinical"),
            ("text_extraction_model", "text_extraction"),
            ("revision_model", "revision"),
            ("timeline_model", "timeline"),
        ):
            if field_name not in fields_set:
                continue
            updates[field_name] = self.resolve_role_model_selection(
                role_name=role_name,
                model_name=self.normalize_optional_text(getattr(payload, field_name)),
                local_model_names=local_model_names,
                available_local_model_names=available_local_model_names,
                use_cloud_models=use_cloud_models,
                cloud_provider=cloud_provider,
                active_cloud_model=active_cloud_model,
            )

    # -------------------------------------------------------------------------
    def _collect_cloud_model_updates(
        self,
        *,
        payload: ModelConfigUpdateRequest,
        fields_set: set[str],
        provider: str,
        updates: dict[str, Any],
    ) -> None:
        if "llm_provider" in fields_set:
            updates["cloud_provider"] = provider

        if "cloud_model" in fields_set or "llm_provider" in fields_set:
            requested_cloud_model = (
                payload.cloud_model if "cloud_model" in fields_set else None
            )
            updates["cloud_model"] = self.resolve_cloud_model(
                provider=provider,
                model_name=requested_cloud_model,
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _collect_runtime_option_updates(
        *,
        payload: ModelConfigUpdateRequest,
        fields_set: set[str],
        updates: dict[str, Any],
    ) -> None:
        if "use_cloud_services" in fields_set:
            updates["use_cloud_models"] = bool(payload.use_cloud_services)

        if "reasoning_level" in fields_set and payload.reasoning_level is not None:
            updates["reasoning_level"] = payload.reasoning_level
        if "ollama_seed" in fields_set:
            updates["ollama_seed"] = payload.ollama_seed

    # -------------------------------------------------------------------------
    @staticmethod
    def _collect_rag_settings_updates(
        *,
        payload: ModelConfigUpdateRequest,
        fields_set: set[str],
        snapshot: ModelConfigSnapshot,
        updates: dict[str, Any],
    ) -> None:
        if "rag_settings" not in fields_set or payload.rag_settings is None:
            return
        updates["rag_settings"] = normalize_rag_settings_patch(
            payload.rag_settings.model_dump(exclude_unset=True),
            persisted_settings=snapshot.rag_settings,
        )

    # -------------------------------------------------------------------------
    def resolve_role_model_selection(
        self,
        *,
        role_name: str,
        model_name: str | None,
        local_model_names: set[str],
        available_local_model_names: set[str],
        use_cloud_models: bool,
        cloud_provider: str,
        active_cloud_model: str | None,
    ) -> str | None:
        if model_name is None:
            return None
        if use_cloud_models:
            cloud_model_names = self.known_provider_model_names(cloud_provider)
            if not cloud_model_names:
                if active_cloud_model and model_name == active_cloud_model:
                    return model_name
                raise ServiceValidationError(
                    "Select a model explicitly from the provider catalog."
                )
            if model_name not in cloud_model_names:
                raise ServiceValidationError(
                    f"Model '{model_name}' is not valid for provider '{cloud_provider}'."
                )
            return model_name
        if not local_model_names:
            raise ServiceValidationError("No model catalog entries are available.")
        if model_name not in local_model_names:
            raise ServiceValidationError(
                f"Model '{model_name}' is not supported for role '{role_name}'.",
            )
        if model_name not in available_local_model_names:
            raise ServiceValidationError(
                f"Install local Ollama model '{model_name}' before using it for role '{role_name}'.",
            )
        return model_name

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_provider(provider: str | None) -> str:
        normalized = (provider or "").strip().lower()
        try:
            return provider_registry.get(normalized).provider_id
        except ValueError as exc:
            raise ServiceValidationError(str(exc)) from exc

    # -------------------------------------------------------------------------
    def resolve_cloud_model(self, provider: str, model_name: str | None) -> str | None:
        normalized = (model_name or "").strip()
        if not normalized:
            return None
        known_models = self.known_provider_model_names(provider)
        if known_models and normalized not in known_models:
            raise ServiceValidationError(
                f"Model '{normalized}' is not valid for provider '{provider}'."
            )
        return normalized

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
        """Load the canonical persisted configuration without changing it."""
        try:
            snapshot = self.serializer.load_snapshot()
        except ValueError as exc:
            raise ServiceValidationError(str(exc)) from exc
        self.validate_current_snapshot(snapshot)
        return snapshot

    # -------------------------------------------------------------------------
    def validate_current_snapshot(self, snapshot: ModelConfigSnapshot) -> None:
        missing_roles = [
            field_name
            for field_name, model_name in (
                ("clinical_model", snapshot.clinical_model),
                ("text_extraction_model", snapshot.text_extraction_model),
                ("revision_model", snapshot.revision_model),
                ("timeline_model", snapshot.timeline_model),
            )
            if self.normalize_optional_text(model_name) is None
        ]
        if missing_roles:
            raise ServiceValidationError(
                "Persisted model configuration is missing required role assignments: "
                + ", ".join(missing_roles)
            )
        provider = self.normalize_optional_text(snapshot.cloud_provider)
        cloud_model = self.normalize_optional_text(snapshot.cloud_model)
        if provider is None:
            raise ServiceValidationError(
                "Model configuration requires a cloud provider selection."
            )
        provider = self.resolve_provider(provider)

        known_local_model_names = self.known_local_model_names()
        for role_name, model_name in (
            ("clinical", snapshot.clinical_model),
            ("text_extraction", snapshot.text_extraction_model),
            ("revision", snapshot.revision_model),
            ("timeline", snapshot.timeline_model),
        ):
            model_name = self.normalize_optional_text(model_name)
            if model_name is None:
                raise ServiceValidationError(
                    f"An explicit model is required for the '{role_name}' role."
                )
            if (
                model_name not in known_local_model_names
                and not snapshot.use_cloud_models
            ):
                raise ServiceValidationError(
                    f"Model '{model_name}' is not supported for role '{role_name}'."
                )

        if snapshot.use_cloud_models:
            if provider is None or cloud_model is None:
                raise ServiceValidationError(
                    "Cloud mode requires both a provider and a model."
                )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    # -------------------------------------------------------------------------
    def _required_role_model(self, model_name: str | None, role_name: str) -> str:
        normalized = self.normalize_optional_text(model_name)
        if normalized is None:
            raise ServiceValidationError(
                f"An explicit model is required for the '{role_name}' role."
            )
        return normalized

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
        record = model_catalog.load_catalog_record(self.catalog_cache, "ollama")
        return {
            str(item.get("id"))
            for item in (record.models if record else [])
            if str(item.get("id") or "").strip()
        }

    # -------------------------------------------------------------------------
    async def list_local_model_cards(
        self,
        *,
        selected_models: Iterable[str | None] = (),
    ) -> list[LocalModelCard]:
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

        selected_candidates = {
            candidate.strip()
            for candidate in selected_models
            if isinstance(candidate, str) and candidate.strip()
        }
        extra_models = sorted(
            (available_models | selected_candidates) - self.local_model_names,
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
        clinical_model = self._required_role_model(snapshot.clinical_model, "clinical")
        text_extraction_model = self._required_role_model(
            snapshot.text_extraction_model, "text_extraction"
        )
        revision_model = self._required_role_model(snapshot.revision_model, "revision")
        timeline_model = self._required_role_model(snapshot.timeline_model, "timeline")
        return ModelConfigPersistResponse(
            use_cloud_services=bool(snapshot.use_cloud_models),
            llm_provider=cast(
                CloudProviderId, self.resolve_provider(snapshot.cloud_provider)
            ),
            cloud_model=self.normalize_optional_text(snapshot.cloud_model),
            clinical_model=clinical_model,
            text_extraction_model=text_extraction_model,
            revision_model=revision_model,
            timeline_model=timeline_model,
            reasoning_level=snapshot.reasoning_level,
            ollama_seed=snapshot.ollama_seed,
            rag_settings=rag_settings_payload(
                build_effective_rag_settings(
                    persisted_settings=snapshot.rag_settings,
                )
            ),
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
        provider = self.resolve_provider(snapshot.cloud_provider)
        cloud_model = self.normalize_optional_text(snapshot.cloud_model)
        clinical_model = self._required_role_model(snapshot.clinical_model, "clinical")
        text_extraction_model = self._required_role_model(
            snapshot.text_extraction_model, "text_extraction"
        )
        revision_model = self._required_role_model(snapshot.revision_model, "revision")
        timeline_model = self._required_role_model(snapshot.timeline_model, "timeline")
        return ModelConfigStateResponse(
            local_models=local_models,
            cloud_providers=(
                cloud_providers
                if cloud_providers is not None
                else self.build_provider_descriptors()
            ),
            local_catalog=local_catalog
            or model_catalog.local_catalog_metadata(self.catalog_cache),
            use_cloud_services=bool(snapshot.use_cloud_models),
            llm_provider=cast(CloudProviderId, provider),
            cloud_model=cloud_model,
            clinical_model=clinical_model,
            text_extraction_model=text_extraction_model,
            revision_model=revision_model,
            timeline_model=timeline_model,
            reasoning_level=snapshot.reasoning_level,
            ollama_seed=snapshot.ollama_seed,
            rag_settings=rag_settings_payload(
                build_effective_rag_settings(persisted_settings=snapshot.rag_settings)
            ),
            embedding_runtime=self.build_embedding_runtime_status(),
            embedding_index=self.build_embedding_index_status(),
            updated_at=snapshot.updated_at,
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
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            return EmbeddingIndexStatus(status="corrupt")
        if not isinstance(payload, dict):
            return EmbeddingIndexStatus(status="corrupt")
        source = payload.get("source")
        source_data = source if isinstance(source, dict) else {}
        is_ready = (
            payload.get("manifest_version") == 2
            and payload.get("status") == "ready"
            and bool(payload.get("embedding_fingerprint"))
        )
        built_at_value = payload.get("built_at")
        built_at = None
        if isinstance(built_at_value, datetime):
            built_at = built_at_value
        elif isinstance(built_at_value, str):
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
            document_count=int(source_data.get("document_count", 0) or 0),
            chunk_count=int(source_data.get("chunk_count", 0) or 0),
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
                    CloudModelDescriptor(id=model, display_name=model)
                    for model in item.models
                ],
            )
            for item in provider_registry.all()
        ]

    # -------------------------------------------------------------------------
    async def discover_provider_descriptors(
        self, snapshot: ModelConfigSnapshot
    ) -> list[CloudProviderDescriptor]:
        descriptors: list[CloudProviderDescriptor] = []
        for item in provider_registry.all():
            if item.models:
                models = [
                    CloudModelDescriptor(id=model, display_name=model)
                    for model in item.models
                ]
                status = "available"
                catalog_updated_at = None
                message = None
            else:
                record = model_catalog.load_catalog_record(
                    self.catalog_cache, item.provider_id
                )
                models = model_catalog.cloud_models_from_record(record)
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
                    message = record.last_error or self._configured_catalog_message(
                        models
                    )
                    models = self._configured_provider_models(
                        snapshot, item.provider_id
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
        if (
            not force_refresh
            and model_catalog.load_catalog_record(self.catalog_cache, provider)
            is not None
        ):
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
        fingerprint = model_catalog.catalog_configuration_fingerprint(provider)
        try:
            if provider == "ollama":
                async with OllamaClient() as client:
                    names = await client.list_models()
                models = [
                    {"id": name.strip(), "display_name": name.strip()}
                    for name in names
                    if isinstance(name, str) and name.strip()
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
            self.catalog_cache.save_success(
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
            self.catalog_cache.save_failure(
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
            self.catalog_cache.save_failure(
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

    # -------------------------------------------------------------------------
    @staticmethod
    def _configured_provider_models(
        snapshot: ModelConfigSnapshot, provider_id: CloudProviderId
    ) -> list[CloudModelDescriptor]:
        if not snapshot.use_cloud_models or snapshot.cloud_provider != provider_id:
            return []
        model_names = {
            model.strip()
            for model in (
                snapshot.cloud_model,
                snapshot.clinical_model,
                snapshot.text_extraction_model,
                snapshot.revision_model,
                snapshot.timeline_model,
            )
            if isinstance(model, str) and model.strip()
        }
        return [
            CloudModelDescriptor(id=model, display_name=model)
            for model in sorted(model_names, key=str.casefold)
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _configured_catalog_message(models: list[CloudModelDescriptor]) -> str:
        if models:
            return "Provider catalog unavailable; showing the configured model."
        return "The provider catalog could not be refreshed. Try again shortly."

    # -------------------------------------------------------------------------
