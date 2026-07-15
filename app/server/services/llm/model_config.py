from __future__ import annotations

from collections.abc import Iterable
from time import monotonic
from typing import Any, Protocol, cast

from common.exceptions import ServiceValidationError
from common.paths import VECTOR_DB_PATH
from common.utils.catalog_loader import CatalogLoader
from common.utils.logger import logger
from configurations.startup import get_server_settings
from common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_positive_int,
    coerce_str,
)
from domain.model_configs import (
    LocalModelCard,
    ModelConfigSnapshot,
    ModelConfigStateResponse,
    ModelConfigUpdateRequest,
    ConnectivityCheckRequest,
    ConnectivityCheckResponse,
)
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)
from services.llm.cloud import CloudLLMClient, LLMError
from services.llm.provider_registry import provider_registry
from domain.llm.providers import CloudModelDescriptor, CloudProviderDescriptor
from domain.llm.providers import CloudProviderId
from services.llm.ollama_client import OllamaClient, OllamaError
from repositories.vectors import LanceVectorDatabase
from services.retrieval.settings import (
    build_effective_rag_settings,
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
        clinical_model: str | None | object = ...,
        text_extraction_model: str | None | object = ...,
        use_cloud_models: bool | object = ...,
        cloud_provider: str | None | object = ...,
        cloud_model: str | None | object = ...,
        ollama_temperature: float | object = ...,
        cloud_temperature: float | object = ...,
        ollama_reasoning: bool | object = ...,
        ollama_seed: int | None | object = ...,
        rag_settings: dict[str, object] | object = ...,
    ) -> ModelConfigSnapshot: ...

###############################################################################
class ModelConfigService:
    _OLLAMA_WARNING_COOLDOWN_SECONDS = 120.0
    _FAST_LOCAL_EXTRACTION_MODELS = ("qwen3.5:2b", "qwen3.5:9b")

    # -------------------------------------------------------------------------
    def __init__(self, serializer: ModelConfigSnapshotStore | None = None) -> None:
        self.serializer = serializer or ModelConfigSerializer()
        self.local_model_catalog = cast(
            tuple[tuple[str, str, str], ...],
            CatalogLoader.get_catalog_records(
                "local_models.json",
                "local_model_catalog",
                ("name", "family", "description"),
            ),
        )
        self.local_model_names = {name for name, _, _ in self.local_model_catalog}
        self._last_ollama_warning_message: str | None = None
        self._last_ollama_warning_at = 0.0

    # -------------------------------------------------------------------------
    async def get_state(
        self,
        *,
        include_local_availability: bool | None = None,
    ) -> ModelConfigStateResponse:
        snapshot = self.ensure_defaults()
        should_check_local_availability = (
            include_local_availability
            if include_local_availability is not None
            else (not snapshot.use_cloud_models)
        )
        local_models = await self.list_local_model_cards(
            selected_models=(snapshot.clinical_model, snapshot.text_extraction_model),
            include_ollama_availability=should_check_local_availability,
        )
        return self.build_response(
            snapshot=snapshot,
            local_models=local_models,
            cloud_providers=await self.discover_provider_descriptors(),
        )

    # -------------------------------------------------------------------------
    async def update_state(
        self, payload: ModelConfigUpdateRequest
    ) -> ModelConfigStateResponse:
        snapshot = self.ensure_defaults()
        fields_set = payload.model_fields_set
        local_roles_updated = self._local_roles_updated(fields_set)
        target_use_cloud_models = (
            bool(payload.use_cloud_services)
            if "use_cloud_services" in fields_set
            else bool(snapshot.use_cloud_models)
        )
        should_refresh_local_availability = (
            not target_use_cloud_models
            or local_roles_updated
            or ("use_cloud_services" in fields_set and not target_use_cloud_models)
        )
        available_local_model_names = await self.list_available_ollama_models()
        local_model_names = await self._build_local_model_names(
            snapshot=snapshot,
            refresh_from_ollama=local_roles_updated,
        )
        updates = self._build_updates(
            payload=payload,
            snapshot=snapshot,
            fields_set=fields_set,
            local_model_names=local_model_names,
            available_local_model_names=available_local_model_names,
        )

        if updates:
            snapshot = self.serializer.save_snapshot(**updates)

        should_check_local_availability = (
            (not snapshot.use_cloud_models)
            or local_roles_updated
            or should_refresh_local_availability
        )
        local_models = await self.list_local_model_cards(
            selected_models=(snapshot.clinical_model, snapshot.text_extraction_model),
            include_ollama_availability=should_check_local_availability,
        )
        return self.build_response(
            snapshot=snapshot,
            local_models=local_models,
            cloud_providers=await self.discover_provider_descriptors(),
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
                    options={"temperature": 0.0},
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
        return "clinical_model" in fields_set or "text_extraction_model" in fields_set

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
        local_model_names = set(self.local_model_names)
        if snapshot.clinical_model:
            local_model_names.add(snapshot.clinical_model)
        if snapshot.text_extraction_model:
            local_model_names.add(snapshot.text_extraction_model)
        if not refresh_from_ollama:
            return local_model_names
        local_models_for_validation = await self.list_local_model_cards(
            selected_models=(snapshot.clinical_model, snapshot.text_extraction_model),
            include_ollama_availability=True,
        )
        return {item.name for item in local_models_for_validation}

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
        if "clinical_model" in fields_set:
            clinical_model = self.resolve_role_model_selection(
                role_name="clinical",
                model_name=self.normalize_optional_text(payload.clinical_model),
                local_model_names=local_model_names,
                available_local_model_names=available_local_model_names,
                use_cloud_models=use_cloud_models,
                cloud_provider=cloud_provider,
                active_cloud_model=active_cloud_model,
            )
            updates["clinical_model"] = clinical_model

        if "text_extraction_model" in fields_set:
            text_extraction_model = self.resolve_role_model_selection(
                role_name="text_extraction",
                model_name=self.normalize_optional_text(payload.text_extraction_model),
                local_model_names=local_model_names,
                available_local_model_names=available_local_model_names,
                use_cloud_models=use_cloud_models,
                cloud_provider=cloud_provider,
                active_cloud_model=active_cloud_model,
            )
            updates["text_extraction_model"] = text_extraction_model

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

        if (
            "ollama_temperature" in fields_set
            and payload.ollama_temperature is not None
        ):
            updates["ollama_temperature"] = payload.ollama_temperature

        if "cloud_temperature" in fields_set and payload.cloud_temperature is not None:
            updates["cloud_temperature"] = payload.cloud_temperature

        if "ollama_reasoning" in fields_set and payload.ollama_reasoning is not None:
            updates["ollama_reasoning"] = payload.ollama_reasoning
        if "ollama_seed" in fields_set:
            updates["ollama_seed"] = payload.ollama_seed

    # -------------------------------------------------------------------------
    @staticmethod
    def _collect_rag_settings_updates(
        *,
        payload: ModelConfigUpdateRequest,
        fields_set: set[str],
        updates: dict[str, Any],
    ) -> None:
        if "rag_settings" not in fields_set or payload.rag_settings is None:
            return
        updates["rag_settings"] = ModelConfigService.normalize_rag_settings_patch(
            payload.rag_settings
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_rag_settings_patch(payload: dict[str, object]) -> dict[str, object]:
        current = build_effective_rag_settings()
        candidate_count = coerce_positive_int(
            payload.get("retrieval_candidate_count"),
            current.retrieval_candidate_count,
        )
        selected_count = coerce_positive_int(
            payload.get("retrieval_selected_count"),
            current.retrieval_selected_count,
        )
        if selected_count > candidate_count:
            raise ServiceValidationError(
                "Selected RAG documents cannot exceed retrieved RAG documents."
            )
        return {
            "chunk_size": coerce_positive_int(
                payload.get("chunk_size"), current.chunk_size
            ),
            "chunk_overlap": coerce_positive_int(
                payload.get("chunk_overlap"), current.chunk_overlap
            ),
            "embedding_batch_size": coerce_positive_int(
                payload.get("embedding_batch_size"), current.embedding_batch_size
            ),
            "use_hybrid_search": coerce_bool(
                payload.get("use_hybrid_search"), current.use_hybrid_search
            ),
            "use_reranking": coerce_bool(
                payload.get("use_reranking"), current.use_reranking
            ),
            "retrieval_candidate_count": candidate_count,
            "retrieval_selected_count": selected_count,
            "reranker_model": coerce_str(
                payload.get("reranker_model"), current.reranker_model
            ),
            "hybrid_vector_weight": max(
                coerce_float(
                    payload.get("hybrid_vector_weight"), current.hybrid_vector_weight
                ),
                0.0,
            ),
            "hybrid_text_weight": max(
                coerce_float(
                    payload.get("hybrid_text_weight"), current.hybrid_text_weight
                ),
                0.0,
            ),
            "embedding_backend": coerce_str(
                payload.get("embedding_backend"), current.embedding_backend
            ),
            "ollama_embedding_model": coerce_str(
                payload.get("ollama_embedding_model"), current.ollama_embedding_model
            ),
            "hf_embedding_model": coerce_str(
                payload.get("hf_embedding_model"), current.hf_embedding_model
            ),
            "cloud_provider": coerce_str(
                payload.get("cloud_provider"), current.cloud_provider
            ),
            "cloud_embedding_model": coerce_str(
                payload.get("cloud_embedding_model"), current.cloud_embedding_model
            ),
            "use_cloud_embeddings": coerce_bool(
                payload.get("use_cloud_embeddings"), current.use_cloud_embeddings
            ),
            "reset_vector_collection": coerce_bool(
                payload.get("reset_vector_collection"), current.reset_vector_collection
            ),
            "vector_stream_batch_size": coerce_positive_int(
                payload.get("vector_stream_batch_size"),
                current.vector_stream_batch_size,
            ),
            "embedding_max_workers": coerce_positive_int(
                payload.get("embedding_max_workers"), current.embedding_max_workers
            ),
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_role_model_selection(
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
            cloud_model_names = set(provider_registry.get(cloud_provider).models)
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
    @staticmethod
    def resolve_cloud_model(provider: str, model_name: str | None) -> str | None:
        normalized = (model_name or "").strip()
        if not normalized:
            return None
        if not provider_registry.is_valid_model(cast(CloudProviderId, provider), normalized):
            raise ServiceValidationError(
                f"Model '{normalized}' is not valid for provider '{provider}'."
            )
        return normalized

    # -------------------------------------------------------------------------
    def ensure_defaults(self) -> ModelConfigSnapshot:
        snapshot = self.serializer.load_snapshot()
        defaults = get_server_settings().llm_defaults
        is_fresh = snapshot.updated_at is None and all(
            value is None
            for value in (
                snapshot.clinical_model,
                snapshot.text_extraction_model,
                snapshot.cloud_provider,
                snapshot.cloud_model,
            )
        )
        if is_fresh:
            return self.serializer.save_snapshot(
                clinical_model=defaults.clinical_model,
                text_extraction_model=defaults.text_extraction_model,
                use_cloud_models=defaults.use_cloud_services,
                cloud_provider=self.resolve_provider(defaults.llm_provider),
                cloud_model=self.resolve_cloud_model(
                    provider=self.resolve_provider(defaults.llm_provider),
                    model_name=defaults.cloud_model,
                ),
                ollama_temperature=defaults.ollama_temperature,
                cloud_temperature=defaults.cloud_temperature,
                ollama_reasoning=defaults.ollama_reasoning,
            )

        self.validate_current_snapshot(snapshot)
        return snapshot

    # -------------------------------------------------------------------------
    def validate_current_snapshot(self, snapshot: ModelConfigSnapshot) -> None:
        provider = snapshot.cloud_provider
        if provider is None and snapshot.cloud_model is not None:
            raise ServiceValidationError(
                "A cloud model cannot be configured without a cloud provider."
            )
        if provider is not None:
            provider = self.resolve_provider(provider)
        if snapshot.cloud_model is not None:
            self.resolve_cloud_model(provider or "", snapshot.cloud_model)

        for role_name, model_name in (
            ("clinical", snapshot.clinical_model),
            ("text_extraction", snapshot.text_extraction_model),
        ):
            if model_name is None:
                if not snapshot.use_cloud_models:
                    raise ServiceValidationError(
                        f"A model is required for the local '{role_name}' role."
                    )
                continue
            if model_name not in self.local_model_names and not snapshot.use_cloud_models:
                raise ServiceValidationError(
                    f"Model '{model_name}' is not supported for role '{role_name}'."
                )

        if snapshot.use_cloud_models:
            if provider is None or snapshot.cloud_model is None:
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
        try:
            async with OllamaClient() as client:
                models = await client.list_models()
        except OllamaError as exc:
            self._log_ollama_availability_warning(exc)
            return set()
        except Exception:
            logger.exception("Unexpected error while listing local Ollama models")
            return set()
        return {
            model.strip()
            for model in models
            if isinstance(model, str) and model.strip()
        }

    # -------------------------------------------------------------------------
    def _log_ollama_availability_warning(self, exc: OllamaError) -> None:
        message = str(exc)
        now = monotonic()
        is_duplicate = message == self._last_ollama_warning_message
        within_cooldown = (
            now - self._last_ollama_warning_at < self._OLLAMA_WARNING_COOLDOWN_SECONDS
        )
        if is_duplicate and within_cooldown:
            return
        logger.warning("Unable to list local Ollama models: %s", exc)
        self._last_ollama_warning_message = message
        self._last_ollama_warning_at = now

    # -------------------------------------------------------------------------
    async def list_local_model_cards(
        self,
        *,
        selected_models: Iterable[str | None] = (),
        include_ollama_availability: bool = True,
    ) -> list[LocalModelCard]:
        available_models = (
            await self.list_available_ollama_models()
            if include_ollama_availability
            else set()
        )
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
    def build_response(
        self,
        *,
        snapshot: ModelConfigSnapshot,
        local_models: list[LocalModelCard],
        cloud_providers: list[CloudProviderDescriptor] | None = None,
    ) -> ModelConfigStateResponse:
        provider = self.resolve_provider(snapshot.cloud_provider)
        cloud_model = self.resolve_cloud_model(
            provider=provider, model_name=snapshot.cloud_model
        )
        return ModelConfigStateResponse(
            local_models=local_models,
            cloud_providers=cloud_providers or self.build_provider_descriptors(),
            use_cloud_services=bool(snapshot.use_cloud_models),
            llm_provider=cast(CloudProviderId, provider),
            cloud_model=cloud_model,
            clinical_model=snapshot.clinical_model,
            text_extraction_model=snapshot.text_extraction_model,
            ollama_temperature=snapshot.ollama_temperature,
            cloud_temperature=snapshot.cloud_temperature,
            ollama_reasoning=snapshot.ollama_reasoning,
            ollama_seed=snapshot.ollama_seed,
            rag_settings=rag_settings_payload(build_effective_rag_settings()),
            rag_model=self.resolve_current_rag_model_label(),
            updated_at=snapshot.updated_at,
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
                catalog_status="available"
                if item.models
                else "authentication_required",
                models=[
                    CloudModelDescriptor(id=model, display_name=model)
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
                    CloudModelDescriptor(id=model, display_name=model)
                    for model in item.models
                ]
                status = "available"
            else:
                try:
                    async with CloudLLMClient(
                        provider=item.provider_id, timeout_s=15.0, max_retries=0
                    ) as client:
                        models = [
                            CloudModelDescriptor(id=model, display_name=model)
                            for model in await client.list_models()
                        ]
                    status = "available"
                except LLMError as exc:
                    models = []
                    status = (
                        "authentication_required"
                        if "access key" in str(exc).lower()
                        else "unavailable"
                    )
                except Exception:
                    models = []
                    status = "unavailable"
            descriptors.append(
                CloudProviderDescriptor(
                    id=item.provider_id,
                    display_name=item.display_name,
                    credential_scope=item.credential_scope,
                    capabilities=item.capabilities,
                    catalog_status=status,
                    models=models,
                )
            )
        return descriptors

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_current_rag_model_label() -> str | None:
        settings = build_effective_rag_settings()
        vector_db = LanceVectorDatabase(
            database_path=str(VECTOR_DB_PATH),
            collection_name=settings.vector_collection_name,
            metric=settings.vector_index_metric,
            index_type=settings.vector_index_type,
            stream_batch_size=settings.vector_stream_batch_size,
        )
        try:
            if not vector_db.has_collection():
                return None
            for batch in vector_db.iter_embeddings(batch_size=1, limit=1):
                for row in batch:
                    provider = str(row.get("vector_model_provider") or "").strip()
                    model_name = str(row.get("vector_model_name") or "").strip()
                    if provider and model_name:
                        return f"{provider}:{model_name}"
                    if model_name:
                        return model_name
        except Exception:
            return None
        return None
