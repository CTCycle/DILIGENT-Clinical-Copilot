from __future__ import annotations

from collections.abc import Iterable
from time import monotonic
from typing import Any, Protocol, cast

from common.constants import CLOUD_MODEL_CHOICES
from common.exceptions import ServiceValidationError
from common.utils.catalog_loader import CatalogLoader
from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from domain.model_configs import (
    LocalModelCard,
    ModelConfigSnapshot,
    ModelConfigStateResponse,
    ModelConfigUpdateRequest,
)
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)
from services.llm.ollama_client import OllamaClient, OllamaError


###############################################################################
class ModelConfigSnapshotStore(Protocol):
    def load_snapshot(self) -> ModelConfigSnapshot: ...
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
    ) -> ModelConfigSnapshot: ...


###############################################################################
class ModelConfigService:
    _OLLAMA_WARNING_COOLDOWN_SECONDS = 120.0

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
        return self.build_response(snapshot=snapshot, local_models=local_models)

    # -------------------------------------------------------------------------
    async def update_state(
        self, payload: ModelConfigUpdateRequest
    ) -> ModelConfigStateResponse:
        snapshot = self.ensure_defaults()
        fields_set = payload.model_fields_set
        local_roles_updated = self._local_roles_updated(fields_set)
        local_model_names = await self._build_local_model_names(
            snapshot=snapshot,
            refresh_from_ollama=local_roles_updated,
        )
        updates = self._build_updates(
            payload=payload,
            snapshot=snapshot,
            fields_set=fields_set,
            local_model_names=local_model_names,
        )

        if updates:
            snapshot = self.serializer.save_snapshot(**updates)

        should_check_local_availability = (
            not snapshot.use_cloud_models
        ) or local_roles_updated
        local_models = await self.list_local_model_cards(
            selected_models=(snapshot.clinical_model, snapshot.text_extraction_model),
            include_ollama_availability=should_check_local_availability,
        )
        return self.build_response(snapshot=snapshot, local_models=local_models)

    # -------------------------------------------------------------------------
    @staticmethod
    def _local_roles_updated(fields_set: set[str]) -> bool:
        return "clinical_model" in fields_set or "text_extraction_model" in fields_set

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
        return updates

    # -------------------------------------------------------------------------
    def _collect_local_model_updates(
        self,
        *,
        payload: ModelConfigUpdateRequest,
        fields_set: set[str],
        local_model_names: set[str],
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

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_role_model_selection(
        *,
        role_name: str,
        model_name: str | None,
        local_model_names: set[str],
        use_cloud_models: bool,
        cloud_provider: str,
        active_cloud_model: str | None,
    ) -> str | None:
        if model_name is None:
            return None
        if use_cloud_models:
            cloud_model_names = set(CLOUD_MODEL_CHOICES.get(cloud_provider, []))
            if not cloud_model_names:
                raise ServiceValidationError(
                    f"No cloud models are available for provider '{cloud_provider}'.",
                )
            if model_name not in cloud_model_names:
                fallback = (active_cloud_model or "").strip()
                if fallback in cloud_model_names:
                    return fallback
                return next(iter(sorted(cloud_model_names)))
            return model_name
        if not local_model_names:
            raise ServiceValidationError("No model catalog entries are available.")
        if model_name not in local_model_names:
            raise ServiceValidationError(
                f"Model '{model_name}' is not supported for role '{role_name}'.",
            )
        return model_name

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_provider(provider: str | None) -> str:
        normalized = (provider or "").strip().lower()
        if normalized in CLOUD_MODEL_CHOICES:
            return normalized
        if LLMRuntimeConfig.get_llm_provider() in CLOUD_MODEL_CHOICES:
            return LLMRuntimeConfig.get_llm_provider()
        available = sorted(CLOUD_MODEL_CHOICES.keys())
        return available[0] if available else "openai"

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_cloud_model(provider: str, model_name: str | None) -> str | None:
        models = CLOUD_MODEL_CHOICES.get(provider, [])
        if not models:
            return None
        normalized = (model_name or "").strip()
        if normalized and normalized in models:
            return normalized
        return models[0]

    # -------------------------------------------------------------------------
    def ensure_defaults(self) -> ModelConfigSnapshot:
        snapshot = self.serializer.load_snapshot()
        updates: dict[str, Any] = {}
        runtime_provider = self.resolve_provider(LLMRuntimeConfig.get_llm_provider())
        snapshot_provider = self.resolve_provider(snapshot.cloud_provider)
        runtime_cloud_model = self.resolve_cloud_model(
            provider=runtime_provider,
            model_name=LLMRuntimeConfig.get_cloud_model(),
        )

        if snapshot.clinical_model is None:
            updates["clinical_model"] = self.normalize_optional_text(
                LLMRuntimeConfig.get_clinical_model()
            )
        if snapshot.text_extraction_model is None:
            updates["text_extraction_model"] = self.normalize_optional_text(
                LLMRuntimeConfig.get_text_extraction_model()
            )
        if snapshot.cloud_provider is None:
            updates["cloud_provider"] = runtime_provider
        if snapshot.cloud_model is None:
            updates["cloud_model"] = self.resolve_cloud_model(
                provider=snapshot_provider,
                model_name=runtime_cloud_model,
            )
        if (
            snapshot.cloud_provider is None
            and snapshot.cloud_model is None
            and snapshot.updated_at is None
        ):
            updates["use_cloud_models"] = LLMRuntimeConfig.is_cloud_enabled()
        if snapshot.updated_at is None:
            updates["ollama_reasoning"] = LLMRuntimeConfig.is_ollama_reasoning_enabled()

        if updates:
            return self.serializer.save_snapshot(**updates)
        return snapshot

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
            LocalModelCard(
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
            LocalModelCard(
                name=model_name,
                family="custom",
                description=self.describe_local_model(model_name),
                available_in_ollama=model_name in available_models,
            )
            for model_name in extra_models
        )
        return cards

    # -------------------------------------------------------------------------
    def build_response(
        self,
        *,
        snapshot: ModelConfigSnapshot,
        local_models: list[LocalModelCard],
    ) -> ModelConfigStateResponse:
        provider = self.resolve_provider(snapshot.cloud_provider)
        cloud_model = self.resolve_cloud_model(
            provider=provider, model_name=snapshot.cloud_model
        )
        return ModelConfigStateResponse(
            local_models=local_models,
            cloud_model_choices=CLOUD_MODEL_CHOICES,
            use_cloud_services=bool(snapshot.use_cloud_models),
            llm_provider=provider,
            cloud_model=cloud_model,
            clinical_model=snapshot.clinical_model,
            text_extraction_model=snapshot.text_extraction_model,
            ollama_temperature=snapshot.ollama_temperature,
            cloud_temperature=snapshot.cloud_temperature,
            ollama_reasoning=snapshot.ollama_reasoning,
        )
