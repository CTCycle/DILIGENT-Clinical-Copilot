from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, status

from DILIGENT.common.constants import CLOUD_MODEL_CHOICES
from DILIGENT.common.utils.logger import logger
from DILIGENT.server.configurations import LLMRuntimeConfig
from DILIGENT.server.entities.modelconfig import (
    LocalModelCard,
    ModelConfigStateResponse,
    ModelConfigUpdateRequest,
)
from DILIGENT.server.models.providers import OllamaClient, OllamaError, OllamaTimeout
from DILIGENT.server.repositories.serialization.modelconfig import (
    ModelConfigSerializer,
    ModelConfigSnapshot,
)

router = APIRouter(prefix="/model-config", tags=["model-config"])
serializer = ModelConfigSerializer()


###############################################################################
class ModelConfigEndpoint:
    def __init__(
        self,
        *,
        router: APIRouter,
        serializer: ModelConfigSerializer,
    ) -> None:
        self.router = router
        self.serializer = serializer

    # -------------------------------------------------------------------------
    async def get_state(self) -> ModelConfigStateResponse:
        snapshot = self.ensure_defaults()
        self.apply_runtime_snapshot(snapshot)
        local_models = await self.list_local_model_cards()
        return self.build_response(snapshot=snapshot, local_models=local_models)

    # -------------------------------------------------------------------------
    async def update_state(
        self,
        payload: ModelConfigUpdateRequest = Body(...),
    ) -> ModelConfigStateResponse:
        snapshot = self.ensure_defaults()
        local_models = await self.list_local_model_cards()
        local_model_names = {item.name for item in local_models}
        updates: dict[str, Any] = {}
        fields_set = payload.model_fields_set

        if "clinical_model" in fields_set:
            clinical_model = self.normalize_optional_text(payload.clinical_model)
            self.validate_local_selection(
                role_name="clinical",
                model_name=clinical_model,
                local_model_names=local_model_names,
            )
            updates["clinical_model"] = clinical_model

        if "text_extraction_model" in fields_set:
            text_extraction_model = self.normalize_optional_text(
                payload.text_extraction_model
            )
            self.validate_local_selection(
                role_name="text_extraction",
                model_name=text_extraction_model,
                local_model_names=local_model_names,
            )
            updates["text_extraction_model"] = text_extraction_model

        provider = self.resolve_provider(
            payload.llm_provider
            if "llm_provider" in fields_set
            else snapshot.cloud_provider,
        )
        if "llm_provider" in fields_set:
            updates["cloud_provider"] = provider

        if "cloud_model" in fields_set or "llm_provider" in fields_set:
            requested_cloud_model = (
                payload.cloud_model if "cloud_model" in fields_set else None
            )
            normalized_cloud_model = self.resolve_cloud_model(
                provider=provider,
                model_name=requested_cloud_model,
            )
            updates["cloud_model"] = normalized_cloud_model

        if "use_cloud_services" in fields_set:
            updates["use_cloud_models"] = bool(payload.use_cloud_services)

        if updates:
            snapshot = self.serializer.save_snapshot(**updates)

        if "ollama_reasoning" in fields_set and payload.ollama_reasoning is not None:
            LLMRuntimeConfig.set_ollama_reasoning(payload.ollama_reasoning)

        self.apply_runtime_snapshot(snapshot)
        return self.build_response(snapshot=snapshot, local_models=local_models)

    # -------------------------------------------------------------------------
    @staticmethod
    def validate_local_selection(
        *,
        role_name: str,
        model_name: str | None,
        local_model_names: set[str],
    ) -> None:
        if model_name is None:
            return
        if not local_model_names:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No installed local Ollama models are currently available.",
            )
        if model_name not in local_model_names:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Model '{model_name}' is not installed for role '{role_name}'.",
            )

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
                LLMRuntimeConfig.get_parsing_model()
            )
        if snapshot.cloud_provider is None:
            updates["cloud_provider"] = runtime_provider
        if snapshot.cloud_model is None:
            updates["cloud_model"] = runtime_cloud_model
        if snapshot.cloud_provider is None and snapshot.cloud_model is None:
            updates["use_cloud_models"] = LLMRuntimeConfig.is_cloud_enabled()

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
    async def list_local_model_cards(self) -> list[LocalModelCard]:
        try:
            async with OllamaClient() as client:
                models = await client.list_models()
        except Exception as exc:
            if isinstance(exc, (OllamaTimeout, OllamaError)):
                logger.warning("Unable to list local Ollama models: %s", exc)
                return []
            logger.exception("Unexpected error while listing local Ollama models")
            return []
        normalized_models = sorted(
            {model.strip() for model in models if isinstance(model, str) and model.strip()},
            key=str.casefold,
        )
        return [
            LocalModelCard(
                name=model_name,
                description=self.describe_local_model(model_name),
            )
            for model_name in normalized_models
        ]

    # -------------------------------------------------------------------------
    def apply_runtime_snapshot(self, snapshot: ModelConfigSnapshot) -> None:
        if snapshot.text_extraction_model:
            LLMRuntimeConfig.set_parsing_model(snapshot.text_extraction_model)
        if snapshot.clinical_model:
            LLMRuntimeConfig.set_clinical_model(snapshot.clinical_model)

        provider = self.resolve_provider(snapshot.cloud_provider)
        LLMRuntimeConfig.set_llm_provider(provider)

        cloud_model = self.resolve_cloud_model(
            provider=provider,
            model_name=snapshot.cloud_model,
        )
        if cloud_model is None:
            LLMRuntimeConfig.set_cloud_model("")
        else:
            LLMRuntimeConfig.set_cloud_model(cloud_model)
        LLMRuntimeConfig.set_use_cloud_services(snapshot.use_cloud_models)

    # -------------------------------------------------------------------------
    def build_response(
        self,
        *,
        snapshot: ModelConfigSnapshot,
        local_models: list[LocalModelCard],
    ) -> ModelConfigStateResponse:
        provider = self.resolve_provider(snapshot.cloud_provider)
        cloud_model = self.resolve_cloud_model(provider=provider, model_name=snapshot.cloud_model)
        return ModelConfigStateResponse(
            local_models=local_models,
            cloud_model_choices=CLOUD_MODEL_CHOICES,
            use_cloud_services=bool(snapshot.use_cloud_models),
            llm_provider=provider,
            cloud_model=cloud_model,
            clinical_model=snapshot.clinical_model,
            text_extraction_model=snapshot.text_extraction_model,
            ollama_reasoning=LLMRuntimeConfig.is_ollama_reasoning_enabled(),
            updated_at=snapshot.updated_at,
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "",
            self.get_state,
            methods=["GET"],
            response_model=ModelConfigStateResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "",
            self.update_state,
            methods=["PUT"],
            response_model=ModelConfigStateResponse,
            status_code=status.HTTP_200_OK,
        )


endpoint = ModelConfigEndpoint(router=router, serializer=serializer)
endpoint.add_routes()
