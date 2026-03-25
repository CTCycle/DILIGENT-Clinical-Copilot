from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status

from DILIGENT.server.common.constants import CLOUD_MODEL_CHOICES
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.configurations import LLMRuntimeConfig
from DILIGENT.server.domain.model_configs import (
    LocalModelCard,
    ModelConfigStateResponse,
    ModelConfigUpdateRequest,
)
from DILIGENT.server.models.providers import OllamaClient, OllamaError, OllamaTimeout
from DILIGENT.server.repositories.serialization.model_configs import (
    ModelConfigSerializer,
    ModelConfigSnapshot,
)

router = APIRouter(prefix="/model-config", tags=["model-config"])
serializer = ModelConfigSerializer()
GRANITE31_FAMILY = "granite3.1"

LOCAL_MODEL_CATALOG: tuple[tuple[str, str, str], ...] = (
    (
        "llama3.1:8b",
        "llama3",
        "Meta Llama 3.1 8B instruction model for general-purpose generation (dense transformer).",
    ),
    (
        "llama3.1:70b",
        "llama3",
        "Meta Llama 3.1 70B instruction model for high-quality long-form reasoning (dense transformer).",
    ),
    (
        "dolphin3:8b",
        "dolphin3",
        "Dolphin 3 8B instruction model based on Llama 3.1, tuned for tool use and structured reasoning.",
    ),
    (
        "olmo2:7b",
        "olmo2",
        "AllenAI OLMo 2 7B fully open model for general language tasks.",
    ),
    (
        "olmo2:13b",
        "olmo2",
        "AllenAI OLMo 2 13B fully open model with expanded capability over 7B.",
    ),
    (
        "deepseek-v3:671b",
        "deepseek-v3",
        "DeepSeek-V3 671B Mixture-of-Experts language model for large-scale reasoning.",
    ),
    (
        "mistral-nemo:12b",
        "mistral-nemo",
        "Mistral NeMo 12B multilingual model from Mistral AI and NVIDIA.",
    ),
    (
        "smollm2:135m",
        "smollm2",
        "SmolLM2 135M compact open model optimized for lightweight local inference.",
    ),
    (
        "smollm2:360m",
        "smollm2",
        "SmolLM2 360M compact open model for low-latency general language tasks.",
    ),
    (
        "smollm2:1.7b",
        "smollm2",
        "SmolLM2 1.7B compact open model balancing quality and efficiency.",
    ),
    (
        "deepcoder:1.5b",
        "deepcoder",
        "DeepCoder 1.5B code-specialized model for programming and code reasoning tasks.",
    ),
    (
        "deepcoder:14b",
        "deepcoder",
        "DeepCoder 14B code-specialized model with stronger synthesis and debugging ability.",
    ),
    (
        "phi4-reasoning:14b",
        "phi4-reasoning",
        "Microsoft Phi-4 Reasoning 14B model focused on deliberate multi-step reasoning.",
    ),
    (
        "phi4-mini-reasoning:3.8b",
        "phi4-mini-reasoning",
        "Microsoft Phi-4 Mini Reasoning 3.8B model for efficient structured reasoning.",
    ),
    (
        "granite3.1-dense:2b",
        GRANITE31_FAMILY,
        "IBM Granite 3.1 Dense 2B model for compact enterprise-oriented language tasks.",
    ),
    (
        "granite3.1-dense:8b",
        GRANITE31_FAMILY,
        "IBM Granite 3.1 Dense 8B model with improved general utility.",
    ),
    (
        "granite3.1-moe:1b",
        GRANITE31_FAMILY,
        "IBM Granite 3.1 MoE 1B model using sparse expert routing for efficiency.",
    ),
    (
        "granite3.1-moe:3b",
        GRANITE31_FAMILY,
        "IBM Granite 3.1 MoE 3B model using sparse expert routing for stronger quality.",
    ),
    (
        "granite3.3:2b",
        "granite3.3",
        "IBM Granite 3.3 2B open model for compact enterprise and coding tasks.",
    ),
    (
        "granite3.3:8b",
        "granite3.3",
        "IBM Granite 3.3 8B open model tuned for higher-quality enterprise use.",
    ),
    (
        "granite4:350m",
        "granite4",
        "IBM Granite 4 350M model for extremely lightweight local inference.",
    ),
    (
        "granite4:1b",
        "granite4",
        "IBM Granite 4 1B model for compact, general language workloads.",
    ),
    (
        "granite4:3b",
        "granite4",
        "IBM Granite 4 3B model for higher-quality local language tasks.",
    ),
    (
        "deepseek-r1:1.5b",
        "deepseek-r1",
        "DeepSeek-R1 1.5B reasoning model from the R1 family.",
    ),
    (
        "deepseek-r1:7b",
        "deepseek-r1",
        "DeepSeek-R1 7B reasoning model for local multi-step problem solving.",
    ),
    (
        "deepseek-r1:8b",
        "deepseek-r1",
        "DeepSeek-R1 8B reasoning model variant in the R1 lineup.",
    ),
    (
        "deepseek-r1:14b",
        "deepseek-r1",
        "DeepSeek-R1 14B reasoning model balancing quality and efficiency.",
    ),
    (
        "deepseek-r1:32b",
        "deepseek-r1",
        "DeepSeek-R1 32B reasoning model for stronger long-form reasoning.",
    ),
    (
        "deepseek-r1:70b",
        "deepseek-r1",
        "DeepSeek-R1 70B reasoning model for high-quality complex tasks.",
    ),
    (
        "deepseek-r1:671b",
        "deepseek-r1",
        "DeepSeek-R1 671B flagship reasoning model in the R1 series.",
    ),
    (
        "exaone-deep:2.4b",
        "exaone-deep",
        "EXAONE Deep 2.4B reasoning model from LG AI Research.",
    ),
    (
        "exaone-deep:7.8b",
        "exaone-deep",
        "EXAONE Deep 7.8B reasoning model for stronger complex reasoning.",
    ),
    (
        "exaone-deep:32b",
        "exaone-deep",
        "EXAONE Deep 32B reasoning model for high-capability local inference.",
    ),
    (
        "qwen3:0.6b",
        "qwen3",
        "Qwen3 0.6B open model in the Qwen3 dense and MoE family.",
    ),
    (
        "qwen3:1.7b",
        "qwen3",
        "Qwen3 1.7B open model in the Qwen3 dense and MoE family.",
    ),
    (
        "qwen3:4b",
        "qwen3",
        "Qwen3 4B open model in the Qwen3 dense and MoE family.",
    ),
    (
        "qwen3:8b",
        "qwen3",
        "Qwen3 8B open model in the Qwen3 dense and MoE family.",
    ),
    (
        "qwen3:14b",
        "qwen3",
        "Qwen3 14B open model in the Qwen3 dense and MoE family.",
    ),
    (
        "qwen3:30b",
        "qwen3",
        "Qwen3 30B open model in the Qwen3 dense and MoE family.",
    ),
    (
        "qwen3:32b",
        "qwen3",
        "Qwen3 32B open model in the Qwen3 dense and MoE family.",
    ),
    (
        "qwen3:235b",
        "qwen3",
        "Qwen3 235B flagship open model in the Qwen3 dense and MoE family.",
    ),
    (
        "ministral3:3b",
        "ministral-3",
        "Mistral Ministral 3B open model for compact multilingual and agentic workloads.",
    ),
    (
        "ministral3:8b",
        "ministral-3",
        "Mistral Ministral 8B open model for stronger multilingual and agentic workloads.",
    ),
    (
        "ministral3:14b",
        "ministral-3",
        "Mistral Ministral 14B open model for high-capability multilingual and agentic workloads.",
    ),
    (
        "gpt-oss:20b",
        "gpt-oss",
        "OpenAI gpt-oss-20b open-weight model for local reasoning and tool usage.",
    ),
    (
        "gpt-oss:120b",
        "gpt-oss",
        "OpenAI gpt-oss-120b open-weight model for advanced local reasoning and agentic tasks.",
    ),
)
LOCAL_MODEL_CATALOG_NAMES = {name for name, _, _ in LOCAL_MODEL_CATALOG}


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
        local_models = await self.list_local_model_cards(
            selected_models=(
                snapshot.clinical_model,
                snapshot.text_extraction_model,
            )
        )
        return self.build_response(snapshot=snapshot, local_models=local_models)

    # -------------------------------------------------------------------------
    async def update_state(
        self,
        payload: ModelConfigUpdateRequest = Body(...),
    ) -> ModelConfigStateResponse:
        snapshot = self.ensure_defaults()
        local_models = await self.list_local_model_cards(
            selected_models=(
                snapshot.clinical_model,
                snapshot.text_extraction_model,
            )
        )
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
                detail="No model catalog entries are available.",
            )
        if model_name not in local_model_names:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Model '{model_name}' is not supported for role '{role_name}'.",
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
    async def list_available_ollama_models(self) -> set[str]:
        try:
            async with OllamaClient() as client:
                models = await client.list_models()
        except Exception as exc:
            if isinstance(exc, (OllamaTimeout, OllamaError)):
                logger.warning("Unable to list local Ollama models: %s", exc)
                return set()
            logger.exception("Unexpected error while listing local Ollama models")
            return set()
        return {
            model.strip()
            for model in models
            if isinstance(model, str) and model.strip()
        }

    # -------------------------------------------------------------------------
    async def list_local_model_cards(
        self,
        *,
        selected_models: Iterable[str | None] = (),
    ) -> list[LocalModelCard]:
        available_models = await self.list_available_ollama_models()
        cards = [
            LocalModelCard(
                name=name,
                family=family,
                description=description,
                available_in_ollama=name in available_models,
            )
            for name, family, description in LOCAL_MODEL_CATALOG
        ]

        selected_candidates = {
            candidate.strip()
            for candidate in selected_models
            if isinstance(candidate, str) and candidate.strip()
        }

        extra_models = sorted(
            (available_models | selected_candidates) - LOCAL_MODEL_CATALOG_NAMES,
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
