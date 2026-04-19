from __future__ import annotations

from contextlib import contextmanager
from typing import TypedDict

from DILIGENT.server.configurations.llm_configs import LLMRuntimeConfig


###############################################################################
class RuntimeSnapshot(TypedDict):
    use_cloud_services: bool
    llm_provider: str
    cloud_model: str
    parsing_model: str
    clinical_model: str
    ollama_temperature: float
    cloud_temperature: float
    ollama_reasoning: bool


###############################################################################
class RuntimeOverrides:
    @staticmethod
    def capture_snapshot() -> RuntimeSnapshot:
        return {
            "use_cloud_services": bool(LLMRuntimeConfig.is_cloud_enabled()),
            "llm_provider": str(LLMRuntimeConfig.get_llm_provider()),
            "cloud_model": str(LLMRuntimeConfig.get_cloud_model()),
            "parsing_model": str(LLMRuntimeConfig.get_parsing_model()),
            "clinical_model": str(LLMRuntimeConfig.get_clinical_model()),
            "ollama_temperature": float(LLMRuntimeConfig.get_ollama_temperature()),
            "cloud_temperature": float(LLMRuntimeConfig.get_cloud_temperature()),
            "ollama_reasoning": bool(LLMRuntimeConfig.is_ollama_reasoning_enabled()),
        }

    @staticmethod
    def apply(
        *,
        use_cloud_services: bool | None,
        llm_provider: str | None,
        cloud_model: str | None,
        parsing_model: str | None,
        clinical_model: str | None,
        ollama_temperature: float | None,
        cloud_temperature: float | None,
        ollama_reasoning: bool | None,
    ) -> None:
        if use_cloud_services is not None:
            LLMRuntimeConfig.set_use_cloud_services(use_cloud_services)
        if llm_provider is not None:
            LLMRuntimeConfig.set_llm_provider(llm_provider)
        if cloud_model is not None:
            LLMRuntimeConfig.set_cloud_model(cloud_model)
        if parsing_model is not None:
            LLMRuntimeConfig.set_parsing_model(parsing_model)
        if clinical_model is not None:
            LLMRuntimeConfig.set_clinical_model(clinical_model)
        if ollama_temperature is not None:
            LLMRuntimeConfig.set_ollama_temperature(ollama_temperature)
        if cloud_temperature is not None:
            LLMRuntimeConfig.set_cloud_temperature(cloud_temperature)
        if ollama_reasoning is not None:
            LLMRuntimeConfig.set_ollama_reasoning(ollama_reasoning)

    @classmethod
    @contextmanager
    def context(
        cls,
        *,
        use_cloud_services: bool | None,
        llm_provider: str | None,
        cloud_model: str | None,
        parsing_model: str | None,
        clinical_model: str | None,
        ollama_temperature: float | None,
        cloud_temperature: float | None,
        ollama_reasoning: bool | None,
    ):
        snapshot = cls.capture_snapshot()
        cls.apply(
            use_cloud_services=use_cloud_services,
            llm_provider=llm_provider,
            cloud_model=cloud_model,
            parsing_model=parsing_model,
            clinical_model=clinical_model,
            ollama_temperature=ollama_temperature,
            cloud_temperature=cloud_temperature,
            ollama_reasoning=ollama_reasoning,
        )
        try:
            yield
        finally:
            cls.apply(
                use_cloud_services=snapshot["use_cloud_services"],
                llm_provider=snapshot["llm_provider"],
                cloud_model=snapshot["cloud_model"],
                parsing_model=snapshot["parsing_model"],
                clinical_model=snapshot["clinical_model"],
                ollama_temperature=snapshot["ollama_temperature"],
                cloud_temperature=snapshot["cloud_temperature"],
                ollama_reasoning=snapshot["ollama_reasoning"],
            )
