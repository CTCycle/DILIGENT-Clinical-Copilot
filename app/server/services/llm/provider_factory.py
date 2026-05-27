from __future__ import annotations

from typing import Any, Literal

from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import get_server_settings
from services.llm.cloud import CloudLLMClient, LLMError
from services.llm.ollama_client import OllamaClient

ProviderName = Literal["openai", "gemini"]
RuntimePurpose = Literal["clinical", "parser"]


def select_llm_provider(
    provider: str = "ollama",
    **kwargs: Any,
) -> OllamaClient | CloudLLMClient:
    """Factory returning an LLM client with a unified interface."""
    p = provider.strip().lower()
    if p == "ollama":
        runtime_timeout = get_server_settings().runtime.default_llm_timeout
        return OllamaClient(
            base_url=kwargs.get("base_url"),
            timeout_s=kwargs.get("timeout_s", runtime_timeout),
            keepalive_connections=kwargs.get("keepalive_connections", 10),
            keepalive_max=kwargs.get("keepalive_max", 20),
            default_model=kwargs.get("default_model"),
        )
    if p in ("openai", "gemini"):
        runtime_timeout = get_server_settings().runtime.default_llm_timeout
        cloud_provider: ProviderName = "openai" if p == "openai" else "gemini"
        return CloudLLMClient(
            provider=cloud_provider,
            base_url=kwargs.get("base_url"),
            timeout_s=kwargs.get("timeout_s", runtime_timeout),
            keepalive_connections=kwargs.get("keepalive_connections", 10),
            keepalive_max=kwargs.get("keepalive_max", 20),
            default_model=kwargs.get("default_model"),
            max_retries=kwargs.get("max_retries", 2),
        )
    raise LLMError(f"Unknown or unsupported provider: {provider}")


def initialize_llm_client(
    *, purpose: RuntimePurpose = "clinical", **kwargs: Any
) -> OllamaClient | CloudLLMClient:
    kwargs.setdefault("timeout_s", get_server_settings().runtime.default_llm_timeout)
    provider, default_model = LLMRuntimeConfig.resolve_provider_and_model(purpose)
    if LLMRuntimeConfig.is_cloud_enabled():
        forced_provider = (LLMRuntimeConfig.get_llm_provider() or "").strip().lower()
        provider = forced_provider or provider
        forced_model = (LLMRuntimeConfig.get_cloud_model() or "").strip()
        if forced_model:
            default_model = forced_model
    selected_model = kwargs.pop("default_model", default_model)
    return select_llm_provider(
        provider=provider,
        default_model=selected_model,
        **kwargs,
    )
