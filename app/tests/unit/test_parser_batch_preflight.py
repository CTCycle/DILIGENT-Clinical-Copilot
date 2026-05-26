from __future__ import annotations

import asyncio

from services.session.preflight import (
    LocalModelBatchPreflightResult,
    check_parser_batch_capacity,
)


class _FakeClient:
    def __init__(self, *, online: bool = True, models: list[str] | None = None) -> None:
        self._online = online
        self._models = models or []

    async def is_server_online(self) -> bool:
        return self._online

    async def list_models(self) -> list[str]:
        return self._models

    async def get_cached_residency_plan(self, *, force_refresh: bool = False):
        return {"ok": True, "force_refresh": force_refresh}

    async def close(self) -> None:
        return None


def test_batch_preflight_allows_non_local_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.session.preflight.LLMRuntimeConfig.resolve_provider_and_model",
        lambda purpose: ("openai", "gpt-4.1-mini"),
    )
    result = asyncio.run(check_parser_batch_capacity(task_count=2))
    assert isinstance(result, LocalModelBatchPreflightResult)
    assert result.concurrency_allowed is True
    assert result.provider == "openai"


def test_batch_preflight_denies_unavailable_local_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.session.preflight.LLMRuntimeConfig.resolve_provider_and_model",
        lambda purpose: ("ollama", "qwen3:14b"),
    )
    monkeypatch.setattr(
        "services.session.preflight.select_llm_provider",
        lambda **kwargs: _FakeClient(online=True, models=["llama3.1:8b"]),
    )
    result = asyncio.run(check_parser_batch_capacity(task_count=2))
    assert result.concurrency_allowed is False
    assert result.reason is not None


def test_batch_preflight_allows_available_local_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.session.preflight.LLMRuntimeConfig.resolve_provider_and_model",
        lambda purpose: ("ollama", "qwen3:14b"),
    )
    monkeypatch.setattr(
        "services.session.preflight.select_llm_provider",
        lambda **kwargs: _FakeClient(online=True, models=["qwen3:14b"]),
    )
    result = asyncio.run(check_parser_batch_capacity(task_count=2))
    assert result.concurrency_allowed is True
