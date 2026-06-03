from __future__ import annotations

import asyncio

from configurations.llm_configs import LLMRuntimeConfig
from services.llm.ollama_client import OllamaClient


def test_build_chat_payload_includes_optional_fields() -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    payload = client.build_chat_payload(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        format="json",
        temperature=0.25,
        think=True,
        options={"num_ctx": 4096, "num_predict": 0},
        keep_alive="10m",
    )

    assert payload["model"] == "llama3.1:8b"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["stream"] is False
    assert payload["temperature"] == 0.25
    assert payload["think"] is True
    assert payload["format"] == "json"
    assert payload["options"] == {"num_ctx": 4096, "num_predict": 0}
    assert payload["keep_alive"] == "10m"
    asyncio.run(client.close())


def test_prepare_generation_parameters_clamps_temperature_and_strips_options(
    monkeypatch,
) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "is_ollama_reasoning_enabled",
        classmethod(lambda cls: False),
    )

    temperature, think, options = client.prepare_generation_parameters(
        temperature=None,
        think=None,
        options={"temperature": 3.9, "num_predict": 4},
    )

    assert temperature == 2.0
    assert think is False
    assert options == {"num_predict": 4}
    asyncio.run(client.close())


def test_ensure_context_option_preserves_explicit_num_ctx_and_computes_when_absent(
    monkeypatch,
) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")

    explicit = asyncio.run(
        client.ensure_context_option(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "probe"}],
            prompt=None,
            options={"num_ctx": 1024, "temperature": 0.1},
        )
    )
    assert explicit == {"num_ctx": 1024, "temperature": 0.1}

    async def fake_window(**kwargs):
        _ = kwargs
        return 2048

    monkeypatch.setattr(client, "calculate_context_window", fake_window)
    computed = asyncio.run(
        client.ensure_context_option(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "probe"}],
            prompt=None,
            options={"num_predict": 4},
        )
    )
    assert computed == {"num_predict": 4, "num_ctx": 2048}
    asyncio.run(client.close())


def test_calculate_context_window_respects_model_context_limit(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    monkeypatch.setattr(client, "estimate_tokens", lambda text: 5_000)

    async def fake_limit(name: str) -> int | None:
        _ = name
        return 2_048

    monkeypatch.setattr(client, "get_model_context_limit", fake_limit)

    value = asyncio.run(
        client.calculate_context_window(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "probe"}],
        )
    )
    assert value == 2_048
    asyncio.run(client.close())
