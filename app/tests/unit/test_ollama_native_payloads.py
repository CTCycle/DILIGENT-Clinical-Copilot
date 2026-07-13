from __future__ import annotations

import asyncio
import inspect

from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.ollama_client import OllamaClient
import services.llm.ollama_chat as ollama_chat

###############################################################################
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

###############################################################################
def test_prepare_generation_parameters_clamps_temperature_and_strips_options(
    monkeypatch,
) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "is_ollama_reasoning_enabled",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "get_ollama_seed",
        classmethod(lambda cls: None),
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

###############################################################################
def test_ensure_context_option_preserves_explicit_num_ctx_and_computes_when_absent(
    monkeypatch,
) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")

    explicit = asyncio.run(
        client.ensure_context_option(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "probe"}],
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
            options={"num_predict": 4},
        )
    )
    assert computed == {"num_predict": 4, "num_ctx": 2048}
    asyncio.run(client.close())

###############################################################################
def test_calculate_context_window_respects_model_context_limit(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    monkeypatch.setattr(client, "estimate_tokens", lambda text: 5_000)

    async def fake_feasible(*a, **kw):
        return None

    monkeypatch.setattr(client, "estimate_max_feasible_context", fake_feasible)

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

###############################################################################
def test_ollama_generate_prompt_helpers_are_removed() -> None:
    assert not hasattr(OllamaClient, "build_generate_payload")
    assert not hasattr(OllamaClient, "messages_to_prompt")

###############################################################################
def test_ollama_chat_module_has_no_generate_fallback_helpers() -> None:
    source = inspect.getsource(ollama_chat)

    assert "build_generate_payload" not in source
    assert "messages_to_prompt" not in source
    assert "/api/generate" not in source

###############################################################################
def test_hardware_aware_context_uses_full(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")

    llama_metadata = {
        "model_info": {
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        },
        "details": {"parameter_size": "8B"},
    }

    native_limit = 32768
    monkeypatch.setattr(
        client,
        "extract_footprint_from_payload",
        lambda m: (8_000_000_000, 6_000_000_000),
    )
    monkeypatch.setattr(ollama_chat, "get_available_vram_bytes", lambda: 24_000_000_000)
    monkeypatch.setattr(ollama_chat, "get_available_memory_bytes", lambda: 0)

    async def fake_show(name: str) -> dict:
        return {**llama_metadata, "context_length": native_limit}

    monkeypatch.setattr(client, "show_model", fake_show)
    monkeypatch.setattr(client, "get_model_context_limit", lambda n: native_limit)

    value = asyncio.run(
        client.calculate_context_window(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert value == native_limit
    asyncio.run(client.close())

###############################################################################
def test_hardware_aware_context_scales_down(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")

    llama_metadata = {
        "model_info": {
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        },
        "details": {"parameter_size": "8B"},
    }

    kv_per_token = ollama_chat.estimate_kv_cache_bytes_per_token(llama_metadata)
    native_limit = 32768
    vram_footprint = 6_000_000_000
    available_vram = 8_000_000_000
    monkeypatch.setattr(
        client,
        "extract_footprint_from_payload",
        lambda m: (8_000_000_000, vram_footprint),
    )
    monkeypatch.setattr(ollama_chat, "get_available_vram_bytes", lambda: available_vram)
    monkeypatch.setattr(ollama_chat, "get_available_memory_bytes", lambda: 0)

    async def fake_show(name: str) -> dict:
        return {**llama_metadata, "context_length": native_limit}

    monkeypatch.setattr(client, "show_model", fake_show)
    monkeypatch.setattr(client, "get_model_context_limit", lambda n: native_limit)

    vram_budget = int(available_vram * ollama_chat.VRAM_SAFETY_RATIO)
    vram_for_kv = max(0, vram_budget - vram_footprint)
    max_by_vram = vram_for_kv // kv_per_token
    expected = min(native_limit, max_by_vram)
    assert expected < native_limit

    value = asyncio.run(
        client.calculate_context_window(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert value == expected
    asyncio.run(client.close())

###############################################################################
def test_hardware_aware_context_fallback(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    monkeypatch.setattr(client, "estimate_tokens", lambda text: 50)

    async def fake_show(name: str) -> dict:
        return {"model_info": {}}

    monkeypatch.setattr(client, "show_model", fake_show)

    async def fake_limit(name):
        return None

    monkeypatch.setattr(client, "get_model_context_limit", fake_limit)

    value = asyncio.run(
        client.calculate_context_window(
            model="unknown:latest",
            messages=[{"role": "user", "content": "short query"}],
        )
    )
    assert value == 4312
    asyncio.run(client.close())
