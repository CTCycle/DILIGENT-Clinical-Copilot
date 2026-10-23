# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from services.llm import ollama_chat
from services.llm.generation_policy import GenerationPurpose
from services.llm.runtime_config import LLMRuntimeConfig
import time
from unittest.mock import AsyncMock
from services.llm.ollama_client import OllamaClient
from services.llm.ollama_runtime import OllamaError
from services.llm import ollama_structured

def test_ollama_gpt_oss_preserves_level_reasoning_and_omits_temperature(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "get_ollama_seed",
        classmethod(lambda cls: None),
    )

    temperature, think, options = ollama_chat.prepare_generation_parameters(
        object(),
        model="gpt-oss:20b",
        purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
        temperature=0.7,
        think="high",
        options={"temperature": 0.7},
    )

    assert temperature is None
    assert think == "high"
    assert options is None

def test_ollama_context_window_is_task_sized_and_intersects_runtime_capacity() -> None:

    ###############################################################################
    class FakeOllama:

        # -------------------------------------------------------------------------
        @staticmethod
        def estimate_tokens(text: str) -> int:
            return max(1, len(text.split()))

        # -------------------------------------------------------------------------
        async def get_model_context_limit(self, model: str) -> int:
            assert model == "qwen3:8b"
            return 8192

        # -------------------------------------------------------------------------
        async def estimate_max_feasible_context(self, model: str) -> int:
            assert model == "qwen3:8b"
            return 4096

    context_window = asyncio.run(
        ollama_chat.calculate_context_window(
            FakeOllama(),
            model="qwen3:8b",
            messages=[{"role": "user", "content": "clinical evidence " * 20}],
            purpose=GenerationPurpose.TIMELINE_EXTRACTION,
            min_ctx=2048,
        )
    )

    assert context_window is not None
    assert 2048 <= context_window <= 4096

def test_ollama_payload_omits_think_when_transport_does_not_support_reasoning() -> None:

    ###############################################################################
    class FakeOllama:

        # -------------------------------------------------------------------------
        @staticmethod
        def compose_payload(
            payload: dict[str, object],
            *,
            format: str | None,
            options: dict[str, object] | None,
            keep_alive: str | None,
        ) -> dict[str, object]:
            return ollama_chat.compose_payload(
                payload,
                format=format,
                options=options,
                keep_alive=keep_alive,
            )

    payload = ollama_chat.build_chat_payload(
        FakeOllama(),
        model="unknown-model",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        format=None,
        temperature=None,
        think=None,
        options=None,
        keep_alive=None,
    )

    assert "think" not in payload

def test_ollama_chat_payload_preserves_json_format_and_thinking() -> None:

    ###############################################################################
    class FakeOllama:

        # -------------------------------------------------------------------------
        @staticmethod
        def compose_payload(
            payload: dict[str, object],
            *,
            format: str | None,
            options: dict[str, object] | None,
            keep_alive: str | None,
        ) -> dict[str, object]:
            return ollama_chat.compose_payload(
                payload,
                format=format,
                options=options,
                keep_alive=keep_alive,
            )

    payload = ollama_chat.build_chat_payload(
        FakeOllama(),
        model="qwen3:8b",
        messages=[{"role": "user", "content": "return JSON"}],
        stream=False,
        format="json",
        temperature=None,
        think="high",
        options=None,
        keep_alive=None,
    )

    assert payload["format"] == "json"
    assert payload["think"] == "high"

def test_evaluate_dual_residency_plan_checks_ram_and_vram(monkeypatch) -> None:
    gib = 1_073_741_824
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "get_clinical_model",
        classmethod(lambda cls: "clinical-model"),
    )
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "get_text_extraction_model",
        classmethod(lambda cls: "text-model"),
    )
    monkeypatch.setattr(
        OllamaClient,
        "get_available_memory_bytes",
        staticmethod(lambda: 20 * gib),
    )
    monkeypatch.setattr(
        OllamaClient,
        "get_available_vram_bytes",
        staticmethod(lambda: 16 * gib),
    )
    monkeypatch.setattr(client, "list_running_models", AsyncMock(return_value={}))
    monkeypatch.setattr(
        client,
        "get_model_footprint_bytes",
        AsyncMock(side_effect=[(5 * gib, 4 * gib), (4 * gib, 3 * gib)]),
    )

    plan = asyncio.run(client.evaluate_dual_residency_plan())
    asyncio.run(client.close())

    assert plan["dual_residency"] is True
    assert plan["required_ram"] == 9 * gib
    assert plan["required_vram"] == 7 * gib

def test_predict_next_target_model_prefers_frequent_transition() -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    now = time.monotonic()
    client.residency_usage_window_s = 600.0
    client.residency_transition_window_s = 120.0
    client.residency_usage_history.extend(
        [
            (now - 50, "clinical-model"),
            (now - 45, "text-model"),
            (now - 40, "clinical-model"),
            (now - 35, "text-model"),
            (now - 20, "clinical-model"),
        ]
    )

    predicted = client.predict_next_target_model(
        current_model="clinical-model",
        target_models=["clinical-model", "text-model"],
    )
    asyncio.run(client.close())

    assert predicted == "text-model"

def test_resolve_policy_keep_alive_uses_dual_setting(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    client.residency_dual_keep_alive = "8h"
    monkeypatch.setattr(
        client,
        "get_cached_residency_plan",
        AsyncMock(
            return_value={
                "models": ["clinical-model", "text-model"],
                "dual_residency": True,
            }
        ),
    )

    keep_alive = asyncio.run(
        client.resolve_policy_keep_alive(
            active_model="clinical-model",
            requested_keep_alive=None,
        )
    )
    asyncio.run(client.close())

    assert keep_alive == "8h"

def test_maybe_prefetch_target_model_prefetches_only_when_dual_safe(
    monkeypatch,
) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    started: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client,
        "get_cached_residency_plan",
        AsyncMock(
            return_value={
                "models": ["clinical-model", "text-model"],
                "dual_residency": True,
                "available_ram": 20,
                "required_ram": 10,
                "available_vram": 16,
                "required_vram": 8,
            }
        ),
    )

    async def fake_prefetch_model(*, model: str, keep_alive: str) -> None:
        started.append((model, keep_alive))

    monkeypatch.setattr(client, "prefetch_model", fake_prefetch_model)

    async def run() -> None:
        await client.maybe_prefetch_target_model(active_model="clinical-model")
        await asyncio.sleep(0)

    asyncio.run(run())
    asyncio.run(client.close())

    assert started == [("text-model", client.residency_dual_keep_alive)]

def test_maybe_prefetch_target_model_skips_when_dual_unsafe(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    started: list[str] = []
    monkeypatch.setattr(
        client,
        "get_cached_residency_plan",
        AsyncMock(
            return_value={
                "models": ["clinical-model", "text-model"],
                "dual_residency": False,
                "available_ram": 4,
                "required_ram": 10,
                "available_vram": 2,
                "required_vram": 8,
            }
        ),
    )

    async def fake_prefetch_model(*, model: str, keep_alive: str) -> None:
        _ = keep_alive
        started.append(model)

    monkeypatch.setattr(client, "prefetch_model", fake_prefetch_model)

    asyncio.run(client.maybe_prefetch_target_model(active_model="clinical-model"))
    asyncio.run(client.close())

    assert started == []
    assert client.residency_usage_history[-1][1] == "clinical-model"

def test_prefetch_model_oom_is_non_fatal(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    monkeypatch.setattr(client, "ensure_model_ready", AsyncMock(return_value=None))

    ###############################################################################
    class FakeHttpClient:

        # -------------------------------------------------------------------------
        async def post(self, path: str, json: dict) -> object:
            assert path == "/api/chat"
            assert json["model"] == "text-model"
            return object()

        # -------------------------------------------------------------------------
        async def aclose(self) -> None:
            return None

    client.client = FakeHttpClient()  # type: ignore[assignment]
    monkeypatch.setattr(
        client,
        "raise_for_status",
        lambda response: (_ for _ in ()).throw(
            OllamaError(
                "Ollama HTTP 500: llama-server reported out-of-memory during startup"
            )
        ),
    )

    asyncio.run(client.prefetch_model(model="text-model", keep_alive="30m"))
    asyncio.run(client.close())

def test_resolve_text_extraction_models_prefers_live_installed_models() -> None:

    ###############################################################################
    class FakeClient:
        default_model = "gpt-4.1-mini"

        # -------------------------------------------------------------------------
        async def get_cached_models(self) -> set[str]:
            return {"qwen3.5:2b", "qwen3.5:9b"}

    models = asyncio.run(
        ollama_structured.resolve_text_extraction_models(FakeClient(), "gpt-4.1-mini")
    )

    assert models == ["qwen3.5:2b", "qwen3.5:9b"]

def test_looks_like_schema_echo_detects_schema_payload() -> None:
    schema_like = (
        '{"$defs":{"Item":{"properties":{"name":{"type":"string"}}}},'
        '"title":"Payload","type":"object","required":["entries"]}'
    )

    assert ollama_structured.looks_like_schema_echo(schema_like) is True

def test_parse_with_repairs_uses_compact_repair_messages_for_schema_echo() -> None:

    ###############################################################################
    class FakeParser:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.calls = 0

        # -------------------------------------------------------------------------
        def parse(self, text: str) -> dict[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("schema echo")
            return {"ok": text}

    ###############################################################################
    class FakeClient:
        captured_messages: list[dict[str, str]] | None = None

        # -------------------------------------------------------------------------
        async def chat(self, **kwargs):
            self.captured_messages = kwargs["messages"]
            return '{"entries":[]}'

        build_repair_messages = staticmethod(ollama_structured.build_repair_messages)
        build_compact_repair_messages = staticmethod(
            ollama_structured.build_compact_repair_messages
        )
        _coerce_llm_text = staticmethod(ollama_structured._coerce_llm_text)

    parser = FakeParser()
    client = FakeClient()

    result = asyncio.run(
        ollama_structured.parse_with_repairs(
            client,
            parser=parser,
            text='{"$defs":{"Item":{"properties":{"name":{"type":"string"}}}},"title":"Payload","type":"object","required":["entries"]}',
            active_model="qwen3.5:2b",
            system_prompt="ignored",
            format_instructions="ignored",
            use_json_mode=True,
            max_repair_attempts=1,
        )
    )

    assert result == {"ok": '{"entries":[]}'}
    assert client.captured_messages is not None
    assert "schema or wrapper instead of the requested data" in client.captured_messages[1]["content"]

def test_chat_structured_model_forwards_generation_purpose() -> None:

    ###############################################################################
    class FakeClient:
        captured_purpose: GenerationPurpose | None = None

        # -------------------------------------------------------------------------
        async def chat(self, **kwargs):
            self.captured_purpose = kwargs["purpose"]
            return '{"ok": true}'

    client = FakeClient()

    result = asyncio.run(
        ollama_structured._chat_structured_model(
            client,
            active_model="gpt-oss:20b",
            messages=[],
            use_json_mode=True,
            temperature=0.0,
            purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
        )
    )

    assert result == '{"ok": true}'
    assert client.captured_purpose is GenerationPurpose.STRUCTURED_EXTRACTION

def test_ollama_client_forwards_generation_purpose_to_structured_models(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_call(_client, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        ollama_structured,
        "call_with_structured_models",
        fake_call,
    )

    client = OllamaClient.__new__(OllamaClient)
    result = asyncio.run(
        client.call_with_structured_models(
            parser=None,
            messages=[],
            system_prompt="",
            format_instructions="",
            preferred=["gpt-oss:20b"],
            temperature=0.0,
            use_json_mode=True,
            purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
            max_repair_attempts=1,
        )
    )

    assert result == {"ok": True}
    assert captured["purpose"] is GenerationPurpose.STRUCTURED_EXTRACTION

def test_ollama_client_forwards_generation_purpose_to_structured_chat(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_chat(_client, **kwargs):
        captured.update(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr(ollama_structured, "_chat_structured_model", fake_chat)

    client = OllamaClient.__new__(OllamaClient)
    result = asyncio.run(
        client._chat_structured_model(
            active_model="gpt-oss:20b",
            messages=[],
            use_json_mode=True,
            temperature=0.0,
            purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
        )
    )

    assert result == '{"ok": true}'
    assert captured["purpose"] is GenerationPurpose.STRUCTURED_EXTRACTION
