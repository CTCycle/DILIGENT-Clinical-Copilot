from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import services.llm.ollama_client as providers_module
from pydantic import BaseModel
from services.llm.structured import StructuredOutputParser

###############################################################################
@dataclass
class FakeResponse:
    payload: dict[str, Any]

    # -------------------------------------------------------------------------
    def json(self) -> dict[str, Any]:
        return self.payload

###############################################################################
class FakeSchema(BaseModel):
    status: str

###############################################################################
class FakeStreamResponse:

    # -------------------------------------------------------------------------
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    # -------------------------------------------------------------------------
    async def aiter_lines(self):
        for line in self._lines:
            yield line

###############################################################################
class FakeStreamContext:

    # -------------------------------------------------------------------------
    def __init__(self, response: FakeStreamResponse) -> None:
        self.response = response

    # -------------------------------------------------------------------------
    async def __aenter__(self) -> FakeStreamResponse:
        return self.response

    # -------------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

###############################################################################
def _patch_generation_prep(monkeypatch, client: providers_module.OllamaClient) -> None:
    async def fake_prepare_common_options(
        **kwargs: Any,
    ) -> tuple[str, float, bool, dict[str, Any] | None]:
        _ = kwargs
        return "llama3.1:8b", 0.4, True, {"num_ctx": 2048}

    async def fake_keep_alive(**kwargs: Any) -> str | None:
        _ = kwargs
        return "10m"

    async def fake_ready(model: str) -> None:
        _ = model
        return None

    async def fake_prefetch(active_model: str) -> None:
        _ = active_model
        return None

    monkeypatch.setattr(client, "prepare_common_options", fake_prepare_common_options)
    monkeypatch.setattr(client, "resolve_policy_keep_alive", fake_keep_alive)
    monkeypatch.setattr(client, "ensure_model_ready", fake_ready)
    monkeypatch.setattr(client, "maybe_prefetch_target_model", fake_prefetch)

###############################################################################
def test_chat_uses_native_ollama_chat_endpoint(monkeypatch) -> None:
    client = providers_module.OllamaClient(base_url="http://127.0.0.1:11434")
    _patch_generation_prep(monkeypatch, client)
    captured: dict[str, Any] = {}

    async def fake_post(path: str, json: dict[str, Any]) -> FakeResponse:
        captured["path"] = path
        captured["json"] = json
        return FakeResponse(
            {"message": {"role": "assistant", "content": '{"status":"ok"}'}}
        )

    monkeypatch.setattr(client.client, "post", fake_post)
    monkeypatch.setattr(client, "raise_for_status", lambda resp: None)

    result = asyncio.run(
        client.chat(model="llama3.1:8b", messages=[{"role": "user", "content": "hi"}])
    )

    assert result == {"status": "ok"}
    assert captured["path"] == "/api/chat"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["json"]["options"] == {"num_ctx": 2048}
    assert captured["json"]["keep_alive"] == "10m"
    assert captured["json"]["temperature"] == 0.4
    assert captured["json"]["think"] is True
    asyncio.run(client.close())

###############################################################################
def test_chat_stream_preserves_stream_behavior(monkeypatch) -> None:
    client = providers_module.OllamaClient(base_url="http://127.0.0.1:11434")
    _patch_generation_prep(monkeypatch, client)

    response = FakeStreamResponse(
        [
            '{"message":{"role":"assistant","content":"chunk-1"},"done":false}',
            '{"message":{"role":"assistant","content":"chunk-2"},"done":false}',
            '{"done":true}',
        ]
    )
    monkeypatch.setattr(
        client.client, "stream", lambda *args, **kwargs: FakeStreamContext(response)
    )
    monkeypatch.setattr(client, "raise_for_status", lambda resp: None)

    async def gather() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for evt in client.chat_stream(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "stream"}],
        ):
            events.append(evt)
        return events

    events = asyncio.run(gather())
    assert events[0]["done"] is False
    assert events[1]["done"] is False
    assert events[-1]["done"] is True
    assert events[-1]["message"]["content"] == "chunk-1chunk-2"
    asyncio.run(client.close())

###############################################################################
def test_embed_uses_native_ollama_embed_endpoint(monkeypatch) -> None:
    client = providers_module.OllamaClient(base_url="http://127.0.0.1:11434")

    async def fake_ready(model: str) -> None:
        _ = model
        return None

    captured: dict[str, Any] = {}

    async def fake_post(path: str, json: dict[str, Any]) -> FakeResponse:
        captured["path"] = path
        captured["json"] = json
        return FakeResponse({"embeddings": [[1, 2], [3, 4]]})

    monkeypatch.setattr(client, "ensure_model_ready", fake_ready)
    monkeypatch.setattr(client.client, "post", fake_post)
    monkeypatch.setattr(client, "raise_for_status", lambda resp: None)

    vectors = asyncio.run(client.embed(model="llama3.1:8b", input_texts=["a", "bb"]))
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert captured["path"] == "/api/embed"
    assert captured["json"] == {"model": "llama3.1:8b", "input": ["a", "bb"]}
    asyncio.run(client.close())

###############################################################################
def test_structured_output_repair_loop_still_works(monkeypatch) -> None:
    client = providers_module.OllamaClient(base_url="http://127.0.0.1:11434")
    parser = StructuredOutputParser(schema=FakeSchema)
    replies = iter(["not-json", '{"status":"ok"}'])

    async def fake_chat(**kwargs: Any) -> dict[str, Any] | str:
        _ = kwargs
        return next(replies)

    monkeypatch.setattr(client, "chat", fake_chat)
    parsed = asyncio.run(
        client.parse_with_repairs(
            parser=parser,
            text="not-json",
            active_model="llama3.1:8b",
            system_prompt="sys",
            format_instructions="fmt",
            use_json_mode=True,
            max_repair_attempts=2,
        )
    )
    assert parsed.status == "ok"
    asyncio.run(client.close())

###############################################################################
def test_ollama_native_timeout_maps_to_existing_error_type(monkeypatch) -> None:
    client = providers_module.OllamaClient(base_url="http://127.0.0.1:11434")
    _patch_generation_prep(monkeypatch, client)

    async def fake_post(path: str, json: dict[str, Any]) -> FakeResponse:
        _ = path, json
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(client.client, "post", fake_post)

    try:
        asyncio.run(
            client.chat(
                model="llama3.1:8b", messages=[{"role": "user", "content": "x"}]
            )
        )
        assert False, "Expected timeout mapping"
    except providers_module.OllamaTimeout:
        pass
    asyncio.run(client.close())
