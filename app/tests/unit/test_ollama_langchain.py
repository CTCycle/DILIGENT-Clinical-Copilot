from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from services.llm import providers as providers_module
from services.llm.structured import StructuredOutputParser


###############################################################################
@dataclass
class FakeMessage:
    content: Any


###############################################################################
class FakeSchema(BaseModel):
    status: str


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
def test_chat_uses_langchain_backed_inference(monkeypatch) -> None:
    client = providers_module.OllamaClient(base_url="http://127.0.0.1:11434")
    _patch_generation_prep(monkeypatch, client)
    captured: dict[str, Any] = {}

    class FakeChatModel:
        async def ainvoke(self, messages: Any) -> FakeMessage:
            captured["messages"] = messages
            return FakeMessage(content='{"status":"ok"}')

        async def astream(self, messages: Any):
            _ = messages
            if False:
                yield None

    def fake_build_model(**kwargs: Any) -> FakeChatModel:
        captured["kwargs"] = kwargs
        return FakeChatModel()

    monkeypatch.setattr(client, "_build_ollama_chat_model", fake_build_model)
    result = asyncio.run(
        client.chat(model="llama3.1:8b", messages=[{"role": "user", "content": "hi"}])
    )
    assert result == {"status": "ok"}
    assert captured["kwargs"]["keep_alive"] == "10m"
    assert captured["kwargs"]["options"] == {"num_ctx": 2048}


# -----------------------------------------------------------------------------
def test_chat_stream_preserves_stream_behavior(monkeypatch) -> None:
    client = providers_module.OllamaClient(base_url="http://127.0.0.1:11434")
    _patch_generation_prep(monkeypatch, client)

    class FakeChatModel:
        async def ainvoke(self, messages: Any) -> FakeMessage:
            _ = messages
            return FakeMessage(content="")

        async def astream(self, messages: Any):
            _ = messages
            yield FakeMessage(content="chunk-1")
            yield FakeMessage(content="chunk-2")

    monkeypatch.setattr(
        client, "_build_ollama_chat_model", lambda **kwargs: FakeChatModel()
    )

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


# -----------------------------------------------------------------------------
def test_embed_uses_langchain_embeddings(monkeypatch) -> None:
    client = providers_module.OllamaClient(base_url="http://127.0.0.1:11434")

    async def fake_ready(model: str) -> None:
        _ = model
        return None

    monkeypatch.setattr(client, "ensure_model_ready", fake_ready)

    class FakeEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

    monkeypatch.setattr(
        client, "_build_ollama_embeddings_model", lambda **kwargs: FakeEmbeddings()
    )
    vectors = asyncio.run(client.embed(model="llama3.1:8b", input_texts=["a", "bb"]))
    assert vectors == [[1.0], [2.0]]


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
def test_ollama_inference_exception_maps_to_existing_error_types(monkeypatch) -> None:
    client = providers_module.OllamaClient(base_url="http://127.0.0.1:11434")
    _patch_generation_prep(monkeypatch, client)

    class FakeChatModel:
        async def ainvoke(self, messages: Any) -> FakeMessage:
            _ = messages
            raise TimeoutError("timeout")

        async def astream(self, messages: Any):
            _ = messages
            if False:
                yield None

    monkeypatch.setattr(
        client, "_build_ollama_chat_model", lambda **kwargs: FakeChatModel()
    )
    try:
        asyncio.run(
            client.chat(
                model="llama3.1:8b", messages=[{"role": "user", "content": "x"}]
            )
        )
        assert False, "Expected timeout mapping"
    except providers_module.OllamaTimeout:
        pass

