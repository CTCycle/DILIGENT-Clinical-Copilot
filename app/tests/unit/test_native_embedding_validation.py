from __future__ import annotations

import asyncio

import services.llm.cloud as cloud_module
from services.llm.ollama_chat import normalize_embedding_payload
from services.llm.ollama_client import OllamaClient, OllamaError


def test_ollama_embedding_payload_returns_normalized_float_vectors(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    captured: dict[str, object] = {}

    async def fake_ready(model: str) -> None:
        _ = model
        return None

    class FakeResponse:
        def json(self) -> dict[str, object]:
            return {"embeddings": [[1, 2], ["3.5", 4]]}

    async def fake_post(path: str, json: dict[str, object]) -> FakeResponse:
        captured["path"] = path
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(client, "ensure_model_ready", fake_ready)
    monkeypatch.setattr(client, "raise_for_status", lambda resp: None)
    monkeypatch.setattr(client.client, "post", fake_post)

    vectors = asyncio.run(
        client.embed(model="nomic-embed-text", input_texts=["a", "bb"])
    )
    assert vectors == [[1.0, 2.0], [3.5, 4.0]]
    assert captured["path"] == "/api/embed"
    assert captured["json"] == {"model": "nomic-embed-text", "input": ["a", "bb"]}
    asyncio.run(client.close())


def test_ollama_embedding_payload_validation_errors() -> None:
    try:
        normalize_embedding_payload({}, 1)
        assert False, "Expected missing embeddings failure"
    except OllamaError:
        pass

    try:
        normalize_embedding_payload({"embeddings": [[1, "x"]]}, 1)
        assert False, "Expected non-numeric embeddings failure"
    except OllamaError:
        pass

    try:
        normalize_embedding_payload({"embeddings": [[1.0]]}, 2)
        assert False, "Expected embedding count mismatch failure"
    except OllamaError:
        pass


def test_openai_embedding_response_sorting_by_index_is_preserved(monkeypatch) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def close(self) -> None:
            return None

    monkeypatch.setattr(cloud_module, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        cloud_module.CloudLLMClient,
        "resolve_provider_access_key",
        lambda self, provider: "openai-key",
    )
    client = cloud_module.CloudLLMClient(
        provider="openai", default_model="text-embedding-3-small"
    )
    monkeypatch.setattr(client, "raise_for_status", lambda resp: None)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"index": 1, "embedding": [3, 4]},
                    {"index": 0, "embedding": [1, 2]},
                ]
            }

    async def fake_post(path: str, json: dict[str, object]) -> FakeResponse:
        _ = path, json
        return FakeResponse()

    monkeypatch.setattr(client.client, "post", fake_post)

    vectors = asyncio.run(
        client.embed_openai(
            model="text-embedding-3-small",
            input_texts=["first", "second"],
        )
    )
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    asyncio.run(client.close())


def test_gemini_embedding_response_count_mismatch_raises(monkeypatch) -> None:
    class FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeGeminiClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeGenAI:
        Client = FakeGeminiClient

    class FakeTypes:
        GenerateContentConfig = FakeGenerateContentConfig

    monkeypatch.setattr(cloud_module, "genai", FakeGenAI)
    monkeypatch.setattr(cloud_module, "genai_types", FakeTypes)
    monkeypatch.setattr(
        cloud_module.CloudLLMClient,
        "resolve_provider_access_key",
        lambda self, provider: "gemini-key",
    )
    client = cloud_module.CloudLLMClient(
        provider="gemini", default_model="gemini-2.5-pro"
    )
    monkeypatch.setattr(client, "raise_for_status", lambda resp: None)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embeddings": [{"values": [1, 2]}]}

    async def fake_post(path: str, json: dict[str, object]) -> FakeResponse:
        _ = path, json
        return FakeResponse()

    monkeypatch.setattr(client.client, "post", fake_post)

    try:
        asyncio.run(
            client.embed_gemini(
                model="gemini-2.5-pro",
                input_texts=["first", "second"],
            )
        )
        assert False, "Expected mismatch failure"
    except cloud_module.LLMError:
        pass
    asyncio.run(client.close())
