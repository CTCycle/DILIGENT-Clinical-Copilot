from __future__ import annotations

import asyncio

from pytest import MonkeyPatch
import services.retrieval.embeddings as embeddings_module

###############################################################################
def test_openai_embedding_provider_selection(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        embeddings_module.CloudEmbeddingGenerator,
        "resolve_provider_access_key",
        staticmethod(lambda provider: "openai-key"),
    )
    provider = embeddings_module.select_embedding_provider(
        backend="cloud",
        cloud_provider="openai",
        cloud_embedding_model="text-embedding-3-large",
    )
    assert isinstance(provider, embeddings_module.CloudEmbeddingGenerator)
    assert provider.provider == "openai"

###############################################################################
def test_gemini_embedding_provider_selection(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        embeddings_module.CloudEmbeddingGenerator,
        "resolve_provider_access_key",
        staticmethod(lambda provider: "gemini-key"),
    )
    provider = embeddings_module.select_embedding_provider(
        backend="cloud",
        cloud_provider="gemini",
        cloud_embedding_model="gemini-embedding-001",
    )
    assert isinstance(provider, embeddings_module.CloudEmbeddingGenerator)
    assert provider.provider == "gemini"

###############################################################################
def test_ollama_embedding_provider_selection() -> None:
    provider = embeddings_module.select_embedding_provider(
        backend="ollama",
        ollama_model="nomic-embed-text",
    )
    assert isinstance(provider, embeddings_module.OllamaEmbeddingGenerator)

###############################################################################
def test_single_query_embedding_return_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        embeddings_module.CloudEmbeddingGenerator,
        "resolve_provider_access_key",
        staticmethod(lambda provider: "openai-key"),
    )

    ###############################################################################
    class FakeClient:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        # -------------------------------------------------------------------------
        async def embed(
            self, *, model: str, input_texts: list[str]
        ) -> list[list[float]]:
            _ = model, input_texts
            return [[1.0, 2.0] for _ in input_texts]

    monkeypatch.setattr(embeddings_module, "CloudLLMClient", FakeClient)
    generator = embeddings_module.CloudEmbeddingGenerator(
        provider="openai",
        model="text-embedding-3-small",
    )
    vector = asyncio.run(generator.embed_query("hello"))
    assert isinstance(vector, list)
    assert vector == [1.0, 2.0]

###############################################################################
def test_batch_embedding_return_shape_and_order_preserved(monkeypatch) -> None:

    ###############################################################################
    class FakeClient:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        # -------------------------------------------------------------------------
        async def embed(
            self, *, model: str, input_texts: list[str]
        ) -> list[list[float]]:
            _ = model
            return [[float(index)] for index, _ in enumerate(input_texts)]

    monkeypatch.setattr(embeddings_module, "OllamaClient", FakeClient)
    generator = embeddings_module.OllamaEmbeddingGenerator(model="nomic-embed-text")
    vectors = asyncio.run(generator.embed_texts(["first", "second", "third"]))
    assert vectors == [[0.0], [1.0], [2.0]]

###############################################################################
def test_provider_validation_and_exception_mapping(monkeypatch) -> None:
    try:
        embeddings_module.select_embedding_provider(
            backend="cloud",
            cloud_provider="unsupported-provider",
            cloud_embedding_model="x",
        )
        assert False, "Expected provider validation failure"
    except ValueError:
        pass

    monkeypatch.setattr(
        embeddings_module.CloudEmbeddingGenerator,
        "resolve_provider_access_key",
        staticmethod(lambda provider: "openai-key"),
    )

    ###############################################################################
    class FailingClient:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        # -------------------------------------------------------------------------
        async def __aenter__(self):
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        # -------------------------------------------------------------------------
        async def embed(
            self, *, model: str, input_texts: list[str]
        ) -> list[list[float]]:
            _ = model, input_texts
            raise TimeoutError("timeout")

    monkeypatch.setattr(embeddings_module, "CloudLLMClient", FailingClient)
    generator = embeddings_module.CloudEmbeddingGenerator(
        provider="openai",
        model="text-embedding-3-small",
    )
    try:
        asyncio.run(generator.embed_texts(["x"]))
        assert False, "Expected timeout mapping"
    except embeddings_module.LLMTimeout:
        pass
