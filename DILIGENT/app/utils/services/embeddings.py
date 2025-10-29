from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from DILIGENT.app.api.models.providers import CloudLLMClient, OllamaClient


###############################################################################
class EmbeddingGenerator:
    def __init__(
        self,
        *,
        backend: str,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        hf_model: str | None = None,
        use_cloud_embeddings: bool = False,
        cloud_provider: str | None = None,
        cloud_embedding_model: str | None = None,
    ) -> None:
        normalized_backend = backend.lower().strip() if backend else "ollama"
        self.backend = "cloud" if use_cloud_embeddings else normalized_backend

        self.ollama_model: str | None = None
        self.ollama_base_url: str | None = None        
        self.cloud_provider: str | None = None
        self.cloud_embedding_model: str | None = None

        if self.backend == "ollama":
            if not ollama_model:
                raise ValueError("Ollama embedding model is required")
            self.ollama_model = ollama_model
            self.ollama_base_url = ollama_base_url
        
        elif self.backend == "cloud":
            if not cloud_provider:
                raise ValueError("Cloud provider is required for embeddings")
            if not cloud_embedding_model:
                raise ValueError("Cloud embedding model is required")
            self.cloud_provider = cloud_provider
            self.cloud_embedding_model = cloud_embedding_model
        else:
            raise ValueError(f"Unsupported embedding backend: {backend}")

    # -------------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        sanitized = [text if text.strip() else " " for text in texts]
        if not sanitized:
            return []

        if self.backend == "ollama":
            if self.ollama_model is None:
                raise ValueError("Ollama embedding model is not configured")
            embeddings = self.run_async(
                self.embed_with_ollama(sanitized, self.ollama_model)
            )        
        elif self.backend == "cloud":
            if self.cloud_provider is None or self.cloud_embedding_model is None:
                raise ValueError("Cloud embedding configuration is not set")
            embeddings = self.run_async(
                self.embed_with_cloud(
                    sanitized,
                    self.cloud_provider,
                    self.cloud_embedding_model,
                )
            )
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported embedding backend: {self.backend}")

        return [[float(value) for value in vector] for vector in embeddings]

    # -------------------------------------------------------------------------
    @staticmethod
    def run_async(
        coroutine: Coroutine[Any, Any, list[list[float]]]
    ) -> list[list[float]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        if loop.is_running():
            try:
                previous_loop = asyncio.get_event_loop()
            except RuntimeError:
                previous_loop = None
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coroutine)
            finally:
                new_loop.close()
                asyncio.set_event_loop(previous_loop)
        return loop.run_until_complete(coroutine)

    # -------------------------------------------------------------------------
    async def embed_with_ollama(
        self, texts: list[str], model: str
    ) -> list[list[float]]:
        async with OllamaClient(
            base_url=self.ollama_base_url,
            default_model=model,
        ) as client:
            return await client.embed(model=model, input_texts=texts)

    # -------------------------------------------------------------------------
    async def embed_with_cloud(
        self, texts: list[str], provider: str, model: str
    ) -> list[list[float]]:
        async with CloudLLMClient(
            provider=provider, default_model=model
        ) as client:
            return await client.embed(model=model, input_texts=texts)


__all__ = ["EmbeddingGenerator"]
