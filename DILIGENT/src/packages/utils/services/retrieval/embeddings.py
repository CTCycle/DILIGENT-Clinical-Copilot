# --- file: embeddings.py (replace entire file) ---
from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any, Literal, cast

from DILIGENT.src.app.backend.models.providers import CloudLLMClient, OllamaClient
from DILIGENT.src.packages.configurations import configurations
from DILIGENT.src.packages.constants import VECTOR_DB_PATH
from DILIGENT.src.packages.logger import logger
from DILIGENT.src.packages.utils.repository.vectors import LanceVectorDatabase

RAG_SETTINGS = configurations.rag

ProviderName = Literal["openai", "azure-openai", "anthropic", "gemini"]
EmbeddingBackend = Literal["ollama", "cloud"]
ALLOWED_PROVIDERS: tuple[ProviderName, ...] = (
    "openai",
    "azure-openai",
    "anthropic",
    "gemini",
)


###############################################################################
class EmbeddingGenerator:
    def __init__(
        self,
        *,
        backend: str,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,        
        use_cloud_embeddings: bool = False,
        cloud_provider: str | None = None,
        cloud_embedding_model: str | None = None,
    ) -> None:
        normalized_backend = backend.lower().strip() if backend else "ollama"
        # Keep parameter flexible, but store as the constrained Literal type.
        self.backend: EmbeddingBackend = (
            "cloud" if use_cloud_embeddings else cast(EmbeddingBackend, normalized_backend)
        )

        self.ollama_model: str | None = None
        self.ollama_base_url: str | None = None
        self.cloud_provider: ProviderName | None = None
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

            provider_normalized = cloud_provider.lower().strip()
            if provider_normalized not in ALLOWED_PROVIDERS:
                raise ValueError(
                    f"Unsupported cloud provider: {cloud_provider}. "
                    f"Allowed: {', '.join(ALLOWED_PROVIDERS)}"
                )
            # Safe cast after validation against ALLOWED_PROVIDERS.
            self.cloud_provider = cast(ProviderName, provider_normalized)
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
        self, texts: list[str], provider: ProviderName, model: str
    ) -> list[list[float]]:
        async with CloudLLMClient(
            provider=provider, default_model=model
        ) as client:
            return await client.embed(model=model, input_texts=texts)


###############################################################################
class SimilaritySearch:
    def __init__(
        self,
        *,
        vector_database: LanceVectorDatabase | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        default_top_k: int = RAG_SETTINGS.top_k_documents,
    ) -> None:
        self.default_top_k = max(int(default_top_k), 1)
        self.vector_database = vector_database or LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=RAG_SETTINGS.vector_collection_name,
            metric=RAG_SETTINGS.vector_index_metric,
            index_type=RAG_SETTINGS.vector_index_type,
        )
        self.embedding_generator = embedding_generator or EmbeddingGenerator(
            backend=RAG_SETTINGS.embedding_backend,
            ollama_base_url=RAG_SETTINGS.ollama_base_url,
            ollama_model=RAG_SETTINGS.ollama_embedding_model,
            use_cloud_embeddings=RAG_SETTINGS.use_cloud_embeddings,
            cloud_provider=RAG_SETTINGS.cloud_provider,
            cloud_embedding_model=RAG_SETTINGS.cloud_embedding_model,
        )
        try:
            self.vector_database.initialize(False)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize vector database: %s", exc)

    # -------------------------------------------------------------------------
    def search(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        if not isinstance(query, str):
            return []
        normalized = query.strip()
        if not normalized:
            return []
        limit = self.default_top_k if top_k is None else max(int(top_k), 1)
        embeddings = self.embedding_generator.embed_texts([normalized])
        if not embeddings:
            return []
        try:
            table = self.vector_database.get_table()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to access LanceDB table: %s", exc)
            return []
        try:
            results = table.search(embeddings[0]).limit(limit).to_list()
        except Exception as exc:  # noqa: BLE001
            logger.error("Similarity search failed: %s", exc)
            return []
        documents: list[dict[str, Any]] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if isinstance(text, str):
                resolved_text = text.strip()
            elif text is None:
                resolved_text = ""
            else:
                resolved_text = str(text)
            metadata_raw = entry.get("metadata")
            metadata: dict[str, Any] = {}
            if isinstance(metadata_raw, dict):
                metadata = metadata_raw
            elif isinstance(metadata_raw, str):
                try:
                    parsed = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    metadata = {}
                else:
                    if isinstance(parsed, dict):
                        metadata = parsed
            distance_value: float | None = None
            distance = entry.get("_distance")
            if isinstance(distance, (int, float)):
                distance_value = float(distance)
            documents.append(
                {
                    "document_id": entry.get("document_id"),
                    "chunk_id": entry.get("chunk_id"),
                    "text": resolved_text,
                    "source": entry.get("source"),
                    "metadata": metadata,
                    "distance": distance_value,
                }
            )
        return documents


__all__ = ["EmbeddingGenerator", "SimilaritySearch"]
