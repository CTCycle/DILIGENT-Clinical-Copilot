from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Coroutine
from typing import Any, Literal, cast

import httpx
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from DILIGENT.server.configurations.startup import server_settings
from DILIGENT.server.common.constants import CLOUD_MODEL_CHOICES, VECTOR_DB_PATH
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.repositories.serialization.access_keys import AccessKeySerializer
from DILIGENT.server.services.llm.cloud import LLMError, LLMTimeout
from DILIGENT.server.services.llm.providers import OllamaError, OllamaTimeout
from DILIGENT.server.repositories.vectors import LanceVectorDatabase

ProviderName = Literal["openai", "gemini"]
EmbeddingBackend = Literal["ollama", "cloud"]


###############################################################################
def _build_openai_embeddings_model(
    *,
    api_key: str,
    model: str,
    timeout_s: float,
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=api_key,
        model=model,
        request_timeout=timeout_s,
    )


###############################################################################
def _build_gemini_embeddings_model(
    *,
    api_key: str,
    model: str,
    timeout_s: float,
) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        google_api_key=api_key,
        model=model,
        request_timeout=timeout_s,
    )


###############################################################################
def _build_ollama_embeddings_model(
    *,
    base_url: str,
    model: str,
    timeout_s: float,
) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=base_url,
        model=model,
        client_kwargs={"timeout": timeout_s},
    )


###############################################################################
def _map_langchain_embedding_exception(
    exc: Exception,
    *,
    provider: str,
) -> LLMError | OllamaError:
    if isinstance(exc, (LLMError, OllamaError)):
        return exc
    if isinstance(exc, TimeoutError):
        if provider == "ollama":
            return OllamaTimeout("Timed out requesting Ollama embeddings")
        return LLMTimeout("Timed out requesting cloud embeddings")
    if isinstance(exc, httpx.TimeoutException):
        if provider == "ollama":
            return OllamaTimeout("Timed out requesting Ollama embeddings")
        return LLMTimeout("Timed out requesting cloud embeddings")
    error_name = exc.__class__.__name__.lower()
    if "timeout" in error_name:
        if provider == "ollama":
            return OllamaTimeout("Timed out requesting Ollama embeddings")
        return LLMTimeout("Timed out requesting cloud embeddings")
    if provider == "ollama":
        return OllamaError(f"Failed to request Ollama embeddings: {exc}")
    return LLMError(f"Failed to request cloud embeddings: {exc}")


###############################################################################
class CloudEmbeddingGenerator:
    def __init__(
        self,
        *,
        provider: ProviderName,
        model: str,
        timeout_s: float = server_settings.external_data.default_llm_timeout,
    ) -> None:
        normalized_provider = (provider or "").strip().lower()
        if normalized_provider not in {"openai", "gemini"}:
            raise ValueError(f"Unsupported cloud provider: {provider}")
        resolved_model = (model or "").strip()
        if not resolved_model:
            raise ValueError("Cloud embedding model is required")
        self.provider = cast(Literal["openai", "gemini"], normalized_provider)
        self.model = resolved_model
        self.timeout_s = float(timeout_s)
        self.api_key = self.resolve_provider_access_key(self.provider)

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_provider_access_key(provider: Literal["openai", "gemini"]) -> str:
        serializer = AccessKeySerializer()
        label = "OpenAI" if provider == "openai" else "Gemini"
        try:
            row = serializer.get_active_key(provider, mark_used=True)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Failed to load active {label} access key") from exc
        if row is None:
            raise LLMError(f"No active {label} access key configured")
        try:
            return serializer.decrypt_key_row(row)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Failed to decrypt active {label} access key") from exc

    # -------------------------------------------------------------------------
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            if self.provider == "openai":
                embeddings_model = _build_openai_embeddings_model(
                    api_key=self.api_key,
                    model=self.model,
                    timeout_s=self.timeout_s,
                )
            else:
                embeddings_model = _build_gemini_embeddings_model(
                    api_key=self.api_key,
                    model=self.model,
                    timeout_s=self.timeout_s,
                )
            vectors = await asyncio.to_thread(embeddings_model.embed_documents, texts)
        except Exception as exc:  # noqa: BLE001
            raise _map_langchain_embedding_exception(
                exc, provider=self.provider
            ) from exc
        return self.normalize_embeddings(vectors, expected=len(texts))

    # -------------------------------------------------------------------------
    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.embed_texts([query])
        if not vectors:
            return []
        return vectors[0]

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_embeddings(vectors: list[Any], *, expected: int) -> list[list[float]]:
        normalized: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list):
                raise LLMError("Invalid embedding payload returned by cloud provider")
            try:
                normalized.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise LLMError("Non-numeric values found in cloud embeddings") from exc
        if len(normalized) != expected:
            raise LLMError("Mismatch between cloud embeddings and inputs")
        return normalized


###############################################################################
class OllamaEmbeddingGenerator:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        timeout_s: float = server_settings.external_data.default_llm_timeout,
    ) -> None:
        resolved_model = (model or "").strip()
        if not resolved_model:
            raise ValueError("Ollama embedding model is required")
        self.model = resolved_model
        self.base_url = (
            base_url or server_settings.llm_defaults.ollama_host_default
        ).rstrip("/")
        self.timeout_s = float(timeout_s)

    # -------------------------------------------------------------------------
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings_model = _build_ollama_embeddings_model(
            base_url=self.base_url,
            model=self.model,
            timeout_s=self.timeout_s,
        )
        try:
            vectors = await asyncio.to_thread(embeddings_model.embed_documents, texts)
        except Exception as exc:  # noqa: BLE001
            raise _map_langchain_embedding_exception(exc, provider="ollama") from exc
        return self.normalize_embeddings(vectors, expected=len(texts))

    # -------------------------------------------------------------------------
    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.embed_texts([query])
        if not vectors:
            return []
        return vectors[0]

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_embeddings(vectors: list[Any], *, expected: int) -> list[list[float]]:
        normalized: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list):
                raise OllamaError("Invalid embedding payload returned by Ollama")
            try:
                normalized.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise OllamaError(
                    "Non-numeric values found in Ollama embeddings"
                ) from exc
        if len(normalized) != expected:
            raise OllamaError("Mismatch between Ollama embeddings and inputs")
        return normalized


###############################################################################
def select_embedding_provider(
    *,
    backend: str,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
    use_cloud_embeddings: bool = False,
    cloud_provider: str | None = None,
    cloud_embedding_model: str | None = None,
    timeout_s: float = server_settings.external_data.default_llm_timeout,
) -> CloudEmbeddingGenerator | OllamaEmbeddingGenerator:
    normalized_backend = backend.lower().strip() if backend else "ollama"
    if use_cloud_embeddings:
        normalized_backend = "cloud"

    if normalized_backend == "cloud":
        if not cloud_provider:
            raise ValueError("Cloud provider is required for embeddings")
        if not cloud_embedding_model:
            raise ValueError("Cloud embedding model is required")
        provider_normalized = cloud_provider.lower().strip()
        if provider_normalized not in CLOUD_MODEL_CHOICES:
            raise ValueError(
                f"Unsupported cloud provider: {cloud_provider}. "
                f"Allowed: {', '.join(CLOUD_MODEL_CHOICES.keys())}"
            )
        return CloudEmbeddingGenerator(
            provider=cast(ProviderName, provider_normalized),
            model=cloud_embedding_model,
            timeout_s=timeout_s,
        )

    if normalized_backend == "ollama":
        return OllamaEmbeddingGenerator(
            model=(ollama_model or "").strip(),
            base_url=ollama_base_url,
            timeout_s=timeout_s,
        )

    raise ValueError(f"Unsupported embedding backend: {backend}")


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
        self.backend: EmbeddingBackend = (
            "cloud"
            if use_cloud_embeddings
            else cast(EmbeddingBackend, normalized_backend)
        )
        self.provider = select_embedding_provider(
            backend=backend,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            use_cloud_embeddings=use_cloud_embeddings,
            cloud_provider=cloud_provider,
            cloud_embedding_model=cloud_embedding_model,
        )

    # -------------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        sanitized = [text if text.strip() else " " for text in texts]
        if not sanitized:
            return []
        embeddings = self.run_async(self.provider.embed_texts(sanitized))
        return [[float(value) for value in vector] for vector in embeddings]

    # -------------------------------------------------------------------------
    @staticmethod
    def run_async(
        coroutine: Coroutine[Any, Any, list[list[float]]],
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


###############################################################################
class SimilaritySearch:
    def __init__(
        self,
        *,
        vector_database: LanceVectorDatabase | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        default_top_k: int = server_settings.rag.rerank_candidate_k,
    ) -> None:
        self.default_top_k = max(int(default_top_k), 1)
        self.vector_database = vector_database or LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=server_settings.rag.vector_collection_name,
            metric=server_settings.rag.vector_index_metric,
            index_type=server_settings.rag.vector_index_type,
            stream_batch_size=server_settings.rag.vector_stream_batch_size,
        )
        self.embedding_generator = embedding_generator or EmbeddingGenerator(
            backend=server_settings.rag.embedding_backend,
            ollama_base_url=server_settings.rag.ollama_base_url,
            ollama_model=server_settings.rag.ollama_embedding_model,
            use_cloud_embeddings=server_settings.rag.use_cloud_embeddings,
            cloud_provider=server_settings.rag.cloud_provider,
            cloud_embedding_model=server_settings.rag.cloud_embedding_model,
        )
        try:
            self.vector_database.initialize()
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

    # -------------------------------------------------------------------------
    def search_with_reranking(
        self,
        query: str,
        *,
        candidate_k: int | None = None,
        final_top_n: int | None = None,
        use_reranking: bool | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str):
            return []
        normalized = query.strip()
        if not normalized:
            return []

        resolved_top_n = (
            max(int(final_top_n), 1)
            if final_top_n is not None
            else max(int(server_settings.rag.rerank_top_n), 1)
        )
        resolved_candidate_k = (
            max(int(candidate_k), 1)
            if candidate_k is not None
            else max(int(self.default_top_k), 1)
        )
        if resolved_candidate_k < resolved_top_n:
            resolved_candidate_k = resolved_top_n

        candidates = self.search(normalized, top_k=resolved_candidate_k)
        if not candidates:
            return []

        should_rerank = (
            server_settings.rag.use_reranking
            if use_reranking is None
            else bool(use_reranking)
        )
        if not should_rerank:
            return candidates[:resolved_top_n]

        return self.rerank_candidates(normalized, candidates, top_n=resolved_top_n)

    # -------------------------------------------------------------------------
    def rerank_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if top_n <= 0:
            return []

        texts = [str(item.get("text") or "").strip() for item in candidates]
        try:
            vectors = self.embedding_generator.embed_texts([query, *texts])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reranking fallback to retrieval order due to embedding error: %s", exc
            )
            return candidates[:top_n]

        expected_vectors = len(candidates) + 1
        if len(vectors) != expected_vectors:
            logger.warning(
                "Reranking fallback: expected %d vectors but got %d",
                expected_vectors,
                len(vectors),
            )
            return candidates[:top_n]

        query_vector = vectors[0]
        scored: list[dict[str, Any]] = []
        for candidate, candidate_vector in zip(candidates, vectors[1:], strict=False):
            score = self.cosine_similarity(query_vector, candidate_vector)
            if score is None:
                logger.warning("Reranking fallback due to invalid vector shape or norm")
                return candidates[:top_n]
            enriched = dict(candidate)
            enriched["rerank_score"] = score
            scored.append(enriched)

        scored.sort(key=self.rerank_sort_key, reverse=True)
        return scored[:top_n]

    # -------------------------------------------------------------------------
    @staticmethod
    def rerank_sort_key(document: dict[str, Any]) -> float:
        raw_score = document.get("rerank_score")
        if isinstance(raw_score, (int, float)):
            return float(raw_score)
        return float("-inf")

    # -------------------------------------------------------------------------
    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float | None:
        if not left or not right or len(left) != len(right):
            return None
        dot_product = 0.0
        left_norm_sq = 0.0
        right_norm_sq = 0.0
        for left_value, right_value in zip(left, right, strict=False):
            dot_product += left_value * right_value
            left_norm_sq += left_value * left_value
            right_norm_sq += right_value * right_value
        if left_norm_sq <= 0.0 or right_norm_sq <= 0.0:
            return None
        denominator = math.sqrt(left_norm_sq) * math.sqrt(right_norm_sq)
        if denominator <= 0.0:
            return None
        return dot_product / denominator


__all__ = [
    "CloudEmbeddingGenerator",
    "OllamaEmbeddingGenerator",
    "select_embedding_provider",
    "EmbeddingGenerator",
    "SimilaritySearch",
]
