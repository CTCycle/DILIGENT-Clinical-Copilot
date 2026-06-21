from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Coroutine
from difflib import SequenceMatcher
from typing import Any, Literal, Protocol, cast

import httpx

from common.catalogs.model_choices import get_cloud_model_choices
from common.paths import VECTOR_DB_PATH
from common.utils.logger import logger
from configurations.startup import get_server_settings
from repositories.serialization.access_keys import AccessKeySerializer
from repositories.vectors import LanceVectorDatabase
from services.llm.cloud import CloudLLMClient, LLMError, LLMTimeout
from services.llm.ollama_client import OllamaClient, OllamaError, OllamaTimeout
from common.utils.embedding_model import (
    EmbeddingModelSpec,
    build_embedding_model_signature,
)
from services.retrieval.settings import build_effective_rag_settings

ProviderName = Literal["openai", "gemini"]
EmbeddingBackend = Literal["ollama", "cloud"]

###############################################################################
class EmbeddingModelMismatchError(RuntimeError):
    pass

###############################################################################
class Reranker(Protocol):

    # -------------------------------------------------------------------------
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...

###############################################################################
class LocalHeuristicReranker:
    TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
    KNOWN_PROFILES = frozenset({
        "lightweight-balanced-v1",
        "lightweight-lexical-v1",
        "lightweight-phrase-v1",
    })
    PROFILE_WEIGHTS = {
        "lightweight-balanced-v1": {
            "query_coverage": 0.45,
            "document_precision": 0.20,
            "sequence_ratio": 0.20,
            "phrase_bonus": 0.35,
            "bigram_bonus": 0.15,
        },
        "lightweight-lexical-v1": {
            "query_coverage": 0.60,
            "document_precision": 0.28,
            "sequence_ratio": 0.10,
            "phrase_bonus": 0.18,
            "bigram_bonus": 0.08,
        },
        "lightweight-phrase-v1": {
            "query_coverage": 0.32,
            "document_precision": 0.12,
            "sequence_ratio": 0.24,
            "phrase_bonus": 0.55,
            "bigram_bonus": 0.24,
        },
    }

    # -------------------------------------------------------------------------
    def __init__(self, model_name: str) -> None:
        self.model_name = self.normalize_profile_name(model_name)

    # -------------------------------------------------------------------------
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        return [self.score_pair(query, document, profile_name=self.model_name) for query, document in pairs]

    # -------------------------------------------------------------------------
    @classmethod
    def score_pair(
        cls,
        query: str,
        document: str,
        *,
        profile_name: str = "lightweight-balanced-v1",
    ) -> float:
        normalized_query = cls.normalize_text(query)
        normalized_document = cls.normalize_text(document)
        if not normalized_query or not normalized_document:
            return 0.0

        query_tokens = cls.tokenize(normalized_query)
        document_tokens = cls.tokenize(normalized_document)
        if not query_tokens or not document_tokens:
            return 0.0

        query_token_set = set(query_tokens)
        document_token_set = set(document_tokens)
        overlap_count = len(query_token_set & document_token_set)
        query_coverage = overlap_count / max(len(query_token_set), 1)
        document_precision = overlap_count / max(len(document_token_set), 1)
        sequence_ratio = SequenceMatcher(
            None,
            normalized_query,
            normalized_document,
        ).ratio()
        weights = cls.profile_weights(profile_name)
        phrase_bonus = (
            weights["phrase_bonus"] if normalized_query in normalized_document else 0.0
        )
        ordered_bigram_bonus = weights["bigram_bonus"] * cls.ordered_bigram_ratio(
            query_tokens,
            document_tokens,
        )

        return (
            (query_coverage * weights["query_coverage"])
            + (document_precision * weights["document_precision"])
            + (sequence_ratio * weights["sequence_ratio"])
            + phrase_bonus
            + ordered_bigram_bonus
        )

    # -------------------------------------------------------------------------
    @classmethod
    def normalize_profile_name(cls, value: str) -> str:
        candidate = (value or "").strip().casefold()
        if candidate in cls.KNOWN_PROFILES:
            return candidate
        return "lightweight-balanced-v1"

    # -------------------------------------------------------------------------
    @classmethod
    def profile_weights(cls, profile_name: str) -> dict[str, float]:
        normalized = cls.normalize_profile_name(profile_name)
        return cls.PROFILE_WEIGHTS[normalized]

    # -------------------------------------------------------------------------
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(cls.TOKEN_PATTERN.findall((value or "").casefold()))

    # -------------------------------------------------------------------------
    @classmethod
    def tokenize(cls, value: str) -> list[str]:
        return cls.TOKEN_PATTERN.findall(value)

    # -------------------------------------------------------------------------
    @staticmethod
    def ordered_bigram_ratio(
        query_tokens: list[str],
        document_tokens: list[str],
    ) -> float:
        if len(query_tokens) < 2 or len(document_tokens) < 2:
            return 0.0
        document_bigrams = {
            (document_tokens[index], document_tokens[index + 1])
            for index in range(len(document_tokens) - 1)
        }
        query_bigrams = [
            (query_tokens[index], query_tokens[index + 1])
            for index in range(len(query_tokens) - 1)
        ]
        matches = sum(1 for bigram in query_bigrams if bigram in document_bigrams)
        if matches <= 0:
            return 0.0
        return matches / len(query_bigrams)

###############################################################################
def _map_embedding_exception(
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

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        provider: ProviderName,
        model: str,
        timeout_s: float = get_server_settings().runtime.default_llm_timeout,
    ) -> None:
        normalized_provider = (provider or "").strip().lower()
        if normalized_provider not in {"openai", "gemini"}:
            raise ValueError(f"Unsupported cloud provider: {provider}")
        resolved_model = (model or "").strip()
        if not resolved_model:
            raise ValueError("Cloud embedding model is required")
        self.provider: ProviderName = cast(
            Literal["openai", "gemini"], normalized_provider
        )
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
            async with CloudLLMClient(
                provider=self.provider,
                default_model=self.model,
                timeout_s=self.timeout_s,
            ) as client:
                vectors = await client.embed(model=self.model, input_texts=texts)
        except Exception as exc:  # noqa: BLE001
            raise _map_embedding_exception(exc, provider=self.provider) from exc
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

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        timeout_s: float = get_server_settings().runtime.default_llm_timeout,
    ) -> None:
        resolved_model = (model or "").strip()
        if not resolved_model:
            raise ValueError("Ollama embedding model is required")
        self.model = resolved_model
        self.base_url = (
            base_url or get_server_settings().llm_defaults.ollama_host_default
        ).rstrip("/")
        self.timeout_s = float(timeout_s)

    # -------------------------------------------------------------------------
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            async with OllamaClient(
                base_url=self.base_url,
                timeout_s=self.timeout_s,
                default_model=self.model,
            ) as client:
                vectors = await client.embed(model=self.model, input_texts=texts)
        except Exception as exc:  # noqa: BLE001
            raise _map_embedding_exception(exc, provider="ollama") from exc
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
    timeout_s: float = get_server_settings().runtime.default_llm_timeout,
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
        if provider_normalized not in get_cloud_model_choices():
            raise ValueError(
                f"Unsupported cloud provider: {cloud_provider}. "
                f"Allowed: {', '.join(get_cloud_model_choices().keys())}"
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

    # -------------------------------------------------------------------------
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
    def resolve_active_embedding_model_spec(self) -> EmbeddingModelSpec:
        provider = "cloud" if self.backend == "cloud" else "ollama"
        if self.backend == "cloud":
            model_name = str(getattr(self.provider, "model", "") or "")
            mode = str(getattr(self.provider, "provider", "cloud"))
        else:
            model_name = str(getattr(self.provider, "model", "") or "")
            mode = "local"
        sample = self.embed_texts(["embedding-dimension-probe"])
        dimension = len(sample[0]) if sample else 0
        signature = build_embedding_model_signature(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            mode=mode,
        )
        return EmbeddingModelSpec(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            mode=mode,
            signature=signature,
        )

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

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        vector_database: LanceVectorDatabase | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        default_top_k: int | None = None,
    ) -> None:
        rag_settings = build_effective_rag_settings()
        self.default_top_k = max(
            int(
                rag_settings.retrieval_candidate_count
                if default_top_k is None
                else default_top_k
            ),
            1,
        )
        self.reranker: Reranker | None = None
        self.vector_database = vector_database or LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=rag_settings.vector_collection_name,
            metric=rag_settings.vector_index_metric,
            index_type=rag_settings.vector_index_type,
            stream_batch_size=rag_settings.vector_stream_batch_size,
        )
        self.embedding_generator = embedding_generator or EmbeddingGenerator(
            backend=rag_settings.embedding_backend,
            ollama_base_url=rag_settings.ollama_base_url,
            ollama_model=rag_settings.ollama_embedding_model,
            use_cloud_embeddings=rag_settings.use_cloud_embeddings,
            cloud_provider=rag_settings.cloud_provider,
            cloud_embedding_model=rag_settings.cloud_embedding_model,
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
        model_spec = self.embedding_generator.resolve_active_embedding_model_spec()
        try:
            self.vector_database.assert_query_model_matches_index(model_spec.signature)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingModelMismatchError(
                "Active embedding model does not match indexed vectors. Rebuild the RAG vector store."
            ) from exc
        embeddings = self.embedding_generator.embed_texts([normalized])
        if not embeddings:
            return []
        try:
            table = self.vector_database.get_table()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to access LanceDB table: %s", exc)
            return []
        try:
            if build_effective_rag_settings().use_hybrid_search:
                results = self.hybrid_search(
                    table=table,
                    query=normalized,
                    query_embedding=embeddings[0],
                    limit=limit,
                )
            else:
                results = self.vector_search(
                    table=table,
                    query_embedding=embeddings[0],
                    limit=limit,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Similarity search failed: %s", exc)
            return []
        return self.normalize_results(results)

    # -------------------------------------------------------------------------
    def vector_search(
        self,
        *,
        table: Any,
        query_embedding: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        return table.search(query_embedding).limit(limit).to_list()

    # -------------------------------------------------------------------------
    def text_search(
        self, *, table: Any, query: str, limit: int
    ) -> list[dict[str, Any]]:
        return table.search(query, query_type="fts").limit(limit).to_list()

    # -------------------------------------------------------------------------
    def hybrid_search(
        self,
        *,
        table: Any,
        query: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        vector_results = self.vector_search(
            table=table,
            query_embedding=query_embedding,
            limit=limit,
        )
        try:
            text_results = self.text_search(table=table, query=query, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Full-text search unavailable; using vector results: %s", exc
            )
            text_results = []
        return self.fuse_results(vector_results, text_results, query=query)[:limit]

    # -------------------------------------------------------------------------
    def fuse_results(
        self,
        vector_results: list[dict[str, Any]],
        text_results: list[dict[str, Any]],
        *,
        query: str,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        rag_settings = build_effective_rag_settings()
        vector_weight = float(rag_settings.hybrid_vector_weight)
        text_weight = float(rag_settings.hybrid_text_weight)
        for rank, entry in enumerate(vector_results, start=1):
            self.add_rank_score(
                fused,
                entry,
                score=vector_weight / (60 + rank),
                source="vector",
            )
        for rank, entry in enumerate(text_results, start=1):
            self.add_rank_score(
                fused,
                entry,
                score=text_weight / (60 + rank),
                source="text",
            )
        for entry in fused.values():
            entry["hybrid_score"] = float(
                entry.get("hybrid_score", 0.0)
            ) + self.metadata_boost(
                entry,
                query,
            )
        return sorted(
            fused.values(),
            key=lambda item: float(item.get("hybrid_score", 0.0)),
            reverse=True,
        )

    # -------------------------------------------------------------------------
    def add_rank_score(
        self,
        fused: dict[str, dict[str, Any]],
        entry: dict[str, Any],
        *,
        score: float,
        source: str,
    ) -> None:
        key = str(entry.get("chunk_id") or entry.get("document_id") or id(entry))
        existing = fused.get(key)
        if existing is None:
            existing = dict(entry)
            existing["retrieval_sources"] = []
            existing["hybrid_score"] = 0.0
            fused[key] = existing
        existing["hybrid_score"] = float(existing["hybrid_score"]) + score
        retrieval_sources = existing.get("retrieval_sources")
        if isinstance(retrieval_sources, list):
            retrieval_sources.append(source)

    # -------------------------------------------------------------------------
    def metadata_boost(self, entry: dict[str, Any], query: str) -> float:
        normalized = query.casefold()
        boost = 0.0
        file_name = str(entry.get("file_name") or "").casefold()
        document_title = str(entry.get("document_title") or "").casefold()
        section_title = str(entry.get("section_title") or "").casefold()
        if file_name and file_name in normalized:
            boost += 0.02
        if document_title and document_title in normalized:
            boost += 0.015
        if section_title and section_title in normalized:
            boost += 0.01
        return boost

    # -------------------------------------------------------------------------
    def normalize_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    "file_name": entry.get("file_name"),
                    "document_title": entry.get("document_title"),
                    "page_number": entry.get("page_number"),
                    "section_title": entry.get("section_title"),
                    "heading_path": entry.get("heading_path"),
                    "content_type": entry.get("content_type"),
                    "metadata": metadata,
                    "distance": distance_value,
                    "hybrid_score": entry.get("hybrid_score"),
                    "retrieval_sources": entry.get("retrieval_sources"),
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
            else max(int(build_effective_rag_settings().retrieval_selected_count), 1)
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
            build_effective_rag_settings().use_reranking
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

        try:
            model = self.get_reranker()
            scores = model.predict(
                [(query, str(item.get("text") or "").strip()) for item in candidates]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reranking fallback to retrieval order due to reranker error: %s", exc
            )
            return candidates[:top_n]

        if len(scores) != len(candidates):
            logger.warning(
                "Reranking fallback: expected %d scores but got %d",
                len(candidates),
                len(scores),
            )
            return candidates[:top_n]

        scored: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores, strict=False):
            enriched = dict(candidate)
            enriched["rerank_score"] = float(score) + self.retrieval_prior(candidate)
            scored.append(enriched)

        scored.sort(key=self.rerank_sort_key, reverse=True)
        return scored[:top_n]

    # -------------------------------------------------------------------------
    def get_reranker(self) -> Reranker:
        if self.reranker is None:
            self.reranker = LocalHeuristicReranker(
                build_effective_rag_settings().reranker_model
            )
        return self.reranker

    # -------------------------------------------------------------------------
    @staticmethod
    def retrieval_prior(document: dict[str, Any]) -> float:
        raw_distance = document.get("distance")
        if isinstance(raw_distance, (int, float)) and raw_distance >= 0:
            return 0.15 / (1.0 + float(raw_distance))
        return 0.0

    # -------------------------------------------------------------------------
    @staticmethod
    def rerank_sort_key(document: dict[str, Any]) -> float:
        raw_score = document.get("rerank_score")
        if isinstance(raw_score, (int, float)):
            return float(raw_score)
        return float("-inf")


__all__ = [
    "EmbeddingModelMismatchError",
    "CloudEmbeddingGenerator",
    "OllamaEmbeddingGenerator",
    "select_embedding_provider",
    "EmbeddingGenerator",
    "LocalHeuristicReranker",
    "SimilaritySearch",
]
