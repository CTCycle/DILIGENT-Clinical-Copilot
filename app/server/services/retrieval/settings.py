from __future__ import annotations

from typing import Any

from common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_positive_int,
    coerce_str,
)
from configurations.startup import get_server_settings
from domain.settings.configuration import RagSettings
from repositories.serialization.model_configs import ModelConfigSerializer


def _runtime_rag_settings() -> dict[str, object]:
    try:
        snapshot = ModelConfigSerializer().load_snapshot()
    except Exception:
        return {}
    return dict(snapshot.rag_settings or {})


def build_effective_rag_settings(overrides: dict[str, object] | None = None) -> RagSettings:
    base = get_server_settings().rag
    data = {**_runtime_rag_settings(), **dict(overrides or {})}

    selected_count = coerce_positive_int(
        data.get("retrieval_selected_count"), base.retrieval_selected_count
    )
    candidate_count = coerce_positive_int(
        data.get("retrieval_candidate_count"), base.retrieval_candidate_count
    )
    if candidate_count < selected_count:
        candidate_count = selected_count

    return base.model_copy(
        update={
            "chunk_size": coerce_positive_int(data.get("chunk_size"), base.chunk_size),
            "chunk_overlap": coerce_positive_int(
                data.get("chunk_overlap"), base.chunk_overlap
            ),
            "embedding_batch_size": coerce_positive_int(
                data.get("embedding_batch_size"), base.embedding_batch_size
            ),
            "use_hybrid_search": coerce_bool(
                data.get("use_hybrid_search"), base.use_hybrid_search
            ),
            "use_reranking": coerce_bool(data.get("use_reranking"), base.use_reranking),
            "retrieval_candidate_count": candidate_count,
            "retrieval_selected_count": selected_count,
            "reranker_model": coerce_str(data.get("reranker_model"), base.reranker_model),
            "hybrid_vector_weight": max(
                coerce_float(data.get("hybrid_vector_weight"), base.hybrid_vector_weight),
                0.0,
            ),
            "hybrid_text_weight": max(
                coerce_float(data.get("hybrid_text_weight"), base.hybrid_text_weight),
                0.0,
            ),
            "embedding_backend": coerce_str(
                data.get("embedding_backend"), base.embedding_backend
            ),
            "ollama_embedding_model": coerce_str(
                data.get("ollama_embedding_model"), base.ollama_embedding_model
            ),
            "hf_embedding_model": coerce_str(
                data.get("hf_embedding_model"), base.hf_embedding_model
            ),
            "cloud_provider": coerce_str(data.get("cloud_provider"), base.cloud_provider),
            "cloud_embedding_model": coerce_str(
                data.get("cloud_embedding_model"), base.cloud_embedding_model
            ),
            "use_cloud_embeddings": coerce_bool(
                data.get("use_cloud_embeddings"), base.use_cloud_embeddings
            ),
            "reset_vector_collection": coerce_bool(
                data.get("reset_vector_collection"), base.reset_vector_collection
            ),
            "vector_stream_batch_size": coerce_positive_int(
                data.get("vector_stream_batch_size"), base.vector_stream_batch_size
            ),
            "embedding_max_workers": coerce_positive_int(
                data.get("embedding_max_workers"), base.embedding_max_workers
            ),
        }
    )


def rag_settings_payload(settings: RagSettings | None = None) -> dict[str, Any]:
    resolved = settings or build_effective_rag_settings()
    return {
        "chunk_size": resolved.chunk_size,
        "chunk_overlap": resolved.chunk_overlap,
        "embedding_batch_size": resolved.embedding_batch_size,
        "use_hybrid_search": resolved.use_hybrid_search,
        "use_reranking": resolved.use_reranking,
        "retrieval_candidate_count": resolved.retrieval_candidate_count,
        "retrieval_selected_count": resolved.retrieval_selected_count,
        "reranker_model": resolved.reranker_model,
        "hybrid_vector_weight": resolved.hybrid_vector_weight,
        "hybrid_text_weight": resolved.hybrid_text_weight,
        "embedding_backend": resolved.embedding_backend,
        "ollama_embedding_model": resolved.ollama_embedding_model,
        "hf_embedding_model": resolved.hf_embedding_model,
        "cloud_provider": resolved.cloud_provider,
        "cloud_embedding_model": resolved.cloud_embedding_model,
        "use_cloud_embeddings": resolved.use_cloud_embeddings,
        "reset_vector_collection": resolved.reset_vector_collection,
        "vector_stream_batch_size": resolved.vector_stream_batch_size,
        "embedding_max_workers": resolved.embedding_max_workers,
    }
