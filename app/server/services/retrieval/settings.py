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


###############################################################################
def _runtime_rag_settings() -> dict[str, object]:
    try:
        snapshot = ModelConfigSerializer().load_snapshot()
    except Exception:
        return {}
    return dict(snapshot.rag_settings or {})


###############################################################################
def build_effective_rag_settings(
    overrides: dict[str, object] | None = None,
) -> RagSettings:
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
            "reranker_model": coerce_str(
                data.get("reranker_model"), base.reranker_model
            ),
            "hybrid_vector_weight": max(
                coerce_float(
                    data.get("hybrid_vector_weight"), base.hybrid_vector_weight
                ),
                0.0,
            ),
            "hybrid_text_weight": max(
                coerce_float(data.get("hybrid_text_weight"), base.hybrid_text_weight),
                0.0,
            ),
            "vector_stream_batch_size": coerce_positive_int(
                data.get("vector_stream_batch_size"), base.vector_stream_batch_size
            ),
            "embedding_offline_mode": coerce_bool(
                data.get("embedding_offline_mode"), base.embedding_offline_mode
            ),
        }
    )


###############################################################################
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
        "vector_stream_batch_size": resolved.vector_stream_batch_size,
        "embedding_offline_mode": resolved.embedding_offline_mode,
    }
