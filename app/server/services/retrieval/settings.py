# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections.abc import Mapping

from common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_positive_int,
    coerce_str,
)
from configurations.startup import get_server_settings
from domain.model_configs import SharedRagSettingsResponse
from domain.settings.configuration import SHARED_RAG_SETTING_KEYS, RagSettings


###############################################################################
def _runtime_rag_settings() -> dict[str, object]:
    # Resolve through the runtime config so a frozen run snapshot wins over
    # the live persisted settings.
    from services.llm.runtime_config import LLMRuntimeConfig

    try:
        return LLMRuntimeConfig.get_rag_settings()
    except Exception:
        return {}


###############################################################################
def build_effective_rag_settings(
    overrides: dict[str, object] | None = None,
    *,
    persisted_settings: dict[str, object] | None = None,
) -> RagSettings:
    base = get_server_settings().rag
    persisted = (
        dict(persisted_settings)
        if persisted_settings is not None
        else _runtime_rag_settings()
    )
    data = {**persisted, **dict(overrides or {})}

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
            "allow_local_filesystem_access": coerce_bool(
                data.get("allow_local_filesystem_access"),
                base.allow_local_filesystem_access,
            ),
            "vector_collection_name": coerce_str(
                data.get("vector_collection_name"), base.vector_collection_name
            ),
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
            "vector_index_metric": coerce_str(
                data.get("vector_index_metric"), base.vector_index_metric
            ),
            "vector_index_type": coerce_str(
                data.get("vector_index_type"), base.vector_index_type
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
def normalize_rag_settings_patch(
    payload: Mapping[str, object],
    *,
    persisted_settings: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Merge a shared RAG settings patch over the persisted shared settings."""
    return rag_settings_payload(
        build_effective_rag_settings(
            dict(payload),
            persisted_settings=(
                dict(persisted_settings) if persisted_settings is not None else None
            ),
        )
    ).model_dump(mode="python")


###############################################################################
def rag_settings_payload(
    settings: RagSettings | None = None,
) -> SharedRagSettingsResponse:
    resolved = settings or build_effective_rag_settings()
    return SharedRagSettingsResponse.model_validate(
        {key: getattr(resolved, key) for key in SHARED_RAG_SETTING_KEYS}
    )
