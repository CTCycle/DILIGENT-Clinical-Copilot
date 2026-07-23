from __future__ import annotations

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from domain.clinical.robustness import RagReadiness
from services.retrieval.embedding_runtime import get_embedding_runtime


def check_rag_readiness(*, requested: bool) -> RagReadiness:
    """Check local embedding cache readiness without loading or downloading a model."""

    backend = "sentence-transformers"
    model = CANONICAL_EMBEDDING_CONFIG.model_id
    if not requested:
        return RagReadiness(
            requested=False,
            available=True,
            backend=backend,
            model=model,
        )

    runtime_status = get_embedding_runtime().status()
    if runtime_status["cache_status"] != "available":
        return RagReadiness(
            requested=True,
            available=False,
            backend=backend,
            model=model,
            reason_code="rag_embedding_cache_missing",
            message=(
                "RAG requires the pinned Granite embedding snapshot. "
                "Prepare the local model cache or disable RAG for this assessment."
            ),
        )
    return RagReadiness(
        requested=True,
        available=True,
        backend=backend,
        model=model,
    )


__all__ = ["check_rag_readiness"]
