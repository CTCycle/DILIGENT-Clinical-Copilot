from __future__ import annotations

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from domain.clinical.robustness import RagReadiness
from services.retrieval.embedding_runtime import get_embedding_runtime

###############################################################################
def check_rag_readiness(*, requested: bool) -> RagReadiness:
    """Check local embedding cache readiness without loading or downloading a model."""

    backend = "onnxruntime"
    model = CANONICAL_EMBEDDING_CONFIG.model_id
    if not requested:
        return RagReadiness(
            requested=False,
            available=True,
            backend=backend,
            model=model,
        )

    runtime_status = get_embedding_runtime().status()
    cache_status = str(runtime_status["cache_status"])
    if cache_status != "available":
        reason_code = {
            "missing": "rag_embedding_cache_missing",
            "dependency_missing": "rag_embedding_dependency_missing",
            "invalid": "rag_embedding_cache_invalid",
        }.get(cache_status, "rag_embedding_cache_invalid")
        message = {
            "missing": "Reconnect and rebuild the RAG cache, or disable RAG for this assessment.",
            "dependency_missing": "Reinstall the ONNX Runtime, Tokenizers, and NumPy dependencies, or disable RAG for this assessment.",
            "invalid": "Remove the corrupted embedding cache and rebuild the RAG cache, or disable RAG for this assessment.",
        }.get(
            cache_status,
            "Repair the embedding cache and rebuild RAG, or disable RAG for this assessment.",
        )
        return RagReadiness(
            requested=True,
            available=False,
            backend=backend,
            model=model,
            reason_code=reason_code,
            message=message,
        )
    return RagReadiness(
        requested=True,
        available=True,
        backend=backend,
        model=model,
    )


__all__ = ["check_rag_readiness"]
