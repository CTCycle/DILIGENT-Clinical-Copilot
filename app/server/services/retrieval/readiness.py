from __future__ import annotations

from typing import Any

import httpx

from domain.clinical.robustness import RagReadiness
from services.retrieval.settings import build_effective_rag_settings

OLLAMA_TAGS_PATH = "/api/tags"
OLLAMA_READINESS_TIMEOUT_SECONDS = 2.5


def check_rag_readiness(*, requested: bool) -> RagReadiness:
    settings = build_effective_rag_settings()
    backend = (
        "cloud"
        if settings.use_cloud_embeddings
        else (settings.embedding_backend or "ollama").strip().lower()
    )
    model = (
        settings.cloud_embedding_model
        if backend == "cloud"
        else settings.ollama_embedding_model
    ).strip() or None

    if not requested:
        return RagReadiness(
            requested=False,
            available=True,
            backend=backend,
            model=model,
        )
    if backend != "ollama":
        return RagReadiness(
            requested=True,
            available=True,
            backend=backend,
            model=model,
        )
    if model is None:
        return RagReadiness(
            requested=True,
            available=False,
            backend=backend,
            model=None,
            reason_code="rag_embedding_model_missing",
            message=(
                "RAG uses Ollama embeddings, but no embedding model is configured. "
                "Configure the RAG embedding model or run this assessment without RAG."
            ),
        )

    url = f"{settings.ollama_base_url.rstrip('/')}{OLLAMA_TAGS_PATH}"
    try:
        response = httpx.get(url, timeout=OLLAMA_READINESS_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload: Any = response.json()
    except Exception:
        return RagReadiness(
            requested=True,
            available=False,
            backend=backend,
            model=model,
            reason_code="rag_ollama_unavailable",
            message=(
                "RAG was enabled, but the Ollama server used by the indexed "
                "embeddings is unavailable. Start Ollama and retry, or run this "
                "assessment without RAG."
            ),
        )

    available_models = {
        str(item.get("name") or item.get("model") or "").strip()
        for item in payload.get("models", [])
        if isinstance(item, dict)
    } if isinstance(payload, dict) else set()
    if model not in available_models:
        return RagReadiness(
            requested=True,
            available=False,
            backend=backend,
            model=model,
            reason_code="rag_ollama_model_unavailable",
            message=(
                f"RAG requires the Ollama embedding model '{model}', but that model "
                "is not available. Start or install the model, then retry, or run "
                "this assessment without RAG."
            ),
        )
    return RagReadiness(
        requested=True,
        available=True,
        backend=backend,
        model=model,
    )
