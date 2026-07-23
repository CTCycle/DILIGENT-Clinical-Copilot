from __future__ import annotations

from types import SimpleNamespace

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from services.retrieval import readiness


def test_rag_readiness_reports_missing_local_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness,
        "get_embedding_runtime",
        lambda: SimpleNamespace(
            status=lambda: {"cache_status": "missing"},
        ),
    )

    result = readiness.check_rag_readiness(requested=True)

    assert result.available is False
    assert result.reason_code == "rag_embedding_cache_missing"
    assert result.backend == "sentence-transformers"


def test_rag_readiness_accepts_complete_local_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness,
        "get_embedding_runtime",
        lambda: SimpleNamespace(
            status=lambda: {"cache_status": "available"},
        ),
    )

    result = readiness.check_rag_readiness(requested=True)

    assert result.available is True
    assert result.backend == "sentence-transformers"
    assert result.model == CANONICAL_EMBEDDING_CONFIG.model_id


def test_rag_readiness_does_not_load_model_when_not_requested(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness,
        "get_embedding_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime should not be read")),
    )

    result = readiness.check_rag_readiness(requested=False)

    assert result.available is True
    assert result.requested is False
