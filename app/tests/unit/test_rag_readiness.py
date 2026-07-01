from __future__ import annotations

from types import SimpleNamespace

import httpx

from services.retrieval import readiness

###############################################################################
def _settings(**overrides: object) -> SimpleNamespace:
    values = {
        "use_cloud_embeddings": False,
        "embedding_backend": "ollama",
        "cloud_embedding_model": "",
        "ollama_embedding_model": "nomic-embed-text:latest",
        "ollama_base_url": "http://127.0.0.1:11434",
    }
    values.update(overrides)
    return SimpleNamespace(**values)

###############################################################################
class _Response:

    # -------------------------------------------------------------------------
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    # -------------------------------------------------------------------------
    def raise_for_status(self) -> None:
        return None

    # -------------------------------------------------------------------------
    def json(self) -> dict[str, object]:
        return self.payload

###############################################################################
def test_rag_readiness_reports_unavailable_ollama(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "build_effective_rag_settings", _settings)

    def fail_get(*args: object, **kwargs: object) -> object:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(readiness.httpx, "get", fail_get)

    result = readiness.check_rag_readiness(requested=True)

    assert result.available is False
    assert result.reason_code == "rag_ollama_unavailable"
    assert "Start Ollama" in (result.message or "")

###############################################################################
def test_rag_readiness_accepts_available_embedding_model(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "build_effective_rag_settings", _settings)
    monkeypatch.setattr(
        readiness.httpx,
        "get",
        lambda *args, **kwargs: _Response(
            {"models": [{"name": "nomic-embed-text:latest"}]}
        ),
    )

    result = readiness.check_rag_readiness(requested=True)

    assert result.available is True
    assert result.backend == "ollama"
    assert result.model == "nomic-embed-text:latest"

###############################################################################
def test_rag_readiness_skips_probe_when_not_requested(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "build_effective_rag_settings", _settings)
    monkeypatch.setattr(
        readiness.httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Ollama should not be probed")
        ),
    )

    result = readiness.check_rag_readiness(requested=False)

    assert result.available is True
    assert result.requested is False
