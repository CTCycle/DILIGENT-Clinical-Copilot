from __future__ import annotations

from DILIGENT.server.configurations.bootstrap import build_rag_settings


# -----------------------------------------------------------------------------
def test_build_rag_settings_reads_reranking_keys() -> None:
    settings = build_rag_settings(
        {
            "use_reranking": True,
            "rerank_candidate_k": 100,
            "rerank_top_n": 10,
            "embedding_backend": "ollama",
            "ollama_embedding_model": "nomic-embed-text:latest",
        },
        default_provider="openai",
        default_cloud_model="gpt-4.1-mini",
        default_ollama_host="http://localhost:11434",
    )

    assert settings.use_reranking is True
    assert settings.rerank_candidate_k == 100
    assert settings.rerank_top_n == 10


# -----------------------------------------------------------------------------
def test_build_rag_settings_enforces_candidate_floor() -> None:
    settings = build_rag_settings(
        {
            "use_reranking": True,
            "rerank_candidate_k": 3,
            "rerank_top_n": 7,
            "embedding_backend": "ollama",
            "ollama_embedding_model": "nomic-embed-text:latest",
        },
        default_provider="openai",
        default_cloud_model="gpt-4.1-mini",
        default_ollama_host="http://localhost:11434",
    )

    assert settings.rerank_top_n == 7
    assert settings.rerank_candidate_k == 7
