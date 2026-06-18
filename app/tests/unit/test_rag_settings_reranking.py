from __future__ import annotations

from configurations.management import (
    EnvironmentSnapshot,
    build_settings_payload_from_json,
)


###############################################################################
def _env() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        ollama_url="http://localhost:11434",
        ollama_host="localhost",
        ollama_port=11434,
    )


###############################################################################
def test_build_rag_settings_reads_retrieval_counts() -> None:
    payload = build_settings_payload_from_json(
        {
            "rag": {
                "use_reranking": True,
                "retrieval_candidate_count": 100,
                "retrieval_selected_count": 10,
                "use_hybrid_search": True,
                "reranker_model": "cross-encoder/test-model",
                "hybrid_vector_weight": 0.7,
                "hybrid_text_weight": 0.3,
                "embedding_backend": "ollama",
                "ollama_embedding_model": "nomic-embed-text:latest",
                "cloud_provider": "openai",
                "cloud_model": "gpt-4.1-mini",
            }
        },
        _env(),
    )
    settings = payload["rag"]

    assert settings["use_reranking"] is True
    assert settings["retrieval_candidate_count"] == 100
    assert settings["retrieval_selected_count"] == 10
    assert settings["use_hybrid_search"] is True
    assert settings["reranker_model"] == "cross-encoder/test-model"
    assert settings["hybrid_vector_weight"] == 0.7
    assert settings["hybrid_text_weight"] == 0.3
    assert settings["embedding_backend"] == "ollama"
    assert settings["ollama_embedding_model"] == "nomic-embed-text:latest"


###############################################################################
def test_build_rag_settings_enforces_candidate_floor() -> None:
    payload = build_settings_payload_from_json(
        {"rag": {"retrieval_candidate_count": 3, "retrieval_selected_count": 10}},
        _env(),
    )
    settings = payload["rag"]
    assert settings["retrieval_selected_count"] == 10
    assert settings["retrieval_candidate_count"] == 10


###############################################################################
def test_build_rag_settings_defaults_to_lightweight_reranker_profile() -> None:
    payload = build_settings_payload_from_json({"rag": {}}, _env())
    settings = payload["rag"]

    assert settings["reranker_model"] == "lightweight-balanced-v1"
