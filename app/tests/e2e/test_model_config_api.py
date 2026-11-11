# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""
E2E tests for the model configuration API endpoints.
"""

from __future__ import annotations

from playwright.sync_api import APIRequestContext


###############################################################################
def test_model_config_get_returns_runtime_payload(api_context: APIRequestContext):
    response = api_context.get("/api/model-config")
    assert response.status == 200
    assert "no-store" in response.headers["cache-control"]

    payload = response.json()
    # Execution settings (models, roles, reasoning, seed) live on agents.
    for removed in (
        "use_cloud_services",
        "llm_provider",
        "cloud_model",
        "clinical_model",
        "text_extraction_model",
        "revision_model",
        "timeline_model",
        "reasoning_level",
        "ollama_seed",
    ):
        assert removed not in payload
    assert "chunk_size" in payload["rag_settings"]
    assert "use_reranking" not in payload["rag_settings"]
    assert payload["local_catalog"]["status"] in {
        "available",
        "cached",
        "not_loaded",
        "unavailable",
        "authentication_required",
    }
    assert "cloud_temperature" not in payload
    assert "ollama_temperature" not in payload


###############################################################################
def test_model_config_put_rejects_agent_owned_fields(
    api_context: APIRequestContext,
):
    response = api_context.put(
        "/api/model-config",
        data={"use_cloud_services": True},
    )
    assert response.status == 422
    detail = response.json().get("detail") or []
    assert any("use_cloud_services" in str(item.get("loc", [])) for item in detail)


###############################################################################
def test_model_config_put_rejects_removed_temperature_field(
    api_context: APIRequestContext,
):
    response = api_context.put(
        "/api/model-config",
        data={"cloud_temperature": 2.5},
    )
    assert response.status == 422
    payload = response.json()
    detail = payload.get("detail") or []
    assert detail
    assert any("cloud_temperature" in str(item.get("loc", [])) for item in detail)


###############################################################################
def test_model_config_put_rejects_unknown_nested_rag_field(
    api_context: APIRequestContext,
):
    response = api_context.put(
        "/api/model-config",
        data={"rag_settings": {"chunk_size": 512, "use_hybrid_search": True}},
    )
    assert response.status == 422
    detail = response.json().get("detail") or []
    assert any("use_hybrid_search" in str(item.get("loc", [])) for item in detail)


###############################################################################
def test_model_config_put_returns_persisted_values_without_catalog_refresh(
    api_context: APIRequestContext,
):
    original = api_context.get("/api/model-config")
    assert original.status == 200
    original_rag = original.json()["rag_settings"]
    next_batch = 16 if original_rag["embedding_batch_size"] != 16 else 8
    try:
        response = api_context.put(
            "/api/model-config",
            data={"rag_settings": {"embedding_batch_size": next_batch}},
        )
        assert response.status == 200
        payload = response.json()
        assert payload["rag_settings"]["embedding_batch_size"] == next_batch
        assert "updated_at" in payload
        assert "local_models" not in payload
        assert "cloud_providers" not in payload
        assert "embedding_runtime" not in payload
        assert "embedding_index" not in payload
    finally:
        restored = api_context.put(
            "/api/model-config",
            data={
                "rag_settings": {
                    "embedding_batch_size": original_rag["embedding_batch_size"]
                }
            },
        )
        assert restored.status == 200


###############################################################################
