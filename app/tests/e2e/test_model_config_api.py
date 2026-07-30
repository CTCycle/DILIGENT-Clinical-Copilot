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
    assert "use_cloud_services" in payload
    assert "llm_provider" in payload
    assert "cloud_model" in payload
    assert "clinical_model" in payload
    assert "text_extraction_model" in payload
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
        data={"rag_settings": {"use_hybrid_search": True, "future_setting": True}},
    )
    assert response.status == 422
    detail = response.json().get("detail") or []
    assert any("future_setting" in str(item.get("loc", [])) for item in detail)

###############################################################################
def test_model_config_put_returns_persisted_values_without_catalog_refresh(
    api_context: APIRequestContext,
):
    response = api_context.put(
        "/api/model-config",
        data={"ollama_reasoning": True},
    )
    assert response.status == 200
    payload = response.json()
    assert payload["ollama_reasoning"] is True
    assert "updated_at" in payload
    assert "local_models" not in payload
    assert "cloud_providers" not in payload
    assert "embedding_runtime" not in payload
    assert "embedding_index" not in payload

###############################################################################
