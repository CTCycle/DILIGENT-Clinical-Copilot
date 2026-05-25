"""
E2E tests for the model configuration API endpoints.
"""

from __future__ import annotations

from playwright.sync_api import APIRequestContext


def test_model_config_get_returns_runtime_payload(api_context: APIRequestContext):
    response = api_context.get("/api/model-config")
    assert response.status == 200

    payload = response.json()
    assert "use_cloud_services" in payload
    assert "llm_provider" in payload
    assert "cloud_model" in payload
    assert "clinical_model" in payload
    assert "text_extraction_model" in payload
    assert "cloud_temperature" in payload
    assert "ollama_temperature" in payload


def test_model_config_put_rejects_out_of_range_temperature(
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


def test_model_config_put_accepts_current_temperature(
    api_context: APIRequestContext,
):
    current = api_context.get("/api/model-config")
    assert current.status == 200
    current_payload = current.json()
    current_temperature = float(current_payload["cloud_temperature"])

    response = api_context.put(
        "/api/model-config",
        data={"cloud_temperature": current_temperature},
    )
    assert response.status == 200
    payload = response.json()
    assert float(payload["cloud_temperature"]) == current_temperature
