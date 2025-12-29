"""
E2E tests for the model management API endpoints.
"""
from __future__ import annotations

from urllib.parse import quote

import pytest
from playwright.sync_api import APIRequestContext


def skip_if_ollama_unavailable(response) -> None:
    if response.status in (502, 504):
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = ""
        reason = "Ollama unavailable for /models endpoints"
        if detail:
            reason = f"{reason}: {detail}"
        pytest.skip(reason)


def test_models_list_returns_payload(api_context: APIRequestContext):
    response = api_context.get("/models/list")
    skip_if_ollama_unavailable(response)

    assert response.ok, f"Expected 200, got {response.status}"
    payload = response.json()
    assert "models" in payload
    assert "count" in payload
    assert isinstance(payload["models"], list)
    assert payload["count"] == len(payload["models"])


def test_models_pull_requires_name(api_context: APIRequestContext):
    response = api_context.get("/models/pull")
    assert response.status == 422


def test_models_pull_noop_when_model_available(api_context: APIRequestContext):
    list_response = api_context.get("/models/list")
    skip_if_ollama_unavailable(list_response)
    assert list_response.ok

    payload = list_response.json()
    models = payload.get("models", [])
    if not models:
        pytest.skip("No local Ollama models available to validate /models/pull.")

    model = models[0]
    pull_response = api_context.get(
        f"/models/pull?name={quote(model, safe='')}&stream=false"
    )
    skip_if_ollama_unavailable(pull_response)

    assert pull_response.ok
    pull_payload = pull_response.json()
    assert pull_payload.get("status") == "success"
    assert pull_payload.get("model") == model
    assert pull_payload.get("pulled") is False
