"""
E2E tests for the model management API endpoints.
"""

from __future__ import annotations

import time
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


def wait_for_pull_job(api_context: APIRequestContext, job_id: str) -> dict:
    for _ in range(120):
        status_response = api_context.get(f"/api/models/jobs/{job_id}")
        assert status_response.status == 200
        payload = status_response.json()
        if payload.get("status") in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.25)
    raise AssertionError("Model pull job did not finish in time")


def test_models_list_returns_payload(api_context: APIRequestContext):
    response = api_context.get("/api/models/list")
    skip_if_ollama_unavailable(response)

    assert response.ok, f"Expected 200, got {response.status}"
    payload = response.json()
    assert "models" in payload
    assert "count" in payload
    assert isinstance(payload["models"], list)
    assert payload["count"] == len(payload["models"])


def test_models_pull_job_requires_name(api_context: APIRequestContext):
    response = api_context.post("/api/models/pull/jobs")
    assert response.status == 422


def test_models_pull_job_noop_when_model_available(api_context: APIRequestContext):
    list_response = api_context.get("/api/models/list")
    skip_if_ollama_unavailable(list_response)
    assert list_response.ok

    payload = list_response.json()
    models = payload.get("models", [])
    if not models:
        pytest.skip("No local Ollama models available to validate /models/pull/jobs.")

    model = models[0]
    start_response = api_context.post(
        f"/api/models/pull/jobs?name={quote(model, safe='')}&stream=true"
    )
    skip_if_ollama_unavailable(start_response)
    assert start_response.status == 202

    job_payload = wait_for_pull_job(api_context, start_response.json()["job_id"])
    assert job_payload.get("status") == "completed"
    result = job_payload.get("result") or {}
    assert result.get("model") == model
