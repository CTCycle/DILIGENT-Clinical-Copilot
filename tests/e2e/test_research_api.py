from __future__ import annotations

import pytest
from playwright.sync_api import APIRequestContext


def get_active_tavily_keys(api_context: APIRequestContext) -> list[dict]:
    response = api_context.get("/access-keys?provider=tavily")
    if response.status != 200:
        return []
    payload = response.json()
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict) and row.get("is_active") is True]


def test_research_returns_503_when_key_missing(api_context: APIRequestContext) -> None:
    if get_active_tavily_keys(api_context):
        pytest.skip("An active Tavily access key is configured; missing-key path not applicable.")

    response = api_context.post("/research", data={"question": "acetaminophen DILI evidence"})
    assert response.status == 503
    assert "No active Tavily access key configured" in response.text()

    alias_response = api_context.post("/api/research", data={"question": "acetaminophen DILI evidence"})
    assert alias_response.status == 503


def test_research_smoke_when_key_available(api_context: APIRequestContext) -> None:
    if not get_active_tavily_keys(api_context):
        pytest.skip("No active Tavily access key is configured.")

    response = api_context.post("/research", data={"question": "acetaminophen DILI evidence"})
    if response.status == 503:
        pytest.skip("Active Tavily key exists but backend could not load it.")
    assert response.status == 200

    payload = response.json()
    assert isinstance(payload.get("answer"), str)
    assert isinstance(payload.get("sources"), list)
    assert isinstance(payload.get("citations"), list)
