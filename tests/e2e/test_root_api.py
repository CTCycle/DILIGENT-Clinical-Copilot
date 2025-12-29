"""
E2E tests for root and OpenAPI endpoints.
"""
from __future__ import annotations

from playwright.sync_api import APIRequestContext


def test_root_redirects_to_docs(api_context: APIRequestContext):
    response = api_context.get("/")
    assert response.ok
    assert response.url.endswith("/docs")


def test_docs_available(api_context: APIRequestContext):
    response = api_context.get("/docs")
    assert response.ok
    assert "swagger" in response.text().lower()


def test_openapi_contains_core_routes(api_context: APIRequestContext):
    response = api_context.get("/openapi.json")
    assert response.ok

    payload = response.json()
    paths = payload.get("paths", {})
    assert "/clinical" in paths
    assert "/models/list" in paths
    assert "/browser/sessions" in paths
