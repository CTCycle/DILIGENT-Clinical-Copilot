from __future__ import annotations

from domain.keys import SUPPORTED_PROVIDERS

from app import app


def test_no_research_route_registered() -> None:
    assert all(not route.path.startswith("/api/research") for route in app.routes)


def test_no_brave_access_key_provider() -> None:
    assert "brave" not in SUPPORTED_PROVIDERS
