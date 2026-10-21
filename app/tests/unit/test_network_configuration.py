# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from types import SimpleNamespace

import pytest
from configurations.network import parse_cors_allow_origins
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import app as server_app_module


def test_cors_origins_parse_and_normalize_exact_http_origins() -> None:
    assert parse_cors_allow_origins(None) == ()
    assert parse_cors_allow_origins("  ") == ()
    assert parse_cors_allow_origins(
        "http://localhost:9847, https://example.org:443/, http://localhost:9847"
    ) == ("http://localhost:9847", "https://example.org")


@pytest.mark.parametrize(
    "value",
    [
        "*",
        "https://*.example.org",
        "https://example.org/path",
        "https://user:secret@example.org",
        "ftp://example.org",
        "https://example.org:99999",
    ],
)
def test_cors_origins_reject_wildcards_and_non_origins(value: str) -> None:
    with pytest.raises(ValueError, match="CORS_ALLOW_ORIGINS"):
        parse_cors_allow_origins(value)


def _create_app(monkeypatch, *, desktop_enabled: bool):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(server_app_module, "initialize_settings", lambda: None)
    monkeypatch.setattr(
        server_app_module,
        "get_desktop_runtime_service",
        lambda: SimpleNamespace(
            security=SimpleNamespace(enabled=desktop_enabled),
        ),
    )
    return server_app_module.create_app()


def test_source_mode_cors_uses_only_configured_origins(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://localhost:9847")
    application = _create_app(monkeypatch, desktop_enabled=False)
    client = TestClient(application)

    allowed = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:9847",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:9847"

    rejected = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:9848",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers


def test_empty_cors_configuration_keeps_same_origin_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    application = _create_app(monkeypatch, desktop_enabled=False)

    assert all(middleware.cls is not CORSMiddleware for middleware in application.user_middleware)


def test_packaged_desktop_ignores_source_mode_cors_configuration(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "this is invalid")
    application = _create_app(monkeypatch, desktop_enabled=True)

    assert all(middleware.cls is not CORSMiddleware for middleware in application.user_middleware)
