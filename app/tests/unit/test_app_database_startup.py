from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI

import app as server_app_module


###############################################################################
def _run_lifespan(database_backend: str, monkeypatch) -> list[str]:  # type: ignore[no-untyped-def]
    events: list[str] = []
    settings = SimpleNamespace(database=SimpleNamespace(backend=database_backend))
    monkeypatch.setattr(server_app_module, "get_server_settings", lambda: settings)
    monkeypatch.setattr(
        server_app_module,
        "ensure_database_ready",
        lambda _database: events.append("database"),
    )
    monkeypatch.setattr(
        server_app_module,
        "initialize_reference_catalog_provider",
        lambda: events.append("provider"),
    )
    monkeypatch.setattr(
        server_app_module,
        "run_startup_validations",
        lambda _settings: events.append("validation"),
    )
    monkeypatch.setattr(
        server_app_module,
        "close_embedding_runtime",
        lambda: events.append("close"),
    )

    async def exercise() -> None:
        async with server_app_module.app_lifespan(FastAPI()):
            events.append("running")

    asyncio.run(exercise())
    return events


###############################################################################
def test_application_startup_synchronizes_sqlite_before_validation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert _run_lifespan("sqlite", monkeypatch) == [
        "database",
        "provider",
        "validation",
        "running",
        "close",
    ]


###############################################################################
def test_application_startup_synchronizes_postgresql_before_validation(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    assert _run_lifespan("postgresql", monkeypatch) == [
        "database",
        "provider",
        "validation",
        "running",
        "close",
    ]
