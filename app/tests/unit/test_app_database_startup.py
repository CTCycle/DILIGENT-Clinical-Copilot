# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from fastapi import FastAPI
import app as server_app_module
import pytest
from configurations.startup import get_server_settings
from services import startup_validation

def _run_lifespan(database_backend: str, monkeypatch) -> list[str]:  # type: ignore[no-untyped-def]
    events: list[str] = []
    settings = SimpleNamespace(database=SimpleNamespace(backend=database_backend))
    monkeypatch.setattr(server_app_module, "get_server_settings", lambda: settings)
    monkeypatch.setattr(
        server_app_module,
        "reload_persisted_settings",
        lambda: events.append("configuration") or settings,
    )
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
        "get_job_manager",
        lambda: SimpleNamespace(
            begin_startup=lambda: events.append("jobs"),
            shutdown=lambda timeout: events.append("close"),
        ),
    )
    monkeypatch.setattr(
        server_app_module,
        "reconcile_interrupted_revision_jobs",
        lambda _jobs: events.append("revision-recovery"),
    )
    monkeypatch.setattr(
        server_app_module,
        "run_startup_validations",
        lambda _settings: events.append("validation"),
    )
    monkeypatch.setattr(
        server_app_module,
        "close_embedding_runtime",
        lambda: None,
    )

    async def exercise() -> None:
        async with server_app_module.app_lifespan(FastAPI()):
            events.append("running")

    asyncio.run(exercise())
    return events

def test_application_startup_synchronizes_sqlite_before_validation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert _run_lifespan("sqlite", monkeypatch) == [
        "database",
        "configuration",
        "jobs",
        "revision-recovery",
        "provider",
        "validation",
        "running",
        "close",
    ]

def test_application_startup_synchronizes_postgresql_before_validation(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    assert _run_lifespan("postgresql", monkeypatch) == [
        "database",
        "configuration",
        "jobs",
        "revision-recovery",
        "provider",
        "validation",
        "running",
        "close",
    ]

class _FakeModelConfigService:

    # -------------------------------------------------------------------------
    def __init__(self, *, clinical_model: str, text_extraction_model: str) -> None:
        self._snapshot = SimpleNamespace(
            clinical_model=clinical_model,
            text_extraction_model=text_extraction_model,
            revision_model=clinical_model,
            timeline_model=text_extraction_model,
        )

    # -------------------------------------------------------------------------
    def load_current_snapshot(self) -> SimpleNamespace:
        return self._snapshot

def test_run_startup_validations_requires_reference_catalogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        startup_validation,
        "get_reference_catalog_snapshot",
        lambda: SimpleNamespace(entries_by_scope={}),
    )
    monkeypatch.setattr(
        startup_validation,
        "ModelConfigService",
        lambda: _FakeModelConfigService(
            clinical_model="clinical-model",
            text_extraction_model="parser-model",
        ),
    )

    with pytest.raises(RuntimeError, match="Reference catalogs"):
        startup_validation.run_startup_validations(get_server_settings())
