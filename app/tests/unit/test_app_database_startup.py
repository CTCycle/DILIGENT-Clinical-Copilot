# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from fastapi import FastAPI
import app as server_app_module
import pytest
from configurations.startup import get_server_settings
from configurations.database_config import DatabaseConfiguration
from services import startup_validation
from services.runtime import database_switch as database_switch_module
from services.runtime.jobs import JobManager


###############################################################################
async def _run_inline_async_work(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    return function(*args, **kwargs)


###############################################################################
def _run_coroutine_without_loop(coroutine: Coroutine[Any, Any, Any]) -> Any:
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    coroutine.close()
    raise RuntimeError("The test coroutine unexpectedly required an event loop.")


###############################################################################
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
            shutdown=lambda timeout: events.append("close") or True,
        ),
    )
    monkeypatch.setattr(
        server_app_module,
        "reconcile_interrupted_revision_jobs",
        lambda _jobs: events.append("revision-recovery"),
    )
    monkeypatch.setattr(
        server_app_module,
        "get_clinical_scheduler",
        lambda: SimpleNamespace(
            reconcile_interrupted=lambda: None,
            dispatch=lambda: None,
        ),
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
    monkeypatch.setattr(
        server_app_module.ollama_process_manager, "stop", lambda: False
    )
    monkeypatch.setattr(
        server_app_module,
        "RepositoryContext",
        SimpleNamespace(
            create=lambda: SimpleNamespace(
                engine=SimpleNamespace(dispose=lambda: None)
            )
        ),
    )
    monkeypatch.setattr(asyncio, "to_thread", _run_inline_async_work)

    async def exercise() -> None:
        async with server_app_module.app_lifespan(FastAPI()):
            events.append("running")

    _run_coroutine_without_loop(exercise())
    return events


###############################################################################
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


###############################################################################
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


###############################################################################
def test_startup_validation_failure_still_releases_application_resources(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    events: list[str] = []
    settings = SimpleNamespace(database=SimpleNamespace(backend="sqlite"))
    monkeypatch.setattr(server_app_module, "get_server_settings", lambda: settings)
    monkeypatch.setattr(server_app_module, "ensure_database_ready", lambda _db: None)
    monkeypatch.setattr(
        server_app_module, "reload_persisted_settings", lambda: settings
    )
    monkeypatch.setattr(
        server_app_module,
        "get_job_manager",
        lambda: SimpleNamespace(begin_startup=lambda: None),
    )
    monkeypatch.setattr(
        server_app_module, "reconcile_interrupted_revision_jobs", lambda _jobs: None
    )
    monkeypatch.setattr(
        server_app_module,
        "get_clinical_scheduler",
        lambda: SimpleNamespace(reconcile_interrupted=lambda: None),
    )
    monkeypatch.setattr(
        server_app_module, "initialize_reference_catalog_provider", lambda: None
    )

    def fail_validation(_settings) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("startup validation failed")

    async def release_resources() -> None:
        events.append("released")

    monkeypatch.setattr(server_app_module, "run_startup_validations", fail_validation)
    monkeypatch.setattr(
        server_app_module, "_release_application_resources", release_resources
    )
    monkeypatch.setattr(asyncio, "to_thread", _run_inline_async_work)

    async def exercise() -> None:
        async with server_app_module.app_lifespan(FastAPI()):
            pytest.fail("The application must not start after validation fails")

    with pytest.raises(RuntimeError, match="startup validation failed"):
        _run_coroutine_without_loop(exercise())

    assert events == ["released"]


###############################################################################
def test_shutdown_continues_after_embedding_and_ollama_cleanup_failures(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    shutdown = Mock(return_value=True)
    stop_ollama = Mock(side_effect=OSError("stop failed"))
    dispose_engine = Mock()
    monkeypatch.setattr(
        server_app_module, "get_job_manager", lambda: SimpleNamespace(shutdown=shutdown)
    )
    monkeypatch.setattr(
        server_app_module,
        "close_embedding_runtime",
        Mock(side_effect=RuntimeError("embedding close failed")),
    )
    monkeypatch.setattr(server_app_module.ollama_process_manager, "stop", stop_ollama)
    monkeypatch.setattr(
        server_app_module,
        "RepositoryContext",
        SimpleNamespace(create=lambda: SimpleNamespace(engine=SimpleNamespace(dispose=dispose_engine))),
    )
    monkeypatch.setattr(asyncio, "to_thread", _run_inline_async_work)

    _run_coroutine_without_loop(server_app_module._release_application_resources())

    shutdown.assert_called_once_with(5.0)
    stop_ollama.assert_called_once()
    dispose_engine.assert_called_once()


###############################################################################
def test_database_switch_rechecks_sessions_after_active_requests_drain(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    events: list[str] = []
    previous = DatabaseConfiguration(sqlite_path="previous.sqlite")
    candidate = DatabaseConfiguration(sqlite_path="candidate.sqlite")
    previous_repository = SimpleNamespace(engine=SimpleNamespace(dispose=Mock()))
    job_manager = JobManager()
    counts = iter((0, 1))

    class SessionRepository:
        def __init__(self, _context) -> None:  # type: ignore[no-untyped-def]
            pass

        def count_sessions_with_status(self, *_statuses: str) -> int:
            events.append("count-sessions")
            return next(counts)

    request_gate = SimpleNamespace(
        begin_switch=lambda: events.append("begin-switch"),
        wait_for_idle=lambda timeout: events.append("drain-requests") or True,
        end_switch=lambda: events.append("end-switch"),
    )
    monkeypatch.setattr(
        database_switch_module,
        "database_settings",
        lambda _configuration: object(),
    )
    monkeypatch.setattr(
        database_switch_module, "load_database_configuration", lambda: previous
    )
    monkeypatch.setattr(
        database_switch_module,
        "get_default_repository",
        lambda: previous_repository,
    )
    monkeypatch.setattr(
        database_switch_module, "get_job_manager", lambda: job_manager
    )
    monkeypatch.setattr(
        database_switch_module,
        "RepositoryContext",
        SimpleNamespace(create=lambda: object()),
    )
    monkeypatch.setattr(
        database_switch_module, "ClinicalSessionRepository", SessionRepository
    )
    monkeypatch.setattr(
        database_switch_module,
        "get_clinical_scheduler",
        lambda: SimpleNamespace(dispatch=lambda: events.append("dispatch")),
    )
    ensure_database_ready = Mock()
    monkeypatch.setattr(
        database_switch_module, "ensure_database_ready", ensure_database_ready
    )

    manager = database_switch_module.DatabaseRuntimeManager()
    manager.request_gate = request_gate

    with pytest.raises(database_switch_module.DatabaseSwitchBusyError):
        manager.switch_configuration(candidate)

    assert events.index("drain-requests") < events.index("count-sessions", 1)
    assert events[-2:] == ["end-switch", "dispatch"]
    ensure_database_ready.assert_not_called()
    previous_repository.engine.dispose.assert_not_called()


###############################################################################
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


###############################################################################
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
