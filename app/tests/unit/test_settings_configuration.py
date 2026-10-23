# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest
from configurations import database_config
from configurations.management import environment_snapshot_from_os_env
from types import SimpleNamespace
from configurations.network import parse_cors_allow_origins
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
import app as server_app_module
from datetime import date
from typing import Any
from api import ollama as ollama_routes
from api import session as session_routes
from configurations.startup import get_server_settings
from domain.clinical import ClinicalSessionRequest
from services.clinical import job_progress as clinical_job_progress
from services.session import session_workflow
from services.session import session_shared

def _patch_configuration_paths(monkeypatch, tmp_path: Path) -> Path:  # type: ignore[no-untyped-def]
    config_path = tmp_path / "database_config.json"
    env_path = tmp_path / "settings" / ".env"
    env_path.parent.mkdir(parents=True)
    monkeypatch.setattr(database_config, "DATABASE_CONFIG_FILE_PATH", config_path)
    monkeypatch.setattr(database_config.paths, "ENV_FILE_PATH", env_path)
    return config_path

def test_missing_configuration_file_defaults_to_sqlite_and_writes_file(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    config_path = _patch_configuration_paths(monkeypatch, tmp_path)
    env_path = database_config.paths.ENV_FILE_PATH
    env_path.write_text(
        "FASTAPI_PORT=7777\n"
        "DATABASE_BACKEND=postgresql\n"
        "DATABASE_HOST=ignored.invalid\n",
        encoding="utf-8",
    )

    loaded = database_config.load_database_configuration()

    assert loaded.backend == "sqlite"
    assert loaded.sqlite_path == str(database_config.paths.DATA_PATH / "database.db")
    assert config_path.is_file()
    # Environment variables never configure database access.
    remaining_env = env_path.read_text(encoding="utf-8")
    assert "FASTAPI_PORT=7777" in remaining_env
    assert "DATABASE_HOST=ignored.invalid" in remaining_env

def test_database_file_is_authoritative_over_environment(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config_path = _patch_configuration_paths(monkeypatch, tmp_path)
    expected = database_config.DatabaseConfiguration(
        backend="sqlite",
        sqlite_path=str(tmp_path / "selected.sqlite3"),
    )
    database_config.save_database_configuration(expected)
    monkeypatch.setenv("DATABASE_BACKEND", "postgresql")
    monkeypatch.setenv("DATABASE_HOST", "ignored.invalid")

    loaded = database_config.load_database_configuration()

    assert loaded == expected
    assert json.loads(config_path.read_text(encoding="utf-8"))["database"]["backend"] == "sqlite"

def test_corrupt_database_file_fails_closed_without_legacy_fallback(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    config_path = _patch_configuration_paths(monkeypatch, tmp_path)
    config_path.write_text('{"schema_version":99}', encoding="utf-8")
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")

    with pytest.raises(RuntimeError, match="Database configuration is invalid"):
        database_config.load_database_configuration()

    assert config_path.read_text(encoding="utf-8") == '{"schema_version":99}'

def test_database_configuration_is_excluded_from_general_environment_snapshot(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DATABASE_URL", "postgresql://ignored:secret@example.invalid/db")
    monkeypatch.setenv("DATABASE_BACKEND", "postgresql")
    snapshot = environment_snapshot_from_os_env()

    assert "database" not in snapshot.model_dump()
    assert "DATABASE_URL" not in str(snapshot.model_dump())

@pytest.mark.skipif(os.name != "nt", reason="Credential protection uses Windows DPAPI")
def test_postgres_password_is_dpapi_protected_in_configuration_file(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    config_path = _patch_configuration_paths(monkeypatch, tmp_path)
    secret = "temporary-test-secret"
    protected = database_config.protect_database_password(secret)
    configuration = database_config.DatabaseConfiguration(
        backend="postgresql",
        host="127.0.0.1",
        database_name="configuration_test",
        username="test_user",
        password_protected=protected,
    )

    database_config.save_database_configuration(configuration)
    content = config_path.read_text(encoding="utf-8")

    assert secret not in content
    assert database_config.unprotect_database_password(protected) == secret
    assert Path(config_path).read_bytes()

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

    cors = [middleware for middleware in application.user_middleware if middleware.cls is CORSMiddleware]
    assert len(cors) == 1
    assert cors[0].kwargs["allow_origins"] == ["http://tauri.localhost"]
    assert cors[0].kwargs["allow_headers"] == [
        "Content-Type",
        "X-DILIGENT-Desktop-Session",
    ]

def test_packaged_desktop_cors_preflight_is_exactly_tauri_origin(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DILIGENT_DESKTOP", "true")
    monkeypatch.setenv("DILIGENT_DESKTOP_SESSION_SECRET", "session-secret")
    monkeypatch.setenv("DILIGENT_DESKTOP_PORT", "48123")
    monkeypatch.setattr(server_app_module, "initialize_settings", lambda: None)
    from common.security.desktop import DesktopSessionSecurity
    from services.runtime.desktop import DesktopRuntimeService

    runtime = DesktopRuntimeService(DesktopSessionSecurity())
    monkeypatch.setattr(server_app_module, "get_desktop_runtime_service", lambda: runtime)
    client = TestClient(server_app_module.create_app())

    allowed = client.options(
        "/api/model-config",
        headers={
            "Origin": "http://tauri.localhost",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-diligent-desktop-session",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://tauri.localhost"
    assert "x-diligent-desktop-session" in allowed.headers["access-control-allow-headers"].casefold()

    rejected = client.options(
        "/api/model-config",
        headers={
            "Origin": "http://127.0.0.1:48123",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-diligent-desktop-session",
        },
    )
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers

def get_route_owner(router: Any, route_path: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", "").endswith(route_path):
            owner = getattr(route.endpoint, "__self__", None)
            if owner is not None:
                return owner
    raise AssertionError(f"Route not found: {route_path}")

class JobManagerStub:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.job_type: str | None = None

    # -------------------------------------------------------------------------
    def is_job_running(
        self, job_type: str | None = None, *, scope_key: str | None = None
    ) -> bool:
        _ = scope_key
        return False

    # -------------------------------------------------------------------------
    def start_job(
        self,
        job_type: str,
        runner: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        scope_key: str | None = None,
    ) -> str:
        _ = scope_key
        self.job_type = job_type
        return "job-1234"

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        if self.job_type is None:
            return None
        return {
            "job_id": job_id,
            "job_type": self.job_type,
            "status": "running",
            "progress": 0.0,
            "result": None,
            "error": None,
        }

def test_start_clinical_job_uses_centralized_poll_interval(monkeypatch) -> None:
    job_manager_stub = JobManagerStub()
    endpoint = get_route_owner(session_routes.router, "/clinical/jobs")
    monkeypatch.setattr(endpoint.service, "job_manager", job_manager_stub)
    monkeypatch.setattr(
        endpoint.service,
        "validate_clinical_input",
        lambda _: SimpleNamespace(ready=True, blocking_issues=[]),
    )
    monkeypatch.setattr(
        session_workflow.LLMRuntimeConfig,
        "is_cloud_enabled",
        staticmethod(lambda: False),
    )
    monkeypatch.setattr(
        endpoint.service,
        "apply_persisted_runtime_configuration",
        lambda: None,
    )
    monkeypatch.setattr(
        endpoint.service,
        "validate_assessment_prerequisites_without_llm",
        lambda _: SimpleNamespace(missing_required_sections=[], malformed_sections=[]),
    )

    monkeypatch.setattr(
        endpoint.service,
        "prepare_structured_clinical_input",
        lambda request_payload, *, parse_result: {
            "normalized_document": object(),
            "section_extraction": object(),
            "patient_payload": object(),
        },
    )
    monkeypatch.setattr(
        endpoint.service, "build_patient_payload", lambda request_payload: object()
    )
    monkeypatch.setattr(
        endpoint.service, "ensure_submission_requirements", lambda _: None
    )

    response = endpoint.service.start_clinical_job(
        ClinicalSessionRequest(
            clinical_input=(
                "Anamnesis: Clinical context\n"
                "Drugs: Acetaminophen 500 mg 1 - 0 - 0 - 0 po started from 01/01/2024\n"
                "Lab analysis: ALT 300 U/L"
            ),
            visit_date=date(2026, 4, 24),
            selected_model_providers=[],
        )
    )

    assert response.poll_interval == get_server_settings().jobs.polling_interval

def test_start_pull_job_uses_centralized_poll_interval(monkeypatch) -> None:
    job_manager_stub = JobManagerStub()
    endpoint = get_route_owner(ollama_routes.router, "/pull/jobs")
    monkeypatch.setattr(endpoint.service, "job_manager", job_manager_stub)

    response = endpoint.start_pull_job(
        name="llama3.1:8b",
        stream=False,
    )

    assert response.poll_interval == get_server_settings().jobs.polling_interval

def test_clinical_progress_callback_raises_when_stop_requested(monkeypatch) -> None:

    ###############################################################################
    class StopRequestedJobManagerStub:

        # -------------------------------------------------------------------------
        def should_stop(self, job_id: str) -> bool:
            return True

    try:
        session_shared.report_clinical_job_progress(
            "llm_analysis",
            75.0,
            job_manager=StopRequestedJobManagerStub(),
            job_id="job-cancelled",
        )
    except clinical_job_progress.ClinicalJobCancelled:
        return
    raise AssertionError("Expected ClinicalJobCancelled when stop is requested.")
