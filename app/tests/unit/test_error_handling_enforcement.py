from __future__ import annotations

import ast
import pathlib
import time
from typing import Any

from api import access_keys as access_keys_api
from api import data_inspection as data_inspection_api
from api import ollama as ollama_api
from api.error_handling import (
    REQUEST_ID_HEADER,
    register_error_handling,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.llm.ollama_client import OllamaError
from services.runtime.jobs import JobManager

import app as server_app_module

EXCLUDED_DIRS = {
    "__pycache__",
    ".venv",
    ".uv-cache",
    ".pytest_cache",
    "node_modules",
    "dist",
}
APP_ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVER_ROOT = APP_ROOT / "server"


def get_route_service(router: Any, route_path: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", "").endswith(route_path):
            endpoint_owner = getattr(route.endpoint, "__self__", None)
            if endpoint_owner is not None:
                return endpoint_owner.service
    raise AssertionError(f"Route not found: {route_path}")


# -----------------------------------------------------------------------------
def test_backend_httpx_asyncclient_calls_require_explicit_timeout() -> None:
    root = SERVER_ROOT
    assert list(root.rglob("*.py")), f"No backend Python files found under {root}"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if not EXCLUDED_DIRS.isdisjoint(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "AsyncClient":
                continue
            if not isinstance(func.value, ast.Name) or func.value.id != "httpx":
                continue
            has_timeout = any(
                keyword.arg == "timeout"
                for keyword in node.keywords
                if keyword.arg is not None
            )
            if not has_timeout:
                violations.append(f"{path}:{node.lineno}")

    assert not violations, (
        "All httpx.AsyncClient calls must include timeout:\n" + "\n".join(violations)
    )


# -----------------------------------------------------------------------------
def wait_for_terminal_job_state(
    manager: JobManager,
    job_id: str,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = manager.get_job_status(job_id)
        if payload is None:
            raise AssertionError("Expected job state to exist.")
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for terminal job state.")


# -----------------------------------------------------------------------------
def test_unhandled_exception_is_masked_and_has_request_id() -> None:
    app = FastAPI()
    register_error_handling(app)

    @app.get("/boom")
    def boom() -> dict[str, str]:
        raise RuntimeError("traceback with token=secret-value")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "Request could not be completed. Please retry."
    assert "request_id" in payload
    assert payload["retryable"] is True
    assert response.headers.get(REQUEST_ID_HEADER) == payload["request_id"]


# -----------------------------------------------------------------------------
def test_validation_error_keeps_detail_and_request_id() -> None:
    app = FastAPI()
    register_error_handling(app)

    @app.get("/validation")
    def validation_required(limit: int) -> dict[str, int]:
        return {"limit": limit}

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/validation", params={"limit": "NaN"})

    assert response.status_code == 422
    payload = response.json()
    assert isinstance(payload.get("detail"), list)
    assert payload.get("request_id")
    assert payload.get("retryable") is False


# -----------------------------------------------------------------------------
def test_validation_error_with_value_error_context_is_json_safe() -> None:
    with TestClient(server_app_module.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/access-keys",
            json={"provider": "openai", "access_key": "short"},
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"][0]["ctx"]["error"] == "access_key is too short"


# -----------------------------------------------------------------------------
def test_job_manager_masks_sensitive_error_details() -> None:
    manager = JobManager()

    def failing_runner() -> dict[str, Any]:
        raise RuntimeError("Traceback: leaked token and stack data")

    job_id = manager.start_job("diagnostic", failing_runner)
    payload = wait_for_terminal_job_state(manager, job_id)

    assert payload["status"] == "failed"
    assert payload["error"] == "Operation failed unexpectedly. Please retry."


# -----------------------------------------------------------------------------
def test_job_manager_reports_timeout_message() -> None:
    manager = JobManager()

    def timeout_runner() -> dict[str, Any]:
        raise TimeoutError("read timeout")

    job_id = manager.start_job("diagnostic", timeout_runner)
    payload = wait_for_terminal_job_state(manager, job_id)

    assert payload["status"] == "failed"
    assert payload["error"] == "Operation timed out. Please retry."


# -----------------------------------------------------------------------------
def test_access_key_endpoint_sanitizes_dependency_failure(monkeypatch) -> None:
    def fake_create_key(provider: str, access_key: str):
        raise RuntimeError("encryption material registry unavailable token=abc123")

    service = get_route_service(access_keys_api.router, "")
    monkeypatch.setattr(service.serializer, "create_key", fake_create_key)

    with TestClient(server_app_module.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/access-keys",
            json={"provider": "openai", "access_key": "sk-test-value-secret"},
        )

    assert response.status_code == 503
    payload = response.json()
    assert (
        payload["detail"] == "Access key service is unavailable. Please retry shortly."
    )


# -----------------------------------------------------------------------------
def test_data_inspection_endpoint_sanitizes_runtime_failure(monkeypatch) -> None:
    def fake_start_update_job(job_type: str, overrides: dict[str, Any] | None = None):
        raise RuntimeError("traceback secret=value")

    service = get_route_service(data_inspection_api.router, "/rxnav/jobs")
    monkeypatch.setattr(service, "start_update_job", fake_start_update_job)

    with TestClient(server_app_module.app, raise_server_exceptions=False) as client:
        response = client.post("/api/inspection/rxnav/jobs")

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "Update job could not start. Please retry."


# -----------------------------------------------------------------------------
def test_ollama_endpoint_sanitizes_provider_error(monkeypatch) -> None:
    class FakeOllamaClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def list_models(self):
            raise OllamaError("stack trace token=internal")

    service = get_route_service(ollama_api.router, "/list")
    monkeypatch.setattr(service, "client_factory", FakeOllamaClient)

    with TestClient(server_app_module.app, raise_server_exceptions=False) as client:
        response = client.get("/api/models/list")

    assert response.status_code == 502
    payload = response.json()
    assert (
        payload["detail"]
        == "Ollama service is unavailable. Verify Ollama is running and retry."
    )
