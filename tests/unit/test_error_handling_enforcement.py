from __future__ import annotations

import ast
import asyncio
import pathlib
import time
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from DILIGENT.server.api.error_handling import REQUEST_ID_HEADER, register_error_handling
from DILIGENT.server.services.jobs import JobManager
from DILIGENT.server.services.research import tavily as tavily_module
from DILIGENT.server.services.research.tavily import TavilyResearchService


# -----------------------------------------------------------------------------
def test_backend_httpx_asyncclient_calls_require_explicit_timeout() -> None:
    root = pathlib.Path("DILIGENT/server")
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
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
        "All httpx.AsyncClient calls must include timeout:\n"
        + "\n".join(violations)
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
def test_tavily_retry_retries_transient_status(monkeypatch) -> None:
    service = TavilyResearchService()
    service.retry_limit = 2

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
            self.status_code = status_code
            self.payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://unit.test")
                response = httpx.Response(
                    status_code=self.status_code,
                    request=request,
                )
                raise httpx.HTTPStatusError(
                    "status error",
                    request=request,
                    response=response,
                )

        def json(self) -> dict[str, Any]:
            return self.payload

    class FakeAsyncClient:
        call_count = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any]):
            FakeAsyncClient.call_count += 1
            if FakeAsyncClient.call_count == 1:
                return FakeResponse(503, {"detail": "unavailable"})
            return FakeResponse(200, {"results": []})

    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(tavily_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(tavily_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(service, "consume_rate_slot", lambda: None)

    payload = asyncio.run(
        service.post_json_with_retry(
            url="https://unit.test/search",
            payload={"query": "acetaminophen"},
            operation="Tavily search",
        )
    )

    assert payload == {"results": []}
    assert FakeAsyncClient.call_count == 2
    assert len(delays) == 1
