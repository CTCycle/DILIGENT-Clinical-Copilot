"""
Pytest configuration for DILIGENT E2E tests.
Provides fixtures for Playwright page objects and API client.
"""

import os
import sys
import threading
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
SERVER_DIR = APP_DIR / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


def _normalize_host_for_url(host: str) -> str:
    if host in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return host


def _build_base_url(
    host_env: str, port_env: str, default_host: str, default_port: str
) -> str:
    host = _normalize_host_for_url(os.getenv(host_env, default_host))
    port = os.getenv(port_env, default_port)
    return f"http://{host}:{port}"


# Base URLs - prefer explicit env vars, then fall back to host/port pairs.
UI_BASE_URL = (
    os.getenv("APP_TEST_FRONTEND_URL")
    or os.getenv("UI_BASE_URL")
    or os.getenv("UI_URL")
    or _build_base_url("UI_HOST", "UI_PORT", "127.0.0.1", "7861")
)
API_BASE_URL = (
    os.getenv("APP_TEST_BACKEND_URL")
    or os.getenv("API_BASE_URL")
    or _build_base_url("FASTAPI_HOST", "FASTAPI_PORT", "127.0.0.1", "8000")
)


@pytest.fixture(scope="session")
def base_url() -> str:
    """Returns the base URL of the UI."""
    return UI_BASE_URL


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Returns the base URL of the API."""
    return API_BASE_URL


@pytest.fixture
def api_context(playwright):
    """
    Creates an API request context for making direct HTTP calls.
    Useful for testing backend endpoints independently of the UI.
    """
    context = playwright.request.new_context(base_url=API_BASE_URL)
    yield context
    context.dispose()


@pytest.fixture(autouse=True, scope="session")
def _patch_asyncio_run_for_nested_loops():
    """
    Make asyncio.run() resilient when tests execute under an already-running loop.
    Several unit tests use asyncio.run() from synchronous test bodies.
    """
    import asyncio

    original_run = asyncio.run

    def safe_run(coro, *args, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return original_run(coro, *args, **kwargs)

        box: dict[str, object] = {}

        def _runner() -> None:
            try:
                box["result"] = original_run(coro, *args, **kwargs)
            except BaseException as exc:  # propagate original test error
                box["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in box:
            raise box["error"]  # type: ignore[misc]
        return box.get("result")

    asyncio.run = safe_run
    try:
        yield
    finally:
        asyncio.run = original_run
