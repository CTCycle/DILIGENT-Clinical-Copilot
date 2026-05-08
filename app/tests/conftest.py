"""
Pytest configuration for DILIGENT E2E tests.
Provides fixtures for Playwright page objects and API client.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Coroutine
from typing import Any

import pytest


class CoroutineThreadRunner:
    def __init__(
        self,
        run_callable: Any,
        coro: Coroutine[Any, Any, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.run_callable = run_callable
        self.coro = coro
        self.args = args
        self.kwargs = kwargs
        self.box: dict[str, Any] = {}

    def __call__(self) -> None:
        try:
            self.box["result"] = self.run_callable(
                self.coro,
                *self.args,
                **self.kwargs,
            )
        except BaseException as exc:
            self.box["error"] = exc


def _normalize_host_for_url(host: str) -> str:
    if host in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return host


def _build_base_url(
    host_env: str,
    port_env: str,
    default_host: str,
    default_port: str,
) -> str:
    host = _normalize_host_for_url(os.getenv(host_env, default_host))
    port = os.getenv(port_env, default_port)
    return f"http://{host}:{port}"


def run_coroutine_in_thread(
    run_callable: Any,
    coro: Coroutine[Any, Any, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    runner = CoroutineThreadRunner(run_callable, coro, args, kwargs)
    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in runner.box:
        raise runner.box["error"]
    return runner.box.get("result")


class AsyncioRunPatch:
    def __init__(self, original_run: Any) -> None:
        self.original_run = original_run

    def __call__(
        self,
        coro: Coroutine[Any, Any, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self.original_run(coro, *args, **kwargs)
        return run_coroutine_in_thread(self.original_run, coro, *args, **kwargs)


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


@pytest.fixture(autouse=True)
def patch_asyncio_run(monkeypatch: pytest.MonkeyPatch):
    """
    Make asyncio.run() resilient when tests execute under an already-running loop.
    Several unit tests use asyncio.run() from synchronous test bodies.
    """
    monkeypatch.setattr(asyncio, "run", AsyncioRunPatch(asyncio.run))
    yield
