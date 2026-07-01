"""
Pytest configuration for DILIGENT E2E tests.
Provides fixtures for Playwright page objects and API client.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest

from common import paths as common_paths
from repositories.database import sqlite as sqlite_module

###############################################################################
def _configure_test_embedded_database_path() -> None:
    temp_root = Path(tempfile.gettempdir()) / "diligent-pytest-dbs"
    temp_root.mkdir(parents=True, exist_ok=True)
    db_path = temp_root / f"embedded-{uuid.uuid4().hex}.db"
    common_paths.DATABASE_FILE_PATH = db_path
    sqlite_module.DATABASE_FILE_PATH = db_path

###############################################################################
def _configure_playwright_node_runtime() -> None:
    """
    Ensure pytest-playwright uses the bundled Node runtime instead of ambient PATH.
    This avoids host-specific Node resolution issues during driver startup.
    """
    if os.getenv("PLAYWRIGHT_NODEJS_PATH"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    bundled_node = repo_root / "runtimes" / "nodejs" / "node.exe"
    if bundled_node.is_file():
        os.environ["PLAYWRIGHT_NODEJS_PATH"] = str(bundled_node)


_configure_playwright_node_runtime()
_configure_test_embedded_database_path()

###############################################################################
class WorkspaceTempPathFactory:

    # -------------------------------------------------------------------------
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        safe_basename = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in basename
        ).strip("_")
        if not safe_basename:
            safe_basename = "tmp"
        suffix = f"-{uuid.uuid4().hex}" if numbered else ""
        path = self.root / f"{safe_basename}{suffix}"
        path.mkdir(parents=True, exist_ok=False)
        return path

###############################################################################
@pytest.fixture(scope="session")
def tmp_path_factory() -> WorkspaceTempPathFactory:
    root = Path(tempfile.gettempdir()) / "diligent-pytest-fixtures" / uuid.uuid4().hex
    factory = WorkspaceTempPathFactory(root)
    yield factory
    shutil.rmtree(root, ignore_errors=True)

###############################################################################
@pytest.fixture
def tmp_path(
    request: pytest.FixtureRequest,
    tmp_path_factory: WorkspaceTempPathFactory,
) -> Path:
    path = tmp_path_factory.mktemp(request.node.name)
    yield path
    shutil.rmtree(path, ignore_errors=True)

###############################################################################
class CoroutineThreadRunner:

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def __call__(self) -> None:
        try:
            self.box["result"] = self.run_callable(
                self.coro,
                *self.args,
                **self.kwargs,
            )
        except BaseException as exc:
            self.box["error"] = exc

###############################################################################
def _normalize_host_for_url(host: str) -> str:
    if host in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return host

###############################################################################
def _build_base_url(
    host_env: str,
    port_env: str,
    default_host: str,
    default_port: str,
) -> str:
    host = _normalize_host_for_url(os.getenv(host_env, default_host))
    port = os.getenv(port_env, default_port)
    return f"http://{host}:{port}"

###############################################################################
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

###############################################################################
class AsyncioRunPatch:

    # -------------------------------------------------------------------------
    def __init__(self, original_run: Any) -> None:
        self.original_run = original_run

    # -------------------------------------------------------------------------
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

###############################################################################
@pytest.fixture(scope="session")
def base_url() -> str:
    """Returns the base URL of the UI."""
    return UI_BASE_URL

###############################################################################
@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Returns the base URL of the API."""
    return API_BASE_URL

###############################################################################
@pytest.fixture
def api_context(playwright):
    """
    Creates an API request context for making direct HTTP calls.
    Useful for testing backend endpoints independently of the UI.
    """
    context = playwright.request.new_context(base_url=API_BASE_URL)
    yield context
    context.dispose()

###############################################################################
@pytest.fixture(autouse=True)
def patch_asyncio_run(monkeypatch: pytest.MonkeyPatch):
    """
    Make asyncio.run() resilient when tests execute under an already-running loop.
    Several unit tests use asyncio.run() from synchronous test bodies.
    """
    monkeypatch.setattr(asyncio, "run", AsyncioRunPatch(asyncio.run))
    yield
