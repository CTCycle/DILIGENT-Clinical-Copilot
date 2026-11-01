# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlparse

from common.utils.logger import logger

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
STOP_GRACE_S = 5.0

###############################################################################
class OllamaProcessError(RuntimeError):
    """Raised when the local Ollama service cannot be started or stopped."""

###############################################################################
class OllamaStartTimeout(OllamaProcessError):
    """Raised when a launched Ollama service does not become reachable in time."""

###############################################################################
class OllamaProcessManager:
    """Starts and stops only the Ollama process launched by this application."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._starting = False
        self._last_error: str | None = None

    # -------------------------------------------------------------------------
    @property
    def owned(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # -------------------------------------------------------------------------
    @property
    def starting(self) -> bool:
        return self._starting

    # -------------------------------------------------------------------------
    @property
    def last_error(self) -> str | None:
        return self._last_error

    # -------------------------------------------------------------------------
    @staticmethod
    def find_executable() -> str | None:
        found = shutil.which("ollama")
        if found:
            return found
        candidates: list[Path] = []
        for variable, relative in (
            ("LOCALAPPDATA", Path("Programs") / "Ollama" / "ollama.exe"),
            ("ProgramFiles", Path("Ollama") / "ollama.exe"),
        ):
            base = os.environ.get(variable)
            if base:
                candidates.append(Path(base) / relative)
        return next((str(path) for path in candidates if path.is_file()), None)

    # -------------------------------------------------------------------------
    @staticmethod
    def is_local_url(base_url: str) -> bool:
        return (urlparse(base_url).hostname or "") in LOOPBACK_HOSTS

    # -------------------------------------------------------------------------
    async def start(
        self,
        base_url: str,
        *,
        is_online: Callable[[], Awaitable[bool]],
        wait_timeout_s: float,
        poll_interval_s: float,
    ) -> None:
        """Launch ``ollama serve`` and wait until it is reachable."""
        if self._starting:
            raise OllamaProcessError("Ollama is already starting.")
        if not self.is_local_url(base_url):
            self._last_error = "Ollama is configured on a remote host and cannot be started here."
            raise OllamaProcessError(self._last_error)
        executable = self.find_executable()
        if executable is None:
            self._last_error = "Ollama executable was not found. Install Ollama or add it to PATH."
            raise OllamaProcessError(self._last_error)
        self._starting = True
        self._last_error = None
        try:
            if await is_online():
                return
            process = self._spawn(executable, base_url)
            self._process = process
            loop = asyncio.get_running_loop()
            deadline = loop.time() + wait_timeout_s
            while loop.time() < deadline:
                if await is_online():
                    return
                if process.poll() is not None:
                    self._process = None
                    raise OllamaProcessError(
                        f"Ollama exited unexpectedly with code {process.returncode}."
                    )
                await asyncio.sleep(poll_interval_s)
            await asyncio.to_thread(self.stop)
            raise OllamaStartTimeout("Timed out waiting for Ollama to start.")
        except OllamaProcessError as exc:
            self._last_error = str(exc)
            raise
        finally:
            self._starting = False

    # -------------------------------------------------------------------------
    @staticmethod
    def _spawn(executable: str, base_url: str) -> subprocess.Popen[bytes]:
        parsed = urlparse(base_url)
        environment = dict(os.environ)
        if parsed.hostname and parsed.port:
            environment["OLLAMA_HOST"] = f"{parsed.hostname}:{parsed.port}"
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        try:
            return subprocess.Popen(
                [executable, "serve"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                creationflags=flags,
            )
        except OSError as exc:
            raise OllamaProcessError(f"Failed to launch Ollama: {exc}") from exc

    # -------------------------------------------------------------------------
    def stop(self) -> bool:
        """Terminate the process this manager launched; never touches others."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return False
        if sys.platform == "win32":
            # Windows terminate() is a hard kill of the parent only; the model
            # runners are child processes, so end the whole tree.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            logger.warning("Ollama did not stop gracefully; killing it.")
            process.kill()
            process.wait(timeout=STOP_GRACE_S)
        return True

###############################################################################
ollama_process_manager = OllamaProcessManager()
