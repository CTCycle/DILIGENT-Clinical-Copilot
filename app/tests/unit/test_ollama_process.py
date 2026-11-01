# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio

import pytest

from services.runtime import ollama_process as process_module
from services.runtime.ollama_process import OllamaProcessError, OllamaProcessManager

###############################################################################
class _FakeProcess:
    pid = 4242
    returncode: int | None = None

    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        return 0

###############################################################################
def _start(manager: OllamaProcessManager, url: str, online: list[bool]) -> None:
    async def is_online() -> bool:
        return online.pop(0) if online else True

    asyncio.run(
        manager.start(url, is_online=is_online, wait_timeout_s=2, poll_interval_s=0.01)
    )

###############################################################################
def test_start_launches_once_and_only_owned_process_can_be_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[_FakeProcess] = []

    def spawn(executable: str, base_url: str) -> _FakeProcess:
        spawned.append(_FakeProcess())
        return spawned[0]

    monkeypatch.setattr(OllamaProcessManager, "find_executable", staticmethod(lambda: "ollama"))
    monkeypatch.setattr(OllamaProcessManager, "_spawn", staticmethod(spawn))
    monkeypatch.setattr(process_module.sys, "platform", "linux")
    manager = OllamaProcessManager()

    _start(manager, "http://127.0.0.1:11434", [False, False, True])
    assert manager.owned and len(spawned) == 1

    assert manager.stop() is True
    assert spawned[0].terminated and not manager.owned
    assert manager.stop() is False

###############################################################################
def test_start_does_not_spawn_when_service_is_already_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OllamaProcessManager, "find_executable", staticmethod(lambda: "ollama"))
    monkeypatch.setattr(
        OllamaProcessManager,
        "_spawn",
        staticmethod(lambda *_: pytest.fail("external service must not be duplicated")),
    )
    manager = OllamaProcessManager()

    _start(manager, "http://127.0.0.1:11434", [True])

    assert not manager.owned
    assert manager.stop() is False

###############################################################################
@pytest.mark.parametrize(
    ("url", "executable", "message"),
    [
        ("http://ollama.example:11434", "ollama", "remote host"),
        ("http://127.0.0.1:11434", None, "not found"),
    ],
)
def test_start_refuses_remote_hosts_and_missing_executable(
    monkeypatch: pytest.MonkeyPatch, url: str, executable: str | None, message: str
) -> None:
    monkeypatch.setattr(
        OllamaProcessManager, "find_executable", staticmethod(lambda: executable)
    )
    manager = OllamaProcessManager()

    with pytest.raises(OllamaProcessError, match=message):
        _start(manager, url, [False])

    assert manager.last_error is not None and not manager.starting
