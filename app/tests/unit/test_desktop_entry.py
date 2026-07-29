from __future__ import annotations

import json
import sys
import types

import pytest

import desktop_entry


def test_host_is_restricted_to_localhost(monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        desktop_entry.run_desktop_backend(
            ready_file=tmp_path / "ready.json",
            host="0.0.0.0",
        )


def test_desktop_entry_writes_dynamic_ready_file_and_cleans_up(
    monkeypatch,
    tmp_path,
) -> None:
    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "data"
    monkeypatch.setenv("DILIGENT_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("DILIGENT_DATA_ROOT", str(data_root))
    monkeypatch.setenv("DILIGENT_RELEASE_VERSION", "3.1.0")
    fake_app = types.ModuleType("app")
    fake_app.app = object()
    monkeypatch.setitem(sys.modules, "app", fake_app)
    observed: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config) -> None:
            observed["config"] = config

        def run(self, *, sockets) -> None:
            observed["sockets"] = sockets
            ready_file = tmp_path / "ready.json"
            observed["payload"] = ready_file.read_text(encoding="utf-8")

    monkeypatch.setattr(desktop_entry.uvicorn, "Server", FakeServer)
    ready_file = tmp_path / "ready.json"

    desktop_entry.run_desktop_backend(ready_file=ready_file)

    payload = observed["payload"]
    assert '"port":' in str(payload)
    assert '"pid":' in str(payload)
    assert '"release_version": "3.1.0"' in str(payload)
    assert json.loads(str(payload))["port"] > 0
    assert not ready_file.exists()


def test_invalid_packaged_environment_fails_before_importing_app(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("DILIGENT_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.delenv("DILIGENT_DATA_ROOT", raising=False)
    monkeypatch.setenv("DILIGENT_RELEASE_VERSION", "3.1")

    with pytest.raises(RuntimeError):
        desktop_entry.run_desktop_backend(ready_file=tmp_path / "ready.json")
