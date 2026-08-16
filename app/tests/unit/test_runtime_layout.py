from __future__ import annotations

from pathlib import Path

import pytest

from common import runtime_layout

###############################################################################
def _clear_layout_cache() -> None:
    runtime_layout.resolve_runtime_layout.cache_clear()

###############################################################################
def test_source_layout_preserves_repository_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DILIGENT_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("DILIGENT_DATA_ROOT", raising=False)
    _clear_layout_cache()

    layout = runtime_layout.resolve_runtime_layout()

    assert layout.packaged is False
    assert layout.runtime_root == Path(runtime_layout.__file__).resolve().parents[3]
    assert layout.application_root.name == "app"
    assert layout.settings_root == layout.settings_template_root
    assert layout.mutable_resources_root == layout.immutable_resources_root
    assert layout.client_dist_root.parts[-3:] == ("client", "dist", "browser")

###############################################################################
def test_packaged_layout_separates_immutable_and_mutable_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "data"
    monkeypatch.setenv("DILIGENT_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("DILIGENT_DATA_ROOT", str(data_root))
    _clear_layout_cache()

    layout = runtime_layout.resolve_runtime_layout()

    assert layout.packaged is True
    assert layout.runtime_root == runtime_root.resolve()
    assert layout.settings_template_root == runtime_root.resolve() / "settings"
    assert layout.settings_root == data_root.resolve() / "settings"
    assert layout.immutable_resources_root == runtime_root.resolve() / "app" / "resources"
    assert layout.mutable_resources_root == data_root.resolve() / "resources"
    assert layout.client_dist_root == runtime_root.resolve() / "app" / "client" / "dist" / "browser"

###############################################################################
def test_only_one_packaged_root_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DILIGENT_RUNTIME_ROOT", "C:/runtime")
    monkeypatch.delenv("DILIGENT_DATA_ROOT", raising=False)
    _clear_layout_cache()

    with pytest.raises(RuntimeError, match="must be supplied together"):
        runtime_layout.resolve_runtime_layout()

###############################################################################
@pytest.mark.parametrize("name", ["DILIGENT_RUNTIME_ROOT", "DILIGENT_DATA_ROOT"])
def test_relative_packaged_root_fails(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    other = "DILIGENT_DATA_ROOT" if name == "DILIGENT_RUNTIME_ROOT" else "DILIGENT_RUNTIME_ROOT"
    monkeypatch.setenv(name, "relative-root")
    monkeypatch.setenv(other, "C:/absolute-root")
    _clear_layout_cache()

    with pytest.raises(RuntimeError, match="absolute path"):
        runtime_layout.resolve_runtime_layout()
