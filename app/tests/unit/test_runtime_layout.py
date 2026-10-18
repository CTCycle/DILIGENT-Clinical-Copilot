# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path

from common.runtime_layout import resolve_runtime_layout


def test_source_data_default_and_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DILIGENT_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("DILIGENT_DATA_ROOT", raising=False)
    monkeypatch.delenv("DILIGENT_DATA_PATH", raising=False)
    resolve_runtime_layout.cache_clear()
    try:
        repository_root = Path(__file__).resolve().parents[3]
        default_layout = resolve_runtime_layout()
        assert default_layout.immutable_data_root == repository_root / "data"
        assert default_layout.mutable_data_root == repository_root / "data"

        custom_root = tmp_path / "custom-data"
        monkeypatch.setenv("DILIGENT_DATA_PATH", str(custom_root))
        resolve_runtime_layout.cache_clear()
        custom_layout = resolve_runtime_layout()
        assert custom_layout.immutable_data_root == custom_root.resolve()
        assert custom_layout.mutable_data_root == custom_root.resolve()

        relative_root = "relative-data"
        monkeypatch.setenv("DILIGENT_DATA_PATH", relative_root)
        resolve_runtime_layout.cache_clear()
        relative_layout = resolve_runtime_layout()
        assert relative_layout.mutable_data_root == (repository_root / relative_root)
    finally:
        resolve_runtime_layout.cache_clear()


def test_packaged_immutable_data_and_persistent_data_roots(
    monkeypatch, tmp_path: Path
) -> None:
    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "data"
    monkeypatch.setenv("DILIGENT_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("DILIGENT_DATA_ROOT", str(data_root))
    monkeypatch.delenv("DILIGENT_DATA_PATH", raising=False)
    resolve_runtime_layout.cache_clear()
    try:
        layout = resolve_runtime_layout()
        assert layout.immutable_data_root == runtime_root / "data"
        assert layout.mutable_data_root == data_root
    finally:
        resolve_runtime_layout.cache_clear()
