from __future__ import annotations

from pathlib import Path

from common.runtime_layout import resolve_runtime_layout


def test_source_resources_default_and_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DILIGENT_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("DILIGENT_DATA_ROOT", raising=False)
    monkeypatch.delenv("DILIGENT_RESOURCES_PATH", raising=False)
    resolve_runtime_layout.cache_clear()
    try:
        repository_root = Path(__file__).resolve().parents[3]
        default_layout = resolve_runtime_layout()
        assert default_layout.immutable_resources_root == repository_root / "resources"
        assert default_layout.mutable_resources_root == repository_root / "resources"

        custom_root = tmp_path / "custom-resources"
        monkeypatch.setenv("DILIGENT_RESOURCES_PATH", str(custom_root))
        resolve_runtime_layout.cache_clear()
        custom_layout = resolve_runtime_layout()
        assert custom_layout.immutable_resources_root == custom_root.resolve()
        assert custom_layout.mutable_resources_root == custom_root.resolve()
    finally:
        resolve_runtime_layout.cache_clear()


def test_packaged_resources_live_at_runtime_root(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "data"
    monkeypatch.setenv("DILIGENT_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("DILIGENT_DATA_ROOT", str(data_root))
    monkeypatch.delenv("DILIGENT_RESOURCES_PATH", raising=False)
    resolve_runtime_layout.cache_clear()
    try:
        layout = resolve_runtime_layout()
        assert layout.immutable_resources_root == runtime_root / "resources"
        assert layout.mutable_resources_root == data_root / "resources"
    finally:
        resolve_runtime_layout.cache_clear()
