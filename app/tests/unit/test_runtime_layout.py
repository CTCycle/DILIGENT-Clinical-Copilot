# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path
from common.runtime_layout import resolve_runtime_layout
import os
from common.paths import EMBEDDING_MODELS_PATH
from configurations.runtime_bootstrap import configure_cache_environment

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

REPO_ROOT = Path(__file__).resolve().parents[3]

CANONICAL_CACHE_ROOT = REPO_ROOT / "runtimes" / "cache"

def test_source_and_packaged_layouts_have_one_cache_root(tmp_path: Path, monkeypatch) -> None:
    source_layout = resolve_runtime_layout()
    assert source_layout.packaged is False
    assert source_layout.cache_root == CANONICAL_CACHE_ROOT
    assert EMBEDDING_MODELS_PATH == CANONICAL_CACHE_ROOT / "embeddings"

    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "data"
    monkeypatch.setenv("DILIGENT_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("DILIGENT_DATA_ROOT", str(data_root))
    cache_environment_names = (
        "CARGO_TARGET_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "MYPY_CACHE_DIR",
        "NPM_CONFIG_CACHE",
        "PIP_CACHE_DIR",
        "PLAYWRIGHT_BROWSERS_PATH",
        "PYTHONPYCACHEPREFIX",
        "PYTEST_ADDOPTS",
        "RUFF_CACHE_DIR",
        "UV_CACHE_DIR",
    )
    original_cache_environment = {
        name: os.environ.get(name) for name in cache_environment_names
    }
    resolve_runtime_layout.cache_clear()
    try:
        packaged_layout = resolve_runtime_layout()
        assert packaged_layout.packaged is True
        assert packaged_layout.runtime_root == runtime_root.resolve()
        assert packaged_layout.cache_root == (data_root / "cache").resolve()
        configure_cache_environment()
        assert Path(os.environ["HF_HOME"]) == (
            data_root / "cache" / "huggingface"
        ).resolve()
        assert Path(os.environ["HF_HUB_CACHE"]) == (
            data_root / "cache" / "huggingface" / "hub"
        ).resolve()
        assert Path(os.environ["PYTHONPYCACHEPREFIX"]) == (
            data_root / "cache" / "python"
        ).resolve()
        assert str(data_root / "cache" / "pytest") in os.environ["PYTEST_ADDOPTS"]
    finally:
        for name, value in original_cache_environment.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        resolve_runtime_layout.cache_clear()
