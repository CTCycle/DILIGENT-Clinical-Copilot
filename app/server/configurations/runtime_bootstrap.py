from __future__ import annotations

import os
import shutil
from pathlib import Path

from common.runtime_layout import resolve_runtime_layout

###############################################################################
def configure_cache_environment() -> None:
    """Keep packaged dependency and model caches inside the runtime cache root."""
    cache_root = resolve_runtime_layout().cache_root.resolve()
    pytest_root = cache_root / "pytest"
    environment_paths = {
        "CARGO_TARGET_DIR": cache_root / "cargo" / "target",
        "COVERAGE_FILE": cache_root / "coverage" / ".coverage",
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "HF_ASSETS_CACHE": cache_root / "huggingface" / "assets",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "MYPY_CACHE_DIR": cache_root / "mypy",
        "NPM_CONFIG_CACHE": cache_root / "npm",
        "PIP_CACHE_DIR": cache_root / "pip",
        "PLAYWRIGHT_BROWSERS_PATH": cache_root / "playwright",
        "PYTHONPYCACHEPREFIX": cache_root / "python",
        "RUFF_CACHE_DIR": cache_root / "ruff",
        "UV_CACHE_DIR": cache_root / "uv",
    }
    for name, path in environment_paths.items():
        os.environ[name] = str(path)
    os.environ["PYTEST_ADDOPTS"] = (
        f'--basetemp="{pytest_root / "basetemp"}" -o cache_dir="{pytest_root}"'
    )

###############################################################################
def copy_initial_file_if_missing(source: Path, destination: Path) -> bool:
    """Atomically seed one persistent file without overwriting user data."""
    if destination.exists():
        return False
    if not source.is_file():
        raise FileNotFoundError(f"Runtime seed file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True

###############################################################################
def create_mutable_resource_directories() -> None:
    layout = resolve_runtime_layout()
    mutable_root = layout.mutable_resources_root
    (layout.cache_root / "embeddings").mkdir(parents=True, exist_ok=True)
    for relative_path in (
        "sources/archives",
        "sources/documents",
        "sources/vectors",
        "exports",
        "state",
    ):
        (mutable_root / relative_path).mkdir(parents=True, exist_ok=True)

###############################################################################
def ensure_runtime_data_layout() -> None:
    layout = resolve_runtime_layout()
    configure_cache_environment()
    if not layout.packaged:
        return

    layout.settings_root.mkdir(parents=True, exist_ok=True)
    copy_initial_file_if_missing(
        layout.settings_template_root / ".env.example",
        layout.settings_root / ".env",
    )
    copy_initial_file_if_missing(
        layout.settings_template_root / "configurations.json",
        layout.settings_root / "configurations.json",
    )
    create_mutable_resource_directories()


__all__ = [
    "copy_initial_file_if_missing",
    "create_mutable_resource_directories",
    "configure_cache_environment",
    "ensure_runtime_data_layout",
]
