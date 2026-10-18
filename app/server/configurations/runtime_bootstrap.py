# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import filecmp
import os
import shutil
from pathlib import Path

from common.runtime_layout import RuntimeLayout, resolve_runtime_layout


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
def create_mutable_data_directories() -> None:
    layout = resolve_runtime_layout()
    mutable_root = layout.mutable_data_root
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
def _find_data_migration_conflicts(source: Path, destination: Path) -> list[Path]:
    if source.is_symlink() or destination.is_symlink():
        return [source]
    if not destination.exists():
        return []
    if source.is_file() and destination.is_file():
        return [] if filecmp.cmp(source, destination, shallow=False) else [source]
    if source.is_dir() and destination.is_dir():
        conflicts: list[Path] = []
        for child in source.iterdir():
            conflicts.extend(
                _find_data_migration_conflicts(child, destination / child.name)
            )
        return conflicts
    return [source]


###############################################################################
def _merge_data_path(source: Path, destination: Path) -> None:
    if source.is_symlink() or destination.is_symlink():
        raise RuntimeError("Data migration does not follow symbolic links")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        return
    if source.is_file() and destination.is_file():
        if not filecmp.cmp(source, destination, shallow=False):
            raise RuntimeError("Data migration encountered a conflicting file")
        source.unlink()
        return
    if source.is_dir() and destination.is_dir():
        for child in source.iterdir():
            _merge_data_path(child, destination / child.name)
        source.rmdir()
        return
    raise RuntimeError("Data migration encountered a conflicting path")


###############################################################################
def migrate_legacy_data_layout(layout: RuntimeLayout | None = None) -> None:
    """Move persisted files from the previous resources layout without overwriting data."""
    layout = layout or resolve_runtime_layout()
    if layout.packaged:
        legacy_root = layout.mutable_data_root / "resources"
    else:
        default_data_root = layout.runtime_root / "data"
        if layout.mutable_data_root.resolve() != default_data_root.resolve():
            return
        legacy_root = layout.runtime_root / "resources"

    if legacy_root.is_symlink():
        raise RuntimeError("Data migration does not follow symbolic links")
    if (
        not legacy_root.is_dir()
        or legacy_root.resolve() == layout.mutable_data_root.resolve()
    ):
        return

    entries = (
        "database.db",
        "database.db-journal",
        "database.db-shm",
        "database.db-wal",
        "access-key-material.json",
        "tools",
        "sources",
        "exports",
        "state",
    )
    migrations = [
        (legacy_root / name, layout.mutable_data_root / name)
        for name in entries
        if (legacy_root / name).exists() or (legacy_root / name).is_symlink()
    ]
    conflicts = [
        source
        for source, destination in migrations
        if _find_data_migration_conflicts(source, destination)
    ]
    if conflicts:
        raise RuntimeError(
            "Previous and current data locations contain conflicting files; "
            "no data was changed. Review both locations and resolve the overlap "
            "before restarting."
        )

    for source, destination in migrations:
        _merge_data_path(source, destination)
    try:
        legacy_root.rmdir()
    except OSError:
        pass


###############################################################################
def ensure_runtime_data_layout() -> None:
    layout = resolve_runtime_layout()
    configure_cache_environment()
    migrate_legacy_data_layout(layout)
    if not layout.packaged:
        return

    layout.settings_root.mkdir(parents=True, exist_ok=True)
    copy_initial_file_if_missing(
        layout.settings_template_root / ".env.example",
        layout.settings_root / ".env",
    )
    create_mutable_data_directories()


__all__ = [
    "copy_initial_file_if_missing",
    "create_mutable_data_directories",
    "migrate_legacy_data_layout",
    "configure_cache_environment",
    "ensure_runtime_data_layout",
]
