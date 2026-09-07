from __future__ import annotations

import os
import shutil
from pathlib import Path

from common.runtime_layout import resolve_runtime_layout

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
    for relative_path in (
        "logs",
        "models/embeddings",
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
    "ensure_runtime_data_layout",
]
