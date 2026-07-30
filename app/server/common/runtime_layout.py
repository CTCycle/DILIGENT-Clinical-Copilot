from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


###############################################################################
@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    """Absolute filesystem roots for source and packaged executions."""

    packaged: bool
    runtime_root: Path
    application_root: Path
    settings_template_root: Path
    settings_root: Path
    immutable_resources_root: Path
    mutable_resources_root: Path
    client_dist_root: Path


###############################################################################
def _resolve_required_absolute_environment_path(name: str) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set for packaged execution")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path")
    return path.resolve()


###############################################################################
def _resolve_source_layout() -> RuntimeLayout:
    repository_root = Path(__file__).resolve().parents[3]
    application_root = repository_root / "app"
    resources_root = application_root / "resources"
    return RuntimeLayout(
        packaged=False,
        runtime_root=repository_root,
        application_root=application_root,
        settings_template_root=repository_root / "settings",
        settings_root=repository_root / "settings",
        immutable_resources_root=resources_root,
        mutable_resources_root=resources_root,
        client_dist_root=application_root / "client" / "dist" / "browser",
    )


###############################################################################
def _resolve_packaged_layout() -> RuntimeLayout:
    runtime_root = _resolve_required_absolute_environment_path(
        "DILIGENT_RUNTIME_ROOT"
    )
    data_root = _resolve_required_absolute_environment_path("DILIGENT_DATA_ROOT")
    application_root = runtime_root / "app"
    return RuntimeLayout(
        packaged=True,
        runtime_root=runtime_root,
        application_root=application_root,
        settings_template_root=runtime_root / "settings",
        settings_root=data_root / "settings",
        immutable_resources_root=application_root / "resources",
        mutable_resources_root=data_root / "resources",
        client_dist_root=application_root / "client" / "dist" / "browser",
    )


###############################################################################
@lru_cache(maxsize=1)
def resolve_runtime_layout() -> RuntimeLayout:
    runtime_root_set = bool(os.getenv("DILIGENT_RUNTIME_ROOT", "").strip())
    data_root_set = bool(os.getenv("DILIGENT_DATA_ROOT", "").strip())
    if runtime_root_set != data_root_set:
        raise RuntimeError(
            "DILIGENT_RUNTIME_ROOT and DILIGENT_DATA_ROOT must be supplied together"
        )
    return _resolve_packaged_layout() if runtime_root_set else _resolve_source_layout()


__all__ = [
    "RuntimeLayout",
    "resolve_runtime_layout",
    "_resolve_required_absolute_environment_path",
    "_resolve_source_layout",
    "_resolve_packaged_layout",
]
