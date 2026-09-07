from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from common.runtime_layout import resolve_runtime_layout

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

###############################################################################
def _validate_version(value: str, source: str) -> str:
    version = value.strip()
    if not _VERSION_PATTERN.fullmatch(version):
        raise RuntimeError(f"Invalid application version from {source}: {value!r}")
    return version

###############################################################################
@lru_cache(maxsize=1)
def resolve_application_version() -> str:
    configured = os.getenv("DILIGENT_RELEASE_VERSION", "").strip()
    if configured:
        return _validate_version(configured, "DILIGENT_RELEASE_VERSION")

    layout = resolve_runtime_layout()
    package_path = layout.application_root / "client" / "package.json"
    try:
        payload = json.loads(package_path.read_text(encoding="utf-8"))
        version = payload.get("version")
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to resolve application version from {package_path}"
        ) from exc
    if not isinstance(version, str):
        raise RuntimeError(f"Application version is missing from {package_path}")
    return _validate_version(version, str(package_path))


__all__ = ["resolve_application_version"]
