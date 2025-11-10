from __future__ import annotations

import re
from typing import Any

###############################################################################
def coerce_str(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else default
    if value is None:
        return default
    normalized = str(value).strip()
    return normalized if normalized else default


# -----------------------------------------------------------------------------
def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


# -----------------------------------------------------------------------------
def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# -----------------------------------------------------------------------------
def extract_positive_int(value: Any) -> int | None:
    candidate: int | None = None
    if isinstance(value, bool) or value is None:
        candidate = None
    elif isinstance(value, int):
        candidate = value
    elif isinstance(value, float):
        candidate = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            candidate = int(stripped)
        else:
            match = re.search(r"\d+", stripped)
            if match:
                candidate = int(match.group(0))
    else:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            candidate = None
    if candidate is None or candidate <= 0:
        return None
    return candidate


# -----------------------------------------------------------------------------
def coerce_positive_int(value: Any, default: int = 1) -> int:
    candidate = extract_positive_int(value)
    return candidate if candidate is not None else default


# -----------------------------------------------------------------------------
def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "coerce_bool",
    "coerce_float",
    "coerce_int",
    "coerce_positive_int",
    "extract_positive_int",
    "coerce_str",
]
