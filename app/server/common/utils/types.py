from __future__ import annotations

import re
from typing import Any

###############################################################################
def _extract_int_from_str(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.isdigit():
        return int(stripped)
    match = re.search(r"\d+", stripped)
    return int(match.group(0)) if match else None

###############################################################################
def extract_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, float):
        candidate = int(value)
    elif isinstance(value, str):
        candidate = _extract_int_from_str(value)
    else:
        try:
            candidate = int(value)
        except TypeError, ValueError:
            return None
    if candidate is None or candidate <= 0:
        return None
    return candidate

###############################################################################
def coerce_positive_int(value: Any, default: int = 1) -> int:
    candidate = extract_positive_int(value)
    return candidate if candidate is not None else default

###############################################################################
def coerce_bool(value: Any, default: bool) -> bool:
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

###############################################################################
def coerce_bool_or_unknown(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"

###############################################################################
def coerce_int(
    value: Any, default: int, minimum: int | None = None, maximum: int | None = None
) -> int:
    candidate: int
    if isinstance(value, bool):
        candidate = int(value)
    else:
        try:
            candidate = int(value)
        except TypeError, ValueError:
            candidate = default
    if minimum is not None and candidate < minimum:
        candidate = minimum
    if maximum is not None and candidate > maximum:
        candidate = maximum
    return candidate

###############################################################################
def coerce_float(
    value: Any,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        candidate = float(value)
    except TypeError, ValueError:
        candidate = default
    if minimum is not None and candidate < minimum:
        candidate = minimum
    if maximum is not None and candidate > maximum:
        candidate = maximum
    return candidate

###############################################################################
def coerce_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or default
    if value is None:
        return default
    return str(value).strip() or default

###############################################################################
def coerce_str_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


__all__ = [
    "coerce_bool",
    "coerce_bool_or_unknown",
    "coerce_float",
    "coerce_int",
    "coerce_positive_int",
    "extract_positive_int",
    "coerce_str",
    "coerce_str_or_none",
]
