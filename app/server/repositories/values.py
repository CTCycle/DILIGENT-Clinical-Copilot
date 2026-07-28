"""Pure value normalization helpers shared by focused repositories."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd


###############################################################################
def normalize_string(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
    elif value is None or pd.isna(value):
        return None
    else:
        normalized = str(value).strip()
    if not normalized or normalized.casefold() in {
        "not available",
        "nan",
        "none",
        "<na>",
        "nat",
    }:
        return None
    return normalized


###############################################################################
def normalize_date_value(value: Any) -> date | None:
    normalized = normalize_string(value)
    if not normalized:
        return None
    if re.fullmatch(r"[+-]?\d+", normalized):
        digits = normalized.lstrip("+-")
        if len(digits) == 8:
            parsed = pd.to_datetime(normalized, errors="coerce", format="%Y%m%d", utc=True)
        else:
            unit = {10: "s", 13: "ms", 16: "us", 19: "ns"}.get(len(digits))
            if unit is None:
                return None
            parsed = pd.to_datetime(int(normalized), errors="coerce", utc=True, unit=unit)
    else:
        parsed = pd.to_datetime(normalized, errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed.date()


###############################################################################
def normalize_date(value: Any) -> str | None:
    parsed = normalize_date_value(value)
    return parsed.isoformat() if parsed is not None else normalize_string(value)


###############################################################################
def normalize_flag(value: Any) -> int | None:
    normalized = normalize_string(value)
    if normalized is None:
        return None
    lowered = normalized.casefold()
    if lowered in {"1", "y", "yes", "true"}:
        return 1
    if lowered in {"0", "n", "no", "false", "2"}:
        return 0
    try:
        return 1 if int(normalized) != 0 else 0
    except (TypeError, ValueError):
        return None


###############################################################################
def normalize_session_status(value: Any) -> str:
    normalized = normalize_string(value)
    return "failed" if normalized and normalized.casefold() == "failed" else "successful"


###############################################################################
def join_values(values: set[str]) -> str | None:
    return "; ".join(sorted({item.strip() for item in values if item.strip()})) or None


###############################################################################
def to_int(value: Any) -> int | None:
    normalized = normalize_string(value)
    try:
        return int(normalized) if normalized is not None else None
    except (TypeError, ValueError):
        return None


###############################################################################
def to_float(value: Any) -> float | None:
    normalized = normalize_string(value)
    try:
        return float(normalized) if normalized is not None else None
    except (TypeError, ValueError):
        return None
