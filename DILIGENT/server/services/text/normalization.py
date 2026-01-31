from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

# -----------------------------------------------------------------------------
def coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


# -----------------------------------------------------------------------------
def normalize_whitespace(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


# -----------------------------------------------------------------------------
def normalize_drug_name(value: str) -> str:
    if not value:
        return ""
    normalized = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return normalize_whitespace(normalized)


# -----------------------------------------------------------------------------
def normalize_token(token: str) -> str:
    if not token:
        return ""
    return re.sub(r"[.,;:]+$", "", token.lower())


__all__ = [
    "coerce_text",
    "normalize_drug_name",
    "normalize_token",
    "normalize_whitespace",
]
