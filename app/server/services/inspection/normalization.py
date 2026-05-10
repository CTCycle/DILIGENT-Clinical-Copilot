from __future__ import annotations

import re
from typing import Any


def normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def first_iso_date(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    matched = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    return matched.group(0) if matched else None


def extract_lab_marker(text: str) -> str | None:
    matched = re.search(
        r"\b(ALT|AST|ALP|TBIL|DBIL|GGT)\b[^.;,\n]*", text, re.IGNORECASE
    )
    if not matched:
        return None
    return matched.group(0).strip()[:120]
