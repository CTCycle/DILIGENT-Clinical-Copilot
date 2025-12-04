from __future__ import annotations

import json
import re
from typing import Any

from DILIGENT.server.utils.services.text.normalization import coerce_text

SYNONYM_SPLIT_RE = re.compile(r"[;,/\n]+")


# -----------------------------------------------------------------------------
def try_parse_json(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
def extract_synonym_strings(
    value: Any, seen_refs: set[int] | None = None
) -> list[str]:
    if seen_refs is None:
        seen_refs = set()
    if value is None:
        return []
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen_refs:
            return []
        seen_refs.add(marker)
        collected: list[str] = []
        for entry in value.values():
            collected.extend(extract_synonym_strings(entry, seen_refs))
        return collected
    if isinstance(value, (list, tuple, set)):
        marker = id(value)
        if marker in seen_refs:
            return []
        seen_refs.add(marker)
        collected: list[str] = []
        for entry in value:
            collected.extend(extract_synonym_strings(entry, seen_refs))
        return collected
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
            parsed = try_parse_json(stripped)
            if isinstance(parsed, (dict, list)):
                return extract_synonym_strings(parsed, seen_refs)
        return [value]
    text = coerce_text(value)
    if text is None:
        return []
    return extract_synonym_strings(text, seen_refs)


# -----------------------------------------------------------------------------
def parse_synonym_list(value: Any) -> list[str]:
    raw_values = extract_synonym_strings(value)
    synonyms: list[str] = []
    for raw in raw_values:
        text = coerce_text(raw)
        if text:
            synonyms.append(text)
    return synonyms


# -----------------------------------------------------------------------------
def split_synonym_variants(value: str) -> list[str]:
    if not value:
        return []
    segments = SYNONYM_SPLIT_RE.split(value)
    variants: list[str] = []
    for segment in segments:
        stripped = segment.strip()
        if stripped:
            variants.append(stripped)
    return variants


__all__ = [
    "extract_synonym_strings",
    "parse_synonym_list",
    "split_synonym_variants",
    "try_parse_json",
]
