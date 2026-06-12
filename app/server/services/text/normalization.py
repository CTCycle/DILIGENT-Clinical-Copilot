from __future__ import annotations

import re
import unicodedata


from common.utils.text_utils import (
    coerce_text,
    normalize_drug_name,
    normalize_whitespace,
    normalize_token,
)
from services.text.vocabulary import get_text_normalization_snapshot

_SCHEDULE_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?){1,3}$")
_NUMERIC_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_DOSAGE_UNIT_TOKEN_RE = re.compile(
    r"^\d+(?:[.,]\d+)?(?:mg|mcg|ug|g|kg|ml|l|ui|iu|u|%|mmol|meq)$"
)
_STRENGTH_FRAGMENT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|g|kg|ml|l|ui|iu|u|mmol|meq|%)\b",
    re.IGNORECASE,
)
_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")


###############################################################################
def canonicalize_drug_query(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = normalized.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    normalized = _PARENTHETICAL_RE.sub(" ", normalized)
    normalized = re.sub(r"\[[^\]]*\]", " ", normalized)
    normalized = re.sub(r"(?<=\w)\s*(?:\+|/|&)\s*(?=\w)", " ", normalized)
    normalized = re.sub(r"[\(\)\{\},;:]+", " ", normalized)
    normalized = _STRENGTH_FRAGMENT_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

    raw_tokens = re.findall(r"[^\s]+", normalized)
    kept_tokens: list[str] = []
    vocabulary = get_text_normalization_snapshot()
    formulation_stopwords = vocabulary.formulation_stopwords
    for raw_token in raw_tokens:
        token = raw_token.strip(" ._-/+")
        if not token:
            continue
        if _SCHEDULE_TOKEN_RE.fullmatch(token):
            continue
        ascii_token = normalize_drug_name(token)
        if not ascii_token:
            continue
        ascii_parts = [part for part in ascii_token.split() if part]
        if ascii_parts and all(
            part in formulation_stopwords
            or _NUMERIC_TOKEN_RE.fullmatch(part)
            or _DOSAGE_UNIT_TOKEN_RE.fullmatch(part)
            for part in ascii_parts
        ):
            continue
        if ascii_token in formulation_stopwords:
            continue
        if _NUMERIC_TOKEN_RE.fullmatch(token):
            continue
        if _DOSAGE_UNIT_TOKEN_RE.fullmatch(ascii_token):
            continue
        if token.endswith("%") and ascii_token[:-1].replace(".", "", 1).isdigit():
            continue
        if any(char.isalpha() for char in token):
            kept_tokens.append(token)

    kept_tokens = strip_manufacturer_suffix_tokens(kept_tokens)
    kept_tokens = strip_trailing_temporal_tokens(kept_tokens)
    if not kept_tokens:
        fallback = normalize_drug_name(normalized)
        return resolve_known_query_alias(fallback)
    canonical = normalize_whitespace(" ".join(kept_tokens))
    return resolve_known_query_alias(canonical)


###############################################################################
def normalize_drug_query_name(value: str | None) -> str:
    canonical = canonicalize_drug_query(value)
    if not canonical:
        return ""
    return normalize_drug_name(canonical)


###############################################################################
def strip_manufacturer_suffix_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    trimmed = list(tokens)
    while trimmed:
        normalized = normalize_drug_name(trimmed[-1])
        vocabulary = get_text_normalization_snapshot()
        if not normalized:
            trimmed.pop()
            continue
        if normalized in vocabulary.manufacturer_tokens or normalized.endswith(
            vocabulary.manufacturer_suffixes
        ):
            trimmed.pop()
            continue
        break
    return trimmed


###############################################################################
def strip_trailing_temporal_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    trimmed = list(tokens)
    while trimmed:
        normalized = normalize_drug_name(trimmed[-1])
        if normalized in get_text_normalization_snapshot().trailing_temporal_tokens:
            trimmed.pop()
            continue
        break
    return trimmed


###############################################################################
def resolve_known_query_alias(value: str) -> str:
    normalized = normalize_drug_name(value)
    if not normalized:
        return ""
    snapshot = get_text_normalization_snapshot()
    if normalized in snapshot.noisy_phrases:
        return ""
    alias = snapshot.query_aliases.get(normalized)
    if alias is not None:
        return alias
    for candidate, replacement in snapshot.query_aliases.items():
        if candidate and " " in candidate and candidate in normalized:
            return replacement
    return normalized


__all__ = [
    "canonicalize_drug_query",
    "coerce_text",
    "normalize_drug_name",
    "normalize_drug_query_name",
    "normalize_token",
    "normalize_whitespace",
]
