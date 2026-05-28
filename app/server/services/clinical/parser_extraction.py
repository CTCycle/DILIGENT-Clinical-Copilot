from __future__ import annotations

import re

from common.utils.patterns import (
    DRUG_BRACKET_TRAIL_RE,
    DRUG_BULLET_RE,
    DRUG_SCHEDULE_RE,
    DRUG_START_DATE_RE,
    DRUG_SUSPENSION_DATE_RE,
    DRUG_SUSPENSION_RE,
)
from services.catalogs.runtime import get_reference_catalog_snapshot

SCHEDULE_RE = DRUG_SCHEDULE_RE
DATE_LIKE_SCHEDULE_RE = re.compile(r"^\d{4}\s*-\s*\d{1,2}\s*-\s*\d{1,2}$")
BULLET_RE = DRUG_BULLET_RE
BRACKET_TRAIL_RE = DRUG_BRACKET_TRAIL_RE
SUSPENSION_RE = DRUG_SUSPENSION_RE
SUSPENSION_DATE_RE = DRUG_SUSPENSION_DATE_RE
START_DATE_RE = DRUG_START_DATE_RE


def _phrase_pattern(values: list[str], *, anchor_word: bool = True) -> re.Pattern[str]:
    escaped = [re.escape(value.strip()) for value in values if value and value.strip()]
    if not escaped:
        return re.compile(r"$^")
    body = "|".join(sorted(set(escaped), key=len, reverse=True))
    if anchor_word:
        return re.compile(r"\b(?:" + body + r")\b", re.IGNORECASE)
    return re.compile(r"(?:" + body + r")", re.IGNORECASE)


def _build_timing_terms() -> list[str]:
    snapshot = get_reference_catalog_snapshot()
    values: list[str] = []
    values.extend(snapshot.values("clinical_extraction", "drug_timing_terms"))
    values.extend(snapshot.values("clinical_extraction", "drug_metadata_labels"))
    values.extend(snapshot.values("clinical_extraction", "drug_continuation_markers"))
    return [value for value in values if value]


def build_route_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    snapshot = get_reference_catalog_snapshot()
    entries = snapshot.entries("clinical_extraction", "drug_route_terms")
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        grouped.setdefault(entry.key, []).append(entry.value)
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for key, values in grouped.items():
        regex = _phrase_pattern(values)
        if regex.pattern != r"$^":
            patterns.append((key, regex))
    return tuple(patterns)


def build_dose_cue_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    units = list(snapshot.values("clinical_extraction", "drug_dosage_units"))
    escaped_units = [re.escape(unit.strip()) for unit in units if unit and unit.strip()]
    if not escaped_units:
        return re.compile(r"$^")
    return re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:" + "|".join(sorted(set(escaped_units))) + r")\b",
        re.IGNORECASE,
    )


def build_dosage_temporal_split_re() -> re.Pattern[str]:
    terms = _build_timing_terms()
    cue = _phrase_pattern(terms).pattern.replace(r"\b(?:", "(?:").replace(r")\b", ")")
    if cue == r"$^":
        return re.compile(r"$^")
    return re.compile(
        r"(?:[,;]\s*|\s+)(?:" + cue + r")\b.*$",
        re.IGNORECASE,
    )


def build_name_temporal_split_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    terms = _build_timing_terms()
    terms.extend(snapshot.values("clinical_extraction", "drug_timing_context_terms"))
    cue = _phrase_pattern(terms).pattern.replace(r"\b(?:", "(?:").replace(r")\b", ")")
    if cue == r"$^":
        return re.compile(r"$^")
    return re.compile(
        r"(?:[,;]\s*|\s+)(?:" + cue + r")\b.*$",
        re.IGNORECASE,
    )


def build_trailing_route_token_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    terms = list(snapshot.values("clinical_extraction", "drug_route_terms"))
    escaped = [re.escape(term.strip()) for term in terms if term and term.strip()]
    if not escaped:
        return re.compile(r"$^")
    return re.compile(
        r"\b(?:" + "|".join(sorted(set(escaped))) + r")\s*$", re.IGNORECASE
    )


def build_start_event_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    terms = list(snapshot.values("clinical_extraction", "drug_start_terms"))
    cue = _phrase_pattern(terms).pattern.replace(r"\b(?:", "(?:").replace(r")\b", ")")
    if cue == r"$^":
        return re.compile(r"$^")
    return re.compile(
        r"\b(?:" + cue + r")\b(?P<tail>[^,;\n]*)",
        re.IGNORECASE,
    )


def build_suspension_event_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    terms = list(snapshot.values("clinical_extraction", "drug_suspension_terms"))
    cue = _phrase_pattern(terms).pattern.replace(r"\b(?:", "(?:").replace(r")\b", ")")
    if cue == r"$^":
        return re.compile(r"$^")
    return re.compile(
        r"\b(?:" + cue + r")\b(?P<tail>[^,;\n]*)",
        re.IGNORECASE,
    )
