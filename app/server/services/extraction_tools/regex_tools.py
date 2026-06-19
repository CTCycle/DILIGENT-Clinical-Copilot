from __future__ import annotations

import re
from collections.abc import Iterable

from services.extraction_tools.schemas import (
    RegexToolMatch,
    RegexToolRequest,
    RegexToolResult,
)

RegexSpec = tuple[str, re.Pattern[str], float]

TOOL_PATTERNS: dict[str, tuple[str, tuple[RegexSpec, ...]]] = {
    "search_drug_by_regex": (
        "clinical_drug_v1",
        (
            (
                "drug_capitalized_with_dose",
                re.compile(
                    r"\b([A-ZÀ-Ö][A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\- ]{1,60}?)(?=\s+\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ug|ml|ui|iu)\b)",
                ),
                0.78,
            ),
            (
                "drug_after_assume",
                re.compile(
                    r"\b(?:assume|takes|terapia con|therapy with)\s+([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\- ]{1,60})",
                    re.IGNORECASE,
                ),
                0.7,
            ),
        ),
    ),
    "search_dosage_by_regex": (
        "clinical_dosage_v1",
        (
            (
                "dose_unit",
                re.compile(
                    r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ug|ml|units?|ui|iu)(?:/day|/die)?\b",
                    re.IGNORECASE,
                ),
                0.92,
            ),
        ),
    ),
    "search_timeline_by_regex": (
        "clinical_timeline_v1",
        (
            (
                "timeline_phrase",
                re.compile(
                    r"\b(?:from|since|started|stopped|suspended|dal|da|iniziat[ao]|sospes[ao])\s+\S+(?:\s+\S+){0,4}",
                    re.IGNORECASE,
                ),
                0.72,
            ),
        ),
    ),
    "search_lab_value_by_regex": (
        "liver_labs_v1",
        (
            (
                "liver_lab_value",
                re.compile(
                    r"\b(?:ALT|ALAT|AST|ASAT|ALP|GGT|INR|bilirubina(?:\s+totale|\s+diretta)?|bilirubin(?:\s+total|\s+direct)?)\b[\s:=<>-]*(\d+(?:[.,]\d+)?)(?:\s*(U/L|UI/L|mg/dL|umol/L|µmol/L|xULN))?",
                    re.IGNORECASE,
                ),
                0.95,
            ),
        ),
    ),
    "search_date_by_regex": (
        "clinical_dates_v1",
        (
            ("iso_date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), 0.96),
            (
                "local_date",
                re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b"),
                0.86,
            ),
        ),
    ),
    "search_frequency_by_regex": (
        "clinical_frequency_v1",
        (
            (
                "numeric_schedule",
                re.compile(r"\b\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?){2,3}\b"),
                0.94,
            ),
            (
                "frequency_term",
                re.compile(
                    r"\b(?:once daily|twice daily|bid|tid|qid|q\d+h|od|die)\b",
                    re.IGNORECASE,
                ),
                0.86,
            ),
        ),
    ),
    "search_route_by_regex": (
        "clinical_route_v1",
        (
            (
                "route_term",
                re.compile(
                    r"\b(?:oral|per os|po|iv|ev|intravenous|sc|subcutaneous|im|intramuscular)\b",
                    re.IGNORECASE,
                ),
                0.9,
            ),
        ),
    ),
}


###############################################################################
def _line_number_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


###############################################################################
def _iter_matches(
    *,
    text: str,
    source_section: str,
    patterns: Iterable[RegexSpec],
) -> list[RegexToolMatch]:
    matches: list[RegexToolMatch] = []
    seen: set[tuple[int, int, str]] = set()
    for pattern_id, pattern, confidence in patterns:
        for match in pattern.finditer(text):
            group = match.group(1) if match.lastindex else match.group(0)
            start, end = match.span(1) if match.lastindex else match.span(0)
            normalized = " ".join(group.strip().split())
            if not normalized:
                continue
            key = (start, end, pattern_id)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                RegexToolMatch(
                    match_text=text[start:end],
                    normalized_value=normalized.casefold(),
                    start_char=start,
                    end_char=end,
                    line_number=_line_number_for_offset(text, start),
                    source_section=source_section,
                    pattern_id=pattern_id,
                    confidence=confidence,
                    warnings=[],
                )
            )
    matches.sort(key=lambda item: (item.start_char, item.end_char, item.pattern_id))
    return matches


###############################################################################
def run_regex_tool(name: str, request: RegexToolRequest) -> RegexToolResult:
    profile, patterns = TOOL_PATTERNS[name]
    requested_profile = request.profile or profile
    warnings: list[str] = []
    if requested_profile != profile:
        warnings.append(f"Unsupported profile '{requested_profile}', used '{profile}'.")
    text = request.text or ""
    return RegexToolResult(
        tool_name=name,
        source_section=request.source_section,
        profile=profile,
        matches=_iter_matches(
            text=text,
            source_section=request.source_section,
            patterns=patterns,
        ),
        warnings=warnings,
    )
