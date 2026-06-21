from __future__ import annotations

import re

from services.catalogs.runtime import get_reference_catalog_snapshot

###############################################################################
NOT_AVAILABLE_TEXT = "Not available"
REDUNDANT_REPORT_LINE_RE = re.compile(
    r"generated\s+report.*?(drug[- ]induced\s+liver\s+injury|\bdili\b)",
    re.IGNORECASE,
)
LIVERTOX_TITLE_LINE_RE = re.compile(
    r"^\s*\*{0,2}[^*\n]+?\s*-\s*LiverTox score\b.*\*{0,2}\s*$",
    re.IGNORECASE,
)
REPORT_LABEL_LINE_RE = re.compile(r"^\s*\*{0,2}\s*Report\s*\*{0,2}\s*$", re.IGNORECASE)
BIBLIOGRAPHY_LINE_RE = re.compile(
    r"^\s*\*{0,2}\s*Bibliography source\s*\*{0,2}\s*:\s*LiverTox\s*$",
    re.IGNORECASE,
)
DRIFT_SECTION_LINE_RE = re.compile(
    r"^\s*(medication|assessment|plan)\s*$", re.IGNORECASE
)
STRUCTURED_DILI_SECTION_LINE_RE = re.compile(
    r"^\s*#{0,6}\s*\*{0,2}\s*Structured\s+DILI\s+Assessment\s+Report\s*\*{0,2}\s*$",
    re.IGNORECASE,
)
RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)


###############################################################################
def is_materially_in_report_language(text: str, report_language: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return True
    language_key = (report_language or "").strip().lower()[:2]
    if language_key == "en":
        return True
    token_pattern = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
    snapshot = get_reference_catalog_snapshot()
    target_markers = set(
        token.casefold()
        for token in snapshot.values(
            "language_detection",
            "clinical_language_scoring_terms",
            key=language_key,
        )
    )
    if not target_markers:
        return True
    english_markers = set(
        token.casefold()
        for token in snapshot.values(
            "language_detection",
            "clinical_language_scoring_terms",
            key="en",
        )
    )
    if not english_markers:
        return True
    target_hits = 0
    english_hits = 0
    for match in token_pattern.finditer(normalized):
        token = match.group(0).casefold()
        if token in target_markers:
            target_hits += 1
        if token in english_markers:
            english_hits += 1
    if target_hits == 0:
        return False
    return target_hits >= english_hits
