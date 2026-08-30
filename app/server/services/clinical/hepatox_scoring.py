from __future__ import annotations

import re

from services.catalogs.runtime import get_reference_catalog_snapshot


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
