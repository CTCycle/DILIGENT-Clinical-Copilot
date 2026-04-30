from __future__ import annotations

from typing import Any


def classify_match_evidence(
    *,
    match_status: str | None,
    match_reason: str | None,
    match_confidence: float | None,
    match_notes: list[Any] | None,
    missing_livertox: bool,
    ambiguous_match: bool,
) -> dict[str, Any]:
    normalized_status = (match_status or "").strip().lower()
    normalized_reason = (match_reason or "").strip().lower()
    normalized_notes = [
        str(note).strip() for note in (match_notes or []) if str(note).strip()
    ]
    notes = {note.lower() for note in normalized_notes}
    warnings: list[str] = []

    if ambiguous_match or normalized_status in {"ambiguous", "ambiguous_match"}:
        return {
            "evidence_quality": "ambiguous_match",
            "evidence_warnings": [
                "Multiple plausible local evidence matches require review."
            ],
        }

    if normalized_status in {"missing", "missing_match"}:
        return {
            "evidence_quality": "missing_match",
            "evidence_warnings": ["No local RxNav/LiverTox match was found."],
        }

    if missing_livertox or normalized_status == "matched_no_excerpt":
        warnings.append("Matched local drug record has no LiverTox excerpt.")

    if "fallback_excerpt_from_related_monograph" in notes:
        warnings.append(
            "Evidence excerpt was borrowed from a related LiverTox monograph."
        )
        quality = "fallback_related_monograph"
    elif (match_confidence is not None and float(match_confidence) < 1.0) or (
        "alias" in normalized_reason or "spelling_correction" in normalized_reason
    ):
        warnings.append("Drug match is not a direct canonical match.")
        quality = "weak_alias_or_class_match"
    elif normalized_status == "matched_with_excerpt":
        quality = "direct_match_with_excerpt"
    elif normalized_status in {"matched", "matched_no_excerpt", "match"}:
        quality = "direct_match_no_excerpt" if missing_livertox else "direct_match"
    else:
        quality = "unknown"
        warnings.append(
            "Match evidence quality could not be determined from stored metadata."
        )

    return {
        "evidence_quality": quality,
        "evidence_warnings": warnings,
    }
