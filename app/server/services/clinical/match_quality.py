# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

ACCEPTED_STATUSES = frozenset({"accepted_verified", "accepted_probable"})
RELATED_FORM_LINKS = frozenset({"form_of_parent", "form_of_child"})
# Statuses with an accepted drug identity, with or without a LiverTox record.
IDENTITY_ACCEPTED_STATUSES = ACCEPTED_STATUSES | {"missing_livertox"}

###############################################################################
def classify_match_evidence(
    *,
    match_status: str | None,
    terminology: dict[str, Any] | None,
    link_type: str | None,
    has_excerpt: bool,
    missing_livertox: bool,
    ambiguous_match: bool,
    livertox_selected: bool = True,
) -> dict[str, Any]:
    status = (match_status or "").strip().lower()
    identity = terminology or {}
    if ambiguous_match or status == "ambiguous_requires_review":
        return {
            "evidence_quality": "ambiguous_match",
            "evidence_warnings": ["Several drug identities are plausible and require review."],
        }
    if status == "rejected_false_positive":
        return {
            "evidence_quality": "rejected_false_positive",
            "evidence_warnings": ["Extracted text was rejected as a non-drug match."],
        }
    if status == "unresolved" or not status:
        return {
            "evidence_quality": "missing_match",
            "evidence_warnings": ["No verified terminology identity was found."],
        }
    if not livertox_selected and status in IDENTITY_ACCEPTED_STATUSES:
        warnings = (
            ["Drug identity relies on a spelling correction and requires review."]
            if identity.get("status") == "probable"
            else []
        )
        warnings.append("LiverTox was not consulted for this session.")
        return {"evidence_quality": "livertox_not_selected", "evidence_warnings": warnings}
    if status == "missing_livertox" or missing_livertox:
        return {
            "evidence_quality": "missing_match",
            "evidence_warnings": ["No LiverTox record is linked to the resolved drug."],
        }
    if status not in ACCEPTED_STATUSES:
        return {
            "evidence_quality": "unknown",
            "evidence_warnings": ["Match evidence quality could not be determined from stored metadata."],
        }
    warnings: list[str] = []
    if identity.get("status") == "probable":
        warnings.append("Drug identity relies on a spelling correction and requires review.")
        quality = "weak_alias_or_class_match"
    elif link_type in RELATED_FORM_LINKS:
        warnings.append("Evidence comes from the LiverTox record of a related salt or parent form.")
        quality = "fallback_related_monograph"
    else:
        quality = "direct_match_with_excerpt" if has_excerpt else "direct_match_no_excerpt"
    if not has_excerpt:
        warnings.append("Matched LiverTox record has no excerpt.")
    return {"evidence_quality": quality, "evidence_warnings": warnings}


# -----------------------------------------------------------------------------
def accepted_link_type(payload: dict[str, Any]) -> str | None:
    return next(
        (
            str(candidate.get("link_type"))
            for candidate in payload.get("livertox_candidates") or []
            if isinstance(candidate, dict) and candidate.get("accepted")
        ),
        None,
    )
