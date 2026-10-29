# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

from domain.clinical.drug_resolution import (
    ACCEPTED_DECISION_STATUSES,
    DrugResolutionDecision,
    NormalizedDrugMention,
)
from services.text.normalization import normalize_drug_query_name

UNLINKED_STATUSES = frozenset(
    {"missing_livertox", "ambiguous_requires_review", "unresolved", "rejected_false_positive"}
)

###############################################################################
def decision_to_payload(
    mention: NormalizedDrugMention,
    decision: DrugResolutionDecision,
    *,
    matched_row: dict[str, Any] | None,
    excerpts: list[str],
    rxcui: str | None,
) -> dict[str, Any]:
    terminology = decision.terminology
    accepted_identity = decision.decision_status in ACCEPTED_DECISION_STATUSES or (
        decision.decision_status == "missing_livertox"
    )
    resolved_name = (
        terminology.preferred_name
        if accepted_identity and terminology.preferred_name
        else mention.canonical_name
    )
    normalized_lookup = normalize_drug_query_name(resolved_name) or mention.normalized_name
    candidate_names = [
        *(candidate.drug_name for candidate in decision.livertox_candidates),
        *(candidate.name for candidate in terminology.candidates),
    ]
    return {
        "lookup_key": normalized_lookup,
        "drug_name": resolved_name,
        "canonical_name": resolved_name,
        "normalized_name": normalized_lookup,
        "matched_livertox_row": matched_row,
        "extracted_excerpts": excerpts,
        "match_reason": next((reason for reason in decision.reasons if reason), None),
        "match_status": decision.decision_status,
        "match_notes": decision.reasons,
        "match_candidates": list(dict.fromkeys(name for name in candidate_names if name)),
        "chosen_candidate": decision.accepted_livertox_name,
        "missing_livertox": decision.decision_status in UNLINKED_STATUSES,
        "ambiguous_match": decision.decision_status == "ambiguous_requires_review",
        "origins": mention.origins,
        "raw_mentions": mention.raw_mentions,
        "extraction_metadata": mention.extraction_metadata,
        "regimen_group_ids": [mention.regimen_group_id] if mention.regimen_group_id else [],
        "is_regimen_parent": mention.is_regimen_parent,
        "regimen_components": mention.regimen_components,
        "livertox_candidates": [
            candidate.model_dump(mode="json") for candidate in decision.livertox_candidates
        ],
        "accepted_rxnav_rxcui": rxcui,
        "rxnav_rxcui": rxcui,
        "accepted_livertox_nbk_id": decision.accepted_livertox_nbk_id,
        "accepted_livertox_name": decision.accepted_livertox_name,
        "accepted_livertox_match_has_excerpt": decision.accepted_livertox_match_has_excerpt,
        "requires_human_review": decision.requires_human_review,
        "decision_status": decision.decision_status,
        "terminology": terminology.model_dump(mode="json"),
        "resolution_decision": decision.model_dump(mode="json"),
    }

###############################################################################
def index_resolved_drugs(
    resolved_drugs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Resolved payloads keyed by normalized lookup key and by each raw mention.

    A payload is keyed by its resolved concept name (for example
    "acetaminophen"), so lookups by the extracted mention ("Dafalgan") must
    also go through ``raw_mentions``.
    """
    index: dict[str, dict[str, Any]] = {}
    payloads = [
        (key, payload)
        for key, payload in (resolved_drugs or {}).items()
        if isinstance(payload, dict)
    ]
    for key, payload in payloads:
        normalized_key = normalize_drug_query_name(key)
        if normalized_key:
            index[normalized_key] = payload
    for _key, payload in payloads:
        for mention in payload.get("raw_mentions") or []:
            normalized_mention = normalize_drug_query_name(str(mention))
            if normalized_mention:
                index.setdefault(normalized_mention, payload)
    return index
