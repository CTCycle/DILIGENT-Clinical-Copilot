# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

from domain.clinical.drug_resolution import (
    DrugResolutionDecision,
    NormalizedDrugMention,
)
from services.text.normalization import normalize_drug_query_name

###############################################################################
def decision_to_payload(
    mention: NormalizedDrugMention,
    decision: DrugResolutionDecision,
    *,
    matched_row: dict[str, Any] | None,
    excerpts: list[str],
) -> dict[str, Any]:
    primary_reason = next(
        (str(reason).strip() for reason in decision.reasons if str(reason).strip()),
        None,
    )
    accepted_candidate = next(
        (candidate for candidate in decision.livertox_candidates if candidate.accepted),
        None,
    )
    normalized_lookup = (
        accepted_candidate.normalized_name
        if decision.accepted_livertox_name and accepted_candidate is not None
        else mention.normalized_name
    )
    missing_livertox = decision.decision_status in {
        "missing_livertox",
        "ambiguous_requires_review",
        "rejected_false_positive",
    }
    payload = {
        "lookup_key": normalized_lookup,
        "drug_name": decision.accepted_livertox_name or mention.canonical_name,
        "canonical_name": decision.accepted_livertox_name or mention.canonical_name,
        "normalized_name": normalized_lookup,
        "matched_livertox_row": matched_row,
        "extracted_excerpts": excerpts,
        "match_confidence": decision.confidence,
        "match_reason": primary_reason,
        "match_status": decision.decision_status,
        "match_notes": decision.reasons,
        "match_candidates": [
            candidate.model_dump(mode="json")
            for candidate in decision.livertox_candidates
        ],
        "chosen_candidate": decision.accepted_livertox_name,
        "rejected_candidates": [
            candidate.model_dump(mode="json")
            for candidate in decision.livertox_candidates
            if candidate.rejected_reason
        ],
        "missing_livertox": missing_livertox,
        "ambiguous_match": decision.decision_status == "ambiguous_requires_review",
        "origins": mention.origins,
        "raw_mentions": mention.raw_mentions,
        "extraction_metadata": mention.extraction_metadata,
        "regimen_group_ids": [mention.regimen_group_id]
        if mention.regimen_group_id
        else [],
        "is_regimen_parent": mention.is_regimen_parent,
        "regimen_components": mention.regimen_components,
        "rxnav_candidates": [
            candidate.model_dump(mode="json") for candidate in decision.rxnav_candidates
        ],
        "livertox_candidates": [
            candidate.model_dump(mode="json")
            for candidate in decision.livertox_candidates
        ],
        "accepted_rxnav_rxcui": decision.accepted_rxnav_rxcui,
        "accepted_livertox_nbk_id": decision.accepted_livertox_nbk_id,
        "accepted_livertox_name": decision.accepted_livertox_name,
        "accepted_livertox_match_has_excerpt": decision.accepted_livertox_match_has_excerpt,
        "requires_human_review": decision.requires_human_review,
        "decision_status": decision.decision_status,
        "resolution_decision": decision.model_dump(mode="json"),
        "rxnav_rxcui": decision.accepted_rxnav_rxcui,
        "rxnav_validated": decision.accepted_rxnav_rxcui is not None,
        "rxnav_validation_status": decision.rxnav_validation_status,
    }
    return payload

###############################################################################
def index_resolved_drugs(
    resolved_drugs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Resolved payloads keyed by normalized lookup key and by each raw mention.

    The identity fallback re-keys an accepted payload under its LiverTox name
    (for example "valproate"), so lookups by the extracted mention ("valproic
    acid") must also go through ``raw_mentions``.
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
