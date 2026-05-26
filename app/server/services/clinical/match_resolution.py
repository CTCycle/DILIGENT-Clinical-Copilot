from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal, Sequence

from services.text.normalization import normalize_drug_query_name


@dataclass(frozen=True)
class DrugEvidenceMatchCandidate:
    candidate_name: str
    source: Literal["extracted", "rxnav_alias", "llm_normalized"]
    normalized_name: str
    rxnav_rxcui: str | None = None


@dataclass(frozen=True)
class DrugEvidenceMatchResult:
    extracted_name: str
    matched_livertox_name: str
    matched_livertox_id: str | None
    rxnav_validated: bool
    rxnav_rxcui: str | None
    match_strategy: Literal[
        "cache",
        "direct_livertox",
        "rxnav_alias_exact",
        "rxnav_alias_partial",
        "rxnav_alias_fuzzy",
        "llm_normalized_retry",
    ]
    supplementary_information_available: bool


def conservative_fuzzy_livertox_match(
    candidate_names: Sequence[str],
    livertox_names: Sequence[str],
) -> str | None:
    best_name: str | None = None
    best_score = 0.0
    for candidate in candidate_names:
        normalized_candidate = normalize_drug_query_name(candidate)
        if not normalized_candidate:
            continue
        for livertox_name in livertox_names:
            normalized_livertox = normalize_drug_query_name(livertox_name)
            if not normalized_livertox:
                continue
            score = SequenceMatcher(
                None, normalized_candidate, normalized_livertox
            ).ratio()
            if score > best_score:
                best_score = score
                best_name = livertox_name
    if best_score >= 0.93:
        return best_name
    return None

