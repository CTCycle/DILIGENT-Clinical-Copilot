# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

import pytest

from domain.clinical.drug_resolution import LiverToxResolutionCandidate, NormalizedDrugMention
from domain.terminology import TerminologyResolution
from services.clinical.drug_resolution.policy import DrugResolutionPolicy
from services.clinical.preparation import ClinicalKnowledgePreparation


# -----------------------------------------------------------------------------
def mention(name: str) -> NormalizedDrugMention:
    return NormalizedDrugMention(
        extracted_name=name,
        canonical_name=name,
        normalized_name=name.casefold(),
        source="therapy",
        raw_mentions=[name],
        origins=["therapy"],
        extraction_metadata=[],
    )


# -----------------------------------------------------------------------------
def terminology(status: str, *, accepted: bool, **extra: Any) -> TerminologyResolution:
    return TerminologyResolution(
        original="x",
        normalized="x",
        status=status,  # type: ignore[arg-type]
        accepted=accepted,
        method=extra.pop("method", "exact_alias"),
        drug_id=extra.pop("drug_id", 1 if accepted else None),
        preferred_name=extra.pop("preferred_name", "metoprolol" if accepted else None),
        **extra,
    )


# -----------------------------------------------------------------------------
def candidate(link_type: str, name: str = "Metoprolol") -> LiverToxResolutionCandidate:
    return LiverToxResolutionCandidate(
        nbk_id="NBK1", drug_name=name, normalized_name=name.casefold(), has_excerpt=True, link_type=link_type
    )


###############################################################################
def test_verified_identity_prefers_the_most_direct_livertox_link() -> None:
    decision = DrugResolutionPolicy().decide(
        mention("Metoprolol succinate"),
        terminology("verified", accepted=True),
        [candidate("form_of_parent", "Metoprolol"), candidate("monograph_owner", "Metoprolol Succinate")],
    )

    assert decision.decision_status == "accepted_verified"
    assert decision.accepted_livertox_name == "Metoprolol Succinate"
    assert decision.requires_human_review is False


###############################################################################
@pytest.mark.parametrize(
    ("resolution", "links", "status", "review"),
    [
        (terminology("probable", accepted=True, method="fuzzy"), [candidate("monograph_owner")], "accepted_probable", True),
        (terminology("verified", accepted=True), [], "missing_livertox", False),
        (terminology("ambiguous", accepted=False, method="fuzzy"), [candidate("monograph_owner")], "ambiguous_requires_review", True),
        (terminology("probable", accepted=False, method="product_unverified"), [], "unresolved", True),
        (terminology("unresolved", accepted=False, method="none"), [], "unresolved", True),
    ],
)
def test_uncertain_identities_never_borrow_livertox_evidence(
    resolution: TerminologyResolution,
    links: list[LiverToxResolutionCandidate],
    status: str,
    review: bool,
) -> None:
    decision = DrugResolutionPolicy().decide(mention("Metoprolol"), resolution, links)

    assert decision.decision_status == status
    assert decision.requires_human_review is review
    if status != "accepted_probable":
        assert decision.accepted_livertox_name is None


###############################################################################
def test_lab_text_is_rejected_as_a_false_positive() -> None:
    policy = DrugResolutionPolicy()
    token = sorted(policy.false_positive_tokens)[0]

    decision = policy.decide(mention(token), terminology("verified", accepted=True), [candidate("monograph_owner")])

    assert decision.decision_status == "rejected_false_positive"
    assert decision.accepted_livertox_name is None


###############################################################################
@pytest.mark.parametrize(
    ("payload", "codes"),
    [
        ({"decision_status": "unresolved", "terminology": {}}, ["drug_identity_unresolved"]),
        (
            {"decision_status": "ambiguous_requires_review", "terminology": {"candidates": [{"name": "hydroxyzine"}]}},
            ["drug_identity_ambiguous"],
        ),
        (
            {"decision_status": "accepted_probable", "terminology": {"status": "probable", "preferred_name": "amoxicillin"}},
            ["drug_identity_probable"],
        ),
        (
            {"decision_status": "unresolved", "terminology": {"method": "product_unverified"}},
            ["drug_composition_unverified"],
        ),
        (
            {"decision_status": "missing_livertox", "terminology": {"status": "verified", "preferred_name": "metformin"}},
            ["livertox_link_missing"],
        ),
        ({"decision_status": "accepted_verified", "terminology": {"status": "verified"}}, []),
    ],
)
def test_match_audit_codes_follow_terminology_status(payload: dict[str, Any], codes: list[str]) -> None:
    issues = ClinicalKnowledgePreparation.match_audit_issues_for(payload, "Entered name")

    assert [issue.code for issue in issues] == codes
