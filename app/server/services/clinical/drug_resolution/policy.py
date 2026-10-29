# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from domain.clinical.drug_resolution import (
    DrugDecisionStatus,
    DrugResolutionDecision,
    LiverToxResolutionCandidate,
    NormalizedDrugMention,
)
from domain.terminology import TerminologyResolution
from services.catalogs.runtime import get_reference_catalog_snapshot

# Explicit link kinds, most direct first.
LINK_PRIORITY = {
    "monograph_owner": 0,
    "component_set": 1,
    "verified_alias": 2,
    "form_of_parent": 3,
    "form_of_child": 4,
}

###############################################################################
class DrugResolutionPolicy:
    """Decide clinical knowledge eligibility from terminology identity.

    LiverTox evidence is attached only for accepted identities, and only via
    explicit concept-to-monograph links; an uncertain identity stays explicit.
    """

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.false_positive_tokens = set(
            get_reference_catalog_snapshot().values(
                "drug_matching",
                "false_positive_drug_tokens",
            )
        )

    # -------------------------------------------------------------------------
    def decide(
        self,
        mention: NormalizedDrugMention,
        terminology: TerminologyResolution,
        livertox_candidates: list[LiverToxResolutionCandidate],
    ) -> DrugResolutionDecision:
        reasons = list(terminology.review_reasons)
        if self._is_false_positive(mention):
            return self._decision(
                mention,
                terminology,
                [],
                None,
                "rejected_false_positive",
                ["extracted text is not a concrete drug mention"],
                requires_review=False,
            )
        if terminology.status == "ambiguous":
            return self._decision(
                mention, terminology, [], None, "ambiguous_requires_review",
                reasons or ["terminology_ambiguous"], requires_review=True,
            )
        if not terminology.accepted:
            return self._decision(
                mention, terminology, [], None, "unresolved",
                reasons or ["terminology_unresolved"], requires_review=True,
            )
        ranked = sorted(livertox_candidates, key=lambda item: LINK_PRIORITY.get(item.link_type, 9))
        chosen = ranked[0] if ranked else None
        if chosen is None:
            return self._decision(
                mention, terminology, [], None, "missing_livertox",
                [*reasons, "no_livertox_record_linked_to_concept"],
                requires_review=terminology.status != "verified",
            )
        chosen.accepted = True
        status: DrugDecisionStatus = (
            "accepted_verified" if terminology.status == "verified" else "accepted_probable"
        )
        return self._decision(
            mention,
            terminology,
            ranked,
            chosen,
            status,
            [*reasons, f"livertox_link_{chosen.link_type}"],
            requires_review=status == "accepted_probable",
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _decision(
        mention: NormalizedDrugMention,
        terminology: TerminologyResolution,
        candidates: list[LiverToxResolutionCandidate],
        chosen: LiverToxResolutionCandidate | None,
        status: DrugDecisionStatus,
        reasons: list[str],
        *,
        requires_review: bool,
    ) -> DrugResolutionDecision:
        return DrugResolutionDecision(
            extracted_name=mention.extracted_name,
            normalized_extracted_name=mention.normalized_name,
            source=mention.source,
            regimen_group_id=mention.regimen_group_id,
            is_regimen_parent=mention.is_regimen_parent,
            regimen_components=mention.regimen_components,
            terminology=terminology,
            livertox_candidates=candidates,
            accepted_livertox_nbk_id=chosen.nbk_id if chosen else None,
            accepted_livertox_name=chosen.drug_name if chosen else None,
            accepted_livertox_match_has_excerpt=bool(chosen and chosen.has_excerpt),
            decision_status=status,
            reasons=list(dict.fromkeys(reasons)),
            requires_human_review=requires_review,
        )

    # -------------------------------------------------------------------------
    def _is_false_positive(self, mention: NormalizedDrugMention) -> bool:
        tokens = set(mention.normalized_name.split())
        return bool(tokens & self.false_positive_tokens) and len(tokens) <= 3
