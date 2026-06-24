from __future__ import annotations

from domain.clinical.drug_resolution import (
    DrugResolutionDecision,
    LiverToxResolutionCandidate,
    RxNavResolutionCandidate,
)
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.clinical.drug_resolution.normalizer import NormalizedDrugMention


###############################################################################
class DrugResolutionPolicy:

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
        rxnav_candidates: list[RxNavResolutionCandidate],
        livertox_candidates: list[LiverToxResolutionCandidate],
    ) -> DrugResolutionDecision:
        if self._is_false_positive(mention):
            return self._base_decision(
                mention,
                rxnav_candidates,
                livertox_candidates,
                status="rejected_false_positive",
                reasons=["extracted text is not a concrete drug mention"],
                requires_review=False,
            )
        accepted_rxnav = self._accepted_rxnav(rxnav_candidates)
        accepted_livertox = self._accepted_livertox(mention, livertox_candidates)
        for candidate in rxnav_candidates:
            candidate.accepted = candidate is accepted_rxnav
            if not candidate.accepted and candidate.rejected_reason is None:
                candidate.rejected_reason = "not selected by deterministic RxNav policy"
        for candidate in livertox_candidates:
            candidate.accepted = candidate is accepted_livertox
            if not candidate.accepted and candidate.rejected_reason is None:
                candidate.rejected_reason = "not selected by deterministic LiverTox policy"
        if accepted_rxnav is None and rxnav_candidates:
            return self._base_decision(
                mention,
                rxnav_candidates,
                livertox_candidates,
                status="ambiguous_requires_review",
                reasons=["RxNav candidates are ambiguous or lack RXCUI provenance"],
                requires_review=True,
            )
        if accepted_livertox is None:
            status = "missing_livertox" if not livertox_candidates else "ambiguous_requires_review"
            return self._base_decision(
                mention,
                rxnav_candidates,
                livertox_candidates,
                status=status,
                accepted_rxnav=accepted_rxnav,
                reasons=["No unique acceptable LiverTox monograph"],
                requires_review=status == "ambiguous_requires_review",
            )
        status = "accepted_livertox_without_rxnav"
        reasons = ["Unique LiverTox monograph accepted without RxNav validation"]
        if accepted_rxnav is not None:
            status = "accepted_rxnav_validated"
            reasons = ["Unique LiverTox monograph accepted with RxNav RXCUI provenance"]
        elif accepted_livertox.normalized_name == mention.normalized_name:
            status = "accepted_exact_livertox"
            reasons = ["Exact normalized LiverTox primary name accepted"]
        return self._base_decision(
            mention,
            rxnav_candidates,
            livertox_candidates,
            status=status,
            accepted_rxnav=accepted_rxnav,
            accepted_livertox=accepted_livertox,
            reasons=reasons,
            requires_review=False,
        )

    # -------------------------------------------------------------------------
    def _accepted_rxnav(
        self, candidates: list[RxNavResolutionCandidate]
    ) -> RxNavResolutionCandidate | None:
        viable = [
            candidate
            for candidate in candidates
            if candidate.rxcui and candidate.rejected_reason is None
        ]
        unique_rxcuis = {candidate.rxcui for candidate in viable}
        if len(unique_rxcuis) == 1 and viable:
            return viable[0]
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _accepted_livertox(
        mention: NormalizedDrugMention,
        candidates: list[LiverToxResolutionCandidate],
    ) -> LiverToxResolutionCandidate | None:
        viable = [candidate for candidate in candidates if candidate.rejected_reason is None]
        if not viable:
            return None
        exact = [
            candidate
            for candidate in viable
            if candidate.normalized_name == mention.normalized_name
        ]
        if len(exact) == 1:
            return exact[0]
        keys = {candidate.monograph_key or candidate.normalized_name for candidate in viable}
        if len(keys) == 1:
            return viable[0]
        return None

    # -------------------------------------------------------------------------
    def _base_decision(
        self,
        mention: NormalizedDrugMention,
        rxnav_candidates: list[RxNavResolutionCandidate],
        livertox_candidates: list[LiverToxResolutionCandidate],
        *,
        status: str,
        reasons: list[str],
        requires_review: bool,
        accepted_rxnav: RxNavResolutionCandidate | None = None,
        accepted_livertox: LiverToxResolutionCandidate | None = None,
    ) -> DrugResolutionDecision:
        rxnav_status = "no_rxnav_match"
        if accepted_rxnav is not None:
            rxnav_status = {
                "brand": "brand_to_rxcui",
                "ingredient": "ingredient_to_rxcui",
                "alias": "alias_to_rxcui",
                "name": "exact_rxcui",
            }.get(accepted_rxnav.alias_kind or "", "alias_to_rxcui")
        elif rxnav_candidates:
            rxnav_status = "ambiguous_rxnav"
        elif status in {"accepted_exact_livertox", "accepted_livertox_without_rxnav"}:
            rxnav_status = "not_applicable_livertox_direct"
        return DrugResolutionDecision(
            extracted_name=mention.extracted_name,
            normalized_extracted_name=mention.normalized_name,
            source=mention.source,
            regimen_group_id=mention.regimen_group_id,
            is_regimen_parent=mention.is_regimen_parent,
            regimen_components=mention.regimen_components,
            rxnav_candidates=rxnav_candidates,
            accepted_rxnav_rxcui=accepted_rxnav.rxcui if accepted_rxnav else None,
            rxnav_validation_status=rxnav_status,
            livertox_candidates=livertox_candidates,
            accepted_livertox_nbk_id=accepted_livertox.nbk_id if accepted_livertox else None,
            accepted_livertox_name=accepted_livertox.drug_name if accepted_livertox else None,
            accepted_livertox_match_has_excerpt=bool(accepted_livertox and accepted_livertox.has_excerpt),
            decision_status=status,  # type: ignore[arg-type]
            confidence=accepted_livertox.confidence if accepted_livertox else None,
            reasons=reasons,
            requires_human_review=requires_review,
        )

    # -------------------------------------------------------------------------
    def _is_false_positive(self, mention: NormalizedDrugMention) -> bool:
        tokens = set(mention.normalized_name.split())
        return bool(tokens & self.false_positive_tokens) and len(tokens) <= 3
