from __future__ import annotations

from collections.abc import Callable
from typing import Any

from domain.clinical.drug_resolution import (
    DrugResolutionDecision,
    LiverToxResolutionCandidate,
)
from domain.clinical.entities import PatientDrugs
from services.clinical.drug_resolution.livertox_resolver import LiverToxCandidateResolver
from services.clinical.drug_resolution.normalizer import (
    DrugMentionNormalizer,
    NormalizedDrugMention,
)
from services.clinical.drug_resolution.policy import DrugResolutionPolicy
from services.clinical.drug_resolution.rxnav_resolver import RxNavCandidateResolver
from services.clinical.drug_resolution.serialization import decision_to_payload
from services.clinical.matches_core import LiverToxMatcher


CacheLookupFn = Callable[[str], dict[str, Any] | None]


###############################################################################
class DrugResolutionService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        matcher: LiverToxMatcher,
        *,
        cache_lookup: CacheLookupFn | None = None,
    ) -> None:
        self.matcher = matcher
        self.cache_lookup = cache_lookup
        self.normalizer = DrugMentionNormalizer()
        self.rxnav_resolver = RxNavCandidateResolver(matcher)
        self.livertox_resolver = LiverToxCandidateResolver(matcher)
        self.policy = DrugResolutionPolicy()

    # -------------------------------------------------------------------------
    def resolve(self, drugs: PatientDrugs) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        mentions = self.normalizer.normalize_entries(drugs)
        for mention in mentions:
            cached = self._try_cache(mention)
            if cached is not None:
                resolved[cached["lookup_key"]] = self._merge_payload(
                    resolved.get(cached["lookup_key"]),
                    cached,
                )
                continue

            rxnav_candidates = self.rxnav_resolver.build_candidates(mention)
            rxnav_names = [candidate.name for candidate in rxnav_candidates]
            livertox_candidates = self.livertox_resolver.build_candidates(
                mention,
                rxnav_names,
            )
            livertox_candidates = self._exclude_cross_mention_candidates(
                mention,
                mentions,
                livertox_candidates,
            )
            decision = self.policy.decide(
                mention,
                rxnav_candidates,
                livertox_candidates,
            )
            matched_row, excerpts = self._accepted_livertox_evidence(decision)
            payload = decision_to_payload(
                mention,
                decision,
                matched_row=matched_row,
                excerpts=excerpts,
            )
            resolved[payload["lookup_key"]] = self._merge_payload(
                resolved.get(payload["lookup_key"]),
                payload,
            )
        return resolved

    # -------------------------------------------------------------------------
    @staticmethod
    def _exclude_cross_mention_candidates(
        mention: NormalizedDrugMention,
        all_mentions: list[NormalizedDrugMention],
        candidates: list[LiverToxResolutionCandidate],
    ) -> list[LiverToxResolutionCandidate]:
        current_tokens = set(mention.normalized_name.split())
        other_token_sets = [
            set(other.normalized_name.split())
            for other in all_mentions
            if other.normalized_name != mention.normalized_name
        ]
        filtered: list[LiverToxResolutionCandidate] = []
        for candidate in candidates:
            candidate_tokens = set(candidate.normalized_name.split())
            belongs_to_other_mention = any(
                other_tokens
                and (
                    other_tokens <= candidate_tokens
                    or candidate_tokens <= other_tokens
                )
                for other_tokens in other_token_sets
            )
            related_to_current = bool(current_tokens & candidate_tokens)
            if belongs_to_other_mention and not related_to_current:
                continue
            filtered.append(candidate)
        return filtered

    # -------------------------------------------------------------------------
    def _try_cache(
        self,
        mention: NormalizedDrugMention,
    ) -> dict[str, Any] | None:
        if self.cache_lookup is None:
            return None
        cached = self.cache_lookup(mention.normalized_name)
        if cached is None:
            return None

        has_rxcui = bool(cached.get("rxnorm_rxcui"))
        exact_name = (
            cached.get("normalized_drug_name") == mention.normalized_name
        )
        if has_rxcui:
            status = "accepted_rxnav_validated"
            rxnav_status = "exact_rxcui"
        elif exact_name:
            status = "accepted_exact_livertox"
            rxnav_status = "not_applicable_livertox_direct"
        else:
            status = "accepted_livertox_without_rxnav"
            rxnav_status = "not_applicable_livertox_direct"

        livertox_candidate = LiverToxResolutionCandidate(
            nbk_id=cached.get("nbk_id"),
            drug_name=str(cached.get("drug_name") or cached.get("normalized_drug_name") or ""),
            normalized_name=cached.get("normalized_drug_name") or "",
            monograph_key=cached.get("monograph_key"),
            has_excerpt=bool(cached.get("excerpt")),
            confidence=cached.get("confidence"),
            reason="cache_hit",
            accepted=True,
        )

        decision = DrugResolutionDecision(
            extracted_name=mention.extracted_name,
            normalized_extracted_name=mention.normalized_name,
            source=mention.source,
            regimen_group_id=mention.regimen_group_id,
            is_regimen_parent=mention.is_regimen_parent,
            regimen_components=mention.regimen_components,
            rxnav_candidates=[],
            accepted_rxnav_rxcui=cached.get("rxnorm_rxcui"),
            rxnav_validation_status=rxnav_status,
            livertox_candidates=[livertox_candidate],
            accepted_livertox_nbk_id=cached.get("nbk_id"),
            accepted_livertox_name=livertox_candidate.drug_name,
            accepted_livertox_match_has_excerpt=livertox_candidate.has_excerpt,
            decision_status=status,
            confidence=cached.get("confidence"),
            reasons=["cache_hit_previous_match"],
            requires_human_review=False,
        )

        matched_row: dict[str, Any] | None = None
        if livertox_candidate.nbk_id or livertox_candidate.drug_name:
            matched_row = {
                "nbk_id": cached.get("nbk_id"),
                "drug_name": livertox_candidate.drug_name,
                "drug_name_norm": cached.get("normalized_drug_name"),
                "excerpt": cached.get("excerpt"),
                "likelihood_score": cached.get("likelihood_score"),
                "reference_count": cached.get("reference_count"),
                "agent_classification": cached.get("agent_classification"),
                "primary_classification": cached.get("primary_classification"),
                "secondary_classification": cached.get("secondary_classification"),
            }
        excerpts = [cached["excerpt"]] if cached.get("excerpt") else []

        return decision_to_payload(
            mention,
            decision,
            matched_row=matched_row,
            excerpts=excerpts,
        )

    # -------------------------------------------------------------------------
    def _accepted_livertox_evidence(
        self,
        decision: Any,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        if not decision.accepted_livertox_name:
            return None, []
        mapping = self.matcher.build_drugs_to_excerpt_mapping(
            [decision.accepted_livertox_name],
            self.matcher.match_drug_names([decision.accepted_livertox_name]),
        )
        if not mapping:
            return None, []
        item = mapping[0]
        return item.get("matched_livertox_row"), list(item.get("extracted_excerpts") or [])

    # -------------------------------------------------------------------------
    @staticmethod
    def _merge_payload(
        existing: dict[str, Any] | None,
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        if existing is None:
            return incoming
        for field_name in (
            "origins",
            "raw_mentions",
            "regimen_group_ids",
            "regimen_components",
        ):
            existing[field_name] = list(
                dict.fromkeys(existing.get(field_name, []) + incoming.get(field_name, []))
            )
        existing["extraction_metadata"] = (
            existing.get("extraction_metadata", []) + incoming.get("extraction_metadata", [])
        )
        existing.setdefault("resolution_decisions", [])
        existing["resolution_decisions"].append(incoming["resolution_decision"])
        return existing
