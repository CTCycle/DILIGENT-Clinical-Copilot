from __future__ import annotations

from typing import Any

from domain.clinical.entities import PatientDrugs
from services.clinical.drug_resolution.livertox_resolver import LiverToxCandidateResolver
from services.clinical.drug_resolution.normalizer import (
    DrugMentionNormalizer,
)
from services.clinical.drug_resolution.policy import DrugResolutionPolicy
from services.clinical.drug_resolution.rxnav_resolver import RxNavCandidateResolver
from services.clinical.drug_resolution.serialization import decision_to_payload
from services.clinical.matches_core import LiverToxMatcher


class DrugResolutionService:
    def __init__(self, matcher: LiverToxMatcher) -> None:
        self.matcher = matcher
        self.normalizer = DrugMentionNormalizer()
        self.rxnav_resolver = RxNavCandidateResolver(matcher)
        self.livertox_resolver = LiverToxCandidateResolver(matcher)
        self.policy = DrugResolutionPolicy()

    def resolve(self, drugs: PatientDrugs) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for mention in self.normalizer.normalize_entries(drugs):
            rxnav_candidates = self.rxnav_resolver.build_candidates(mention)
            rxnav_names = [candidate.name for candidate in rxnav_candidates]
            livertox_candidates = self.livertox_resolver.build_candidates(
                mention,
                rxnav_names,
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
