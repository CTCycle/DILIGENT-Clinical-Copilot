# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from dataclasses import replace
from typing import Any

from domain.clinical.drug_resolution import (
    LiverToxResolutionCandidate,
    NormalizedDrugMention,
)
from domain.clinical.entities import PatientDrugs
from domain.terminology import TerminologyResolution
from repositories.knowledge_repository import KnowledgeRepository
from repositories.terminology_lookup_repository import (
    LiverToxLinkInfo,
    TerminologyLookupRepository,
)
from services.clinical.drug_resolution.normalizer import DrugMentionNormalizer
from services.clinical.drug_resolution.policy import DrugResolutionPolicy
from services.clinical.drug_resolution.serialization import decision_to_payload
from services.terminology.normalization import fold
from services.terminology.resolver import TerminologyResolver
from services.text.normalization import normalize_drug_query_name

###############################################################################
class DrugResolutionService:
    """Resolve extracted mentions to terminology concepts and linked LiverTox evidence."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        terminology: TerminologyResolver,
        lookup: TerminologyLookupRepository,
        knowledge_repository: KnowledgeRepository,
    ) -> None:
        self.terminology = terminology
        self.lookup = lookup
        self.knowledge_repository = knowledge_repository
        self.normalizer = DrugMentionNormalizer()
        self.policy = DrugResolutionPolicy()

    # -------------------------------------------------------------------------
    def resolve(self, drugs: PatientDrugs) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for mention in self.normalizer.normalize_entries(drugs):
            query = mention.canonical_name if mention.regimen_group_id and not mention.is_regimen_parent else mention.extracted_name
            terminology = self.terminology.resolve(query)
            for item_mention, item_terminology in self._expand_combination(mention, terminology):
                payload = self._payload(item_mention, item_terminology)
                resolved[payload["lookup_key"]] = self.merge_payload(
                    resolved.get(payload["lookup_key"]), payload
                )
        return resolved

    # -------------------------------------------------------------------------
    def _expand_combination(
        self,
        mention: NormalizedDrugMention,
        terminology: TerminologyResolution,
    ) -> list[tuple[NormalizedDrugMention, TerminologyResolution]]:
        """Split a combination product without a combination monograph into its
        ingredients, so each keeps its own LiverTox evidence."""
        if (
            mention.regimen_group_id
            or not terminology.accepted
            or terminology.drug_id is None
            or len(terminology.ingredients) < 2
            or self.lookup.livertox_links([terminology.drug_id]).get(terminology.drug_id)
        ):
            return [(mention, terminology)]
        names = [ingredient.name for ingredient in terminology.ingredients]
        group_id = "|".join(sorted(fold(name) for name in names))
        expanded = [
            (
                replace(
                    mention,
                    regimen_group_id=group_id,
                    is_regimen_parent=True,
                    regimen_components=names,
                ),
                terminology,
            )
        ]
        for ingredient in terminology.ingredients:
            component_mention = replace(
                mention,
                canonical_name=ingredient.name,
                normalized_name=normalize_drug_query_name(ingredient.name) or fold(ingredient.name),
                raw_mentions=mention.raw_mentions[:],
                origins=mention.origins[:],
                extraction_metadata=mention.extraction_metadata[:],
                regimen_group_id=group_id,
                is_regimen_parent=False,
                regimen_components=names,
            )
            component = terminology.model_copy(
                update={
                    "drug_id": ingredient.drug_id,
                    "preferred_name": ingredient.name,
                    "concept_type": ingredient.concept_type,
                    "ingredients": [ingredient],
                    "precise_forms": [],
                    "candidates": [],
                }
            )
            expanded.append((component_mention, component))
        return expanded

    # -------------------------------------------------------------------------
    def _payload(
        self,
        mention: NormalizedDrugMention,
        terminology: TerminologyResolution,
    ) -> dict[str, Any]:
        links: list[LiverToxLinkInfo] = []
        rxcui: str | None = None
        if terminology.accepted and terminology.drug_id is not None:
            links = self.lookup.livertox_links([terminology.drug_id]).get(terminology.drug_id, [])
            rxcui = self.lookup.rxcuis_for([terminology.drug_id]).get(terminology.drug_id)
        rows = self.knowledge_repository.get_monograph_rows([link.monograph_id for link in links])
        candidates = [
            LiverToxResolutionCandidate(
                nbk_id=link.nbk_id,
                drug_name=link.drug_name,
                normalized_name=normalize_drug_query_name(link.drug_name),
                monograph_key=link.monograph_key,
                has_excerpt=bool((rows.get(link.monograph_id) or {}).get("excerpt")),
                link_type=link.link_type,
            )
            for link in links
        ]
        decision = self.policy.decide(mention, terminology, candidates)
        chosen = next((candidate for candidate in decision.livertox_candidates if candidate.accepted), None)
        matched_row: dict[str, Any] | None = None
        if chosen is not None:
            monograph_id = next(link.monograph_id for link in links if link.monograph_key == chosen.monograph_key)
            matched_row = rows.get(monograph_id)
        excerpts = [str(matched_row["excerpt"])] if matched_row and matched_row.get("excerpt") else []
        return decision_to_payload(
            mention,
            decision,
            matched_row=matched_row,
            excerpts=excerpts,
            rxcui=rxcui,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def merge_payload(
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
                dict.fromkeys(
                    existing.get(field_name, []) + incoming.get(field_name, [])
                )
            )
        existing["extraction_metadata"] = existing.get(
            "extraction_metadata", []
        ) + incoming.get("extraction_metadata", [])
        existing.setdefault("resolution_decisions", [])
        existing["resolution_decisions"].append(incoming["resolution_decision"])
        return existing
