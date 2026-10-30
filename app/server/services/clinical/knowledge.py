# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

from common.prompts.clinical_context import (
    build_dilirank_knowledge_fragment,
    build_livertox_knowledge_fragment,
)
from domain.clinical.entities import EvidenceSelection
from repositories import values as repository_values
from repositories.context import RepositoryContext
from repositories.dilirank_repository import DiliRankRepository
from repositories.knowledge_repository import KnowledgeRepository
from services.text.normalization import normalize_drug_query_name

DILIRANK_PROVENANCE_KEY = "fda_dilirank_2"
# Monograph fields that only identify the drug; every other field is LiverTox
# evidence and is withheld when the session did not select LiverTox.
LIVERTOX_IDENTITY_FIELDS = frozenset({"drug_id", "drug_name", "drug_name_norm"})

###############################################################################
class ClinicalKnowledgeComposer:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        knowledge_repository: KnowledgeRepository,
    ) -> None:
        self.knowledge_repository = knowledge_repository
        context = getattr(knowledge_repository, "context", None)
        self.dilirank_repository = (
            DiliRankRepository(context)
            if isinstance(context, RepositoryContext)
            else None
        )

    # -------------------------------------------------------------------------
    def enrich_resolved_drugs(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
        evidence: EvidenceSelection,
    ) -> None:
        """Attach knowledge from the session's selected sources only.

        Drug identity (``drug_id``, terminology) is resolved regardless of the
        selection; unselected sources are never queried and their evidence is
        withheld, so downstream consumers see "not consulted" rather than a
        negative finding.
        """
        dilirank_available = self._dilirank_available() if evidence.dilirank else False
        for payload in resolved_drugs.values():
            matched_row = payload.get("matched_livertox_row")
            drug_id = self._knowledge_drug_id(payload, matched_row)
            payload["drug_id"] = drug_id
            if not evidence.livertox:
                self._withhold_livertox_evidence(payload)
            if drug_id is None:
                payload["dilirank_records"] = []
                payload["knowledge_prompt"] = ""
                payload["evidence_status"] = self._evidence_status(
                    evidence,
                    payload,
                    dilirank_records=[],
                    dilirank_available=dilirank_available,
                )
                payload["source_provenance"] = self._build_source_provenance(
                    payload
                )
                continue
            bundle = (
                self.knowledge_repository.get_drug_knowledge_bundle(drug_id)
                if evidence.livertox and isinstance(matched_row, dict)
                else {}
            )
            livertox_excerpt = self.select_livertox_excerpt(payload)
            if evidence.livertox and not livertox_excerpt:
                livertox_excerpt = str(bundle.get("livertox_excerpt") or "")
            dilirank_records = (
                self.dilirank_repository.get_records_for_drug(drug_id)
                if dilirank_available and self.dilirank_repository is not None
                else []
            )
            payload["livertox_monographs"] = bundle.get("livertox_monographs") or []
            payload["dilirank_records"] = dilirank_records
            self._attach_dilirank_provenance(payload, dilirank_records)
            payload["knowledge_prompt"] = self.build_combined_prompt_fragment(
                livertox_excerpt=livertox_excerpt,
                dilirank_records=dilirank_records,
                evidence=evidence,
            )
            payload["evidence_status"] = self._evidence_status(
                evidence,
                payload,
                dilirank_records=dilirank_records,
                dilirank_available=dilirank_available,
            )
            payload["source_provenance"] = self._build_source_provenance(payload)

        if evidence.dilirank:
            self._enrich_regimen_component_provenance(resolved_drugs, evidence)

    # -------------------------------------------------------------------------
    def _dilirank_available(self) -> bool:
        return (
            self.dilirank_repository is not None
            and self.dilirank_repository.count_records() > 0
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _withhold_livertox_evidence(payload: dict[str, Any]) -> None:
        matched_row = payload.get("matched_livertox_row")
        if isinstance(matched_row, dict):
            payload["matched_livertox_row"] = {
                key: value
                for key, value in matched_row.items()
                if key in LIVERTOX_IDENTITY_FIELDS
            }
        payload["extracted_excerpts"] = []
        payload["livertox_monographs"] = []
        payload["accepted_livertox_match_has_excerpt"] = False

    # -------------------------------------------------------------------------
    @staticmethod
    def _evidence_status(
        evidence: EvidenceSelection,
        payload: dict[str, Any],
        *,
        dilirank_records: list[dict[str, Any]],
        dilirank_available: bool,
    ) -> dict[str, str]:
        if not evidence.livertox:
            livertox = "not_selected"
        elif payload.get("extracted_excerpts") or payload.get("livertox_monographs"):
            livertox = "retrieved"
        else:
            livertox = "no_record"
        if not evidence.dilirank:
            dilirank = "not_selected"
        elif not dilirank_available:
            dilirank = "unavailable"
        elif dilirank_records:
            dilirank = "retrieved"
        else:
            dilirank = "no_record"
        return {"livertox": livertox, "dilirank": dilirank}

    # -------------------------------------------------------------------------
    @staticmethod
    def _knowledge_drug_id(
        payload: dict[str, Any],
        matched_row: Any,
    ) -> int | None:
        """Concept whose knowledge (LiverTox, DILIrank) applies to this payload.

        A linked LiverTox monograph's owner wins; otherwise an accepted
        terminology identity is used, so DILIrank still applies when LiverTox
        has no record. Unaccepted identities get no knowledge.
        """
        if isinstance(matched_row, dict):
            owner = repository_values.to_int(matched_row.get("drug_id"))
            if owner is not None:
                return owner
        terminology = payload.get("terminology")
        if isinstance(terminology, dict) and terminology.get("accepted"):
            ingredients = terminology.get("ingredients") or []
            if len(ingredients) == 1 and isinstance(ingredients[0], dict):
                return repository_values.to_int(
                    ingredients[0].get("parent_drug_id") or terminology.get("drug_id")
                )
            return repository_values.to_int(terminology.get("drug_id"))
        return None

    # -------------------------------------------------------------------------
    def _enrich_regimen_component_provenance(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
        evidence: EvidenceSelection,
    ) -> None:
        payload_by_normalized_name: dict[str, dict[str, Any]] = {}
        for key, payload in resolved_drugs.items():
            normalized = normalize_drug_query_name(
                str(payload.get("normalized_name") or key)
            )
            if normalized:
                payload_by_normalized_name[normalized] = payload

        for payload in resolved_drugs.values():
            if not payload.get("is_regimen_parent"):
                continue
            components = [
                str(component).strip()
                for component in payload.get("regimen_components") or []
                if str(component).strip()
            ]
            if not components:
                continue

            component_records: list[dict[str, Any]] = []
            component_provenance: list[dict[str, Any]] = []
            seen_component_records: set[tuple[str, str]] = set()
            for component in components:
                component_key = normalize_drug_query_name(component)
                component_payload = payload_by_normalized_name.get(component_key)
                if component_payload is None or component_payload is payload:
                    continue
                records = component_payload.get("dilirank_records") or []
                if not isinstance(records, list) or not records:
                    continue
                component_name = str(
                    component_payload.get("accepted_livertox_name")
                    or component_payload.get("drug_name")
                    or component
                ).strip()
                component_id = component_payload.get("drug_id")
                ltkb_ids: list[str] = []
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    ltkb_id = str(record.get("ltkb_id") or "").strip()
                    record_key = (component_name.casefold(), ltkb_id)
                    if record_key in seen_component_records:
                        continue
                    seen_component_records.add(record_key)
                    if ltkb_id and ltkb_id not in ltkb_ids:
                        ltkb_ids.append(ltkb_id)
                    scoped_record = dict(record)
                    scoped_record["evidence_scope"] = "regimen_component"
                    scoped_record["evidence_component_name"] = component_name
                    scoped_record["evidence_component_drug_id"] = component_id
                    component_records.append(scoped_record)
                if ltkb_ids:
                    component_provenance.append(
                        {
                            "name": component_name,
                            "drug_id": component_id,
                            "ltkb_ids": ltkb_ids,
                        }
                    )

            if not component_records:
                continue

            direct_records = [
                record
                for record in payload.get("dilirank_records") or []
                if isinstance(record, dict)
            ]
            all_records = [*direct_records, *component_records]
            payload["dilirank_records"] = all_records
            scope = (
                "drug_and_regimen_component"
                if direct_records
                else "regimen_component"
            )
            self._attach_dilirank_provenance(
                payload,
                all_records,
                evidence_scope=scope,
                components=component_provenance,
            )
            livertox_excerpt = self.select_livertox_excerpt(payload)
            if not livertox_excerpt:
                livertox_excerpt = str(payload.get("livertox_excerpt") or "")
            payload["knowledge_prompt"] = self.build_combined_prompt_fragment(
                livertox_excerpt=livertox_excerpt,
                dilirank_records=all_records,
                evidence=evidence,
            )
            status = payload.get("evidence_status")
            if isinstance(status, dict):
                status["dilirank"] = "retrieved"
            payload["source_provenance"] = self._build_source_provenance(payload)

    # -------------------------------------------------------------------------
    @staticmethod
    def _attach_dilirank_provenance(
        payload: dict[str, Any],
        records: list[dict[str, Any]],
        *,
        evidence_scope: str | None = None,
        components: list[dict[str, Any]] | None = None,
    ) -> None:
        metadata = payload.get("extraction_metadata")
        metadata_items = list(metadata) if isinstance(metadata, list) else []
        metadata_items = [
            item
            for item in metadata_items
            if not (
                isinstance(item, dict)
                and item.get("knowledge_source") == DILIRANK_PROVENANCE_KEY
            )
        ]
        if records:
            provenance: dict[str, Any] = {
                "knowledge_source": DILIRANK_PROVENANCE_KEY,
                "record_count": len(records),
                "ltkb_ids": [
                    str(record.get("ltkb_id") or "").strip()
                    for record in records
                    if str(record.get("ltkb_id") or "").strip()
                ],
            }
            if evidence_scope:
                provenance["evidence_scope"] = evidence_scope
            if components:
                provenance["components"] = components
            metadata_items.append(provenance)
        payload["extraction_metadata"] = metadata_items

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_source_provenance(payload: dict[str, Any]) -> dict[str, Any]:
        terminology = payload.get("terminology")
        terminology = terminology if isinstance(terminology, dict) else {}
        provenance: dict[str, Any] = {
            "terminology": {
                "status": terminology.get("status"),
                "method": terminology.get("method"),
                "source": terminology.get("source"),
                "source_version": terminology.get("source_version"),
                "matched_alias": terminology.get("matched_alias"),
                "rxcui": payload.get("accepted_rxnav_rxcui"),
            }
        }
        for item in payload.get("extraction_metadata") or []:
            if (
                isinstance(item, dict)
                and item.get("knowledge_source") == DILIRANK_PROVENANCE_KEY
            ):
                provenance["dilirank"] = dict(item)
                break
        return provenance

    # -------------------------------------------------------------------------
    def select_livertox_excerpt(self, payload: dict[str, Any]) -> str:
        excerpts = payload.get("extracted_excerpts")
        if isinstance(excerpts, list):
            chunks = [str(item).strip() for item in excerpts if str(item).strip()]
            if chunks:
                return "\n\n".join(chunks)
        return ""

    # -------------------------------------------------------------------------
    def build_combined_prompt_fragment(
        self,
        *,
        livertox_excerpt: str,
        dilirank_records: list[dict[str, Any]],
        evidence: EvidenceSelection,
    ) -> str:
        fragments: list[str] = []
        if evidence.livertox:
            fragments.append(
                build_livertox_knowledge_fragment(
                    livertox_excerpt=livertox_excerpt.strip(),
                )
            )
        if evidence.dilirank:
            fragments.append(build_dilirank_knowledge_fragment(records=dilirank_records))
        return "\n\n".join(fragments)
