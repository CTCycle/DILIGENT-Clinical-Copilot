from __future__ import annotations

from typing import Any

from DILIGENT.server.repositories.serialization.data import DataSerializer


###############################################################################
class ClinicalKnowledgeComposer:
    def __init__(self, *, serializer: DataSerializer | None = None) -> None:
        self.serializer = serializer or DataSerializer()

    # -------------------------------------------------------------------------
    def enrich_resolved_drugs(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
    ) -> None:
        for payload in resolved_drugs.values():
            matched_row = payload.get("matched_livertox_row")
            if not isinstance(matched_row, dict):
                payload["drug_id"] = None
                payload["knowledge_prompt"] = ""
                continue
            drug_id = self.serializer.to_int(matched_row.get("drug_id"))
            payload["drug_id"] = drug_id
            if drug_id is None:
                payload["knowledge_prompt"] = ""
                continue
            bundle = self.serializer.get_drug_knowledge_bundle(drug_id)
            livertox_excerpt = self.select_livertox_excerpt(payload)
            if not livertox_excerpt:
                livertox_excerpt = str(bundle.get("livertox_excerpt") or "")
            payload["livertox_monographs"] = bundle.get("livertox_monographs") or []
            payload["knowledge_prompt"] = self.build_combined_prompt_fragment(
                livertox_excerpt=livertox_excerpt,
            )

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
    ) -> str:
        return "LiverTox excerpt:\n" + (
            livertox_excerpt.strip() or "No local LiverTox excerpt available."
        )
