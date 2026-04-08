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
                payload["dili_annotations"] = []
                payload["label_sections"] = []
                payload["knowledge_prompt"] = ""
                continue
            drug_id = self.serializer.to_int(matched_row.get("drug_id"))
            payload["drug_id"] = drug_id
            if drug_id is None:
                payload["dili_annotations"] = []
                payload["label_sections"] = []
                payload["knowledge_prompt"] = ""
                continue
            bundle = self.serializer.get_drug_knowledge_bundle(drug_id)
            dili_annotations = bundle.get("dili_annotations") or []
            label_sections = bundle.get("label_sections") or []
            payload["dili_annotations"] = dili_annotations
            payload["label_sections"] = label_sections
            payload["knowledge_prompt"] = self.build_combined_prompt_fragment(
                livertox_excerpt=self.select_livertox_excerpt(payload),
                dili_annotations=dili_annotations,
                label_sections=label_sections,
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
    def build_dili_prior_block(self, annotations: list[dict[str, Any]]) -> str:
        if not annotations:
            return "DILI Priors: no DILIrank/DILIst prior annotations available."
        lines = ["DILI Priors (DILIrank/DILIst; prior-risk annotations only):"]
        for item in annotations:
            source = str(item.get("source_dataset") or "unknown").strip().upper()
            classification = str(item.get("classification") or "not reported").strip()
            severity = str(item.get("severity_class") or "not reported").strip()
            concern = str(item.get("concern_class") or "not reported").strip()
            lines.append(
                f"- {source}: class={classification}; severity={severity}; concern={concern}"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def build_label_section_block(self, sections: list[dict[str, Any]]) -> str:
        if not sections:
            return "DailyMed official label: no retained hepatic-relevant sections available."
        lines = ["DailyMed official label sections:"]
        for section in sections:
            title = str(section.get("section_title") or section.get("section_key") or "Section")
            text = str(section.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"- {title}:\n{text}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def build_combined_prompt_fragment(
        self,
        *,
        livertox_excerpt: str,
        dili_annotations: list[dict[str, Any]],
        label_sections: list[dict[str, Any]],
    ) -> str:
        blocks: list[str] = []
        blocks.append(
            "LiverTox excerpt:\n"
            + (livertox_excerpt.strip() or "No local LiverTox excerpt available.")
        )
        blocks.append(self.build_dili_prior_block(dili_annotations))
        blocks.append(self.build_label_section_block(label_sections))
        return "\n\n".join(blocks).strip()
