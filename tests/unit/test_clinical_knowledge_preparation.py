from __future__ import annotations

import asyncio
from typing import Any

from DILIGENT.server.domain.clinical import DrugEntry, PatientDrugs
from DILIGENT.server.services.clinical.knowledge import ClinicalKnowledgeComposer
from DILIGENT.server.services.clinical.preparation import ClinicalKnowledgePreparation


###############################################################################
class SerializerStub:
    def get_drug_knowledge_bundle(self, drug_id: int) -> dict[str, Any]:
        if drug_id == 101:
            return {
                "drug_id": 101,
                "drug_name": "Acetaminophen",
                "livertox_excerpt": "LiverTox excerpt.",
                "livertox_monographs": [],
            }
        return {
            "drug_id": drug_id,
            "drug_name": "Unknown",
            "livertox_excerpt": None,
            "livertox_monographs": [],
        }

    def get_livertox_records(self) -> Any:
        return None

    @staticmethod
    def to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


###############################################################################
class MatcherStub:
    def match_drug_names(self, names: list[str]) -> list[dict[str, Any]]:
        _ = names
        return []

    def build_drugs_to_excerpt_mapping(
        self,
        names: list[str],
        matches: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        _ = matches
        return {
            names[0].lower(): {
                "drug_name": names[0],
                "canonical_drug_name": names[0].lower(),
                "normalized_drug_name": names[0].lower(),
                "matched_livertox_row": {"drug_id": 101, "drug_name": "Acetaminophen"},
                "extracted_excerpts": ["LiverTox excerpt."],
                "match_status": "matched_with_excerpt",
                "match_reason": "exact",
            }
        }


# -----------------------------------------------------------------------------
def test_prepare_inputs_enriches_resolved_drugs_with_knowledge() -> None:
    preparation = object.__new__(ClinicalKnowledgePreparation)
    preparation.serializer = SerializerStub()  # type: ignore[assignment]
    preparation.knowledge_composer = ClinicalKnowledgeComposer(
        serializer=preparation.serializer  # type: ignore[arg-type]
    )
    preparation.livertox_matcher = MatcherStub()  # type: ignore[assignment]

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name="Acetaminophen", source="therapy")]),
            clinical_context="clinical context",
            pattern_score=None,
        )
    )

    assert prepared is not None
    assert prepared.resolved_drugs
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["drug_id"] == 101
    assert "LiverTox excerpt." in payload["knowledge_prompt"]
    assert payload["knowledge_prompt"]


# -----------------------------------------------------------------------------
def test_prepare_inputs_handles_missing_livertox_monographs() -> None:
    preparation = object.__new__(ClinicalKnowledgePreparation)
    preparation.serializer = SerializerStub()  # type: ignore[assignment]
    preparation.knowledge_composer = ClinicalKnowledgeComposer(
        serializer=preparation.serializer  # type: ignore[arg-type]
    )
    preparation.livertox_matcher = MatcherStub()  # type: ignore[assignment]

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name="Acetaminophen", source="therapy")]),
            clinical_context="",
            pattern_score=None,
        )
    )

    assert prepared is not None
    payload = next(iter(prepared.resolved_drugs.values()))
    assert "livertox_monographs" in payload
