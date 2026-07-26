from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from domain.clinical import DrugEntry, PatientDrugs
from services.clinical.knowledge import ClinicalKnowledgeComposer
from services.clinical.matches_core import LiverToxMatcher
from services.clinical.preparation import ClinicalKnowledgePreparation

###############################################################################
class SerializerStub:

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def get_livertox_records(self) -> Any:
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def load_livertox_match_from_db_cache(
        *,
        normalized_drug_key: str,
    ) -> None:
        _ = normalized_drug_key
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

###############################################################################
def build_matcher(*, excerpt: str = "LiverTox excerpt.") -> LiverToxMatcher:
    return LiverToxMatcher(
        pd.DataFrame(
            [
                {
                    "drug_id": 101,
                    "nbk_id": "NBK101",
                    "drug_name": "Acetaminophen",
                    "excerpt": excerpt,
                    "synonyms": "Paracetamol",
                    "ingredient": "Acetaminophen",
                    "brand_name": "Tylenol",
                }
            ]
        )
    )

###############################################################################
def test_prepare_inputs_enriches_resolved_drugs_with_knowledge() -> None:
    preparation = object.__new__(ClinicalKnowledgePreparation)
    preparation.knowledge_repository = SerializerStub()  # type: ignore[assignment]
    preparation.drug_catalog_repository = preparation.knowledge_repository  # type: ignore[assignment]
    preparation.knowledge_composer = ClinicalKnowledgeComposer(
        knowledge_repository=preparation.knowledge_repository  # type: ignore[arg-type]
    )
    preparation.livertox_matcher = build_matcher()

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

###############################################################################
def test_prepare_inputs_handles_missing_livertox_monographs() -> None:
    preparation = object.__new__(ClinicalKnowledgePreparation)
    preparation.knowledge_repository = SerializerStub()  # type: ignore[assignment]
    preparation.drug_catalog_repository = preparation.knowledge_repository  # type: ignore[assignment]
    preparation.knowledge_composer = ClinicalKnowledgeComposer(
        knowledge_repository=preparation.knowledge_repository  # type: ignore[arg-type]
    )
    preparation.livertox_matcher = build_matcher(excerpt="")

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
