from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from domain.clinical import DrugEntry, PatientDrugs
from services.clinical.matches_core import LiverToxMatcher
from services.clinical.preparation import ClinicalKnowledgePreparation


###############################################################################
def test_prepare_inputs_exposes_resolution_audit_payload() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK100",
                "drug_name": "Acetaminophen",
                "excerpt": "Acetaminophen excerpt.",
                "synonyms": "Tylenol",
                "ingredient": "Acetaminophen",
                "brand_name": "Tylenol",
            }
        ]
    )
    catalog = pd.DataFrame(
        [
            {
                "rxcui": "161",
                "term_type": "IN",
                "raw_name": "Acetaminophen",
                "name": "Acetaminophen",
                "brand_names": "Tylenol",
                "synonyms": '["Tylenol"]',
            }
        ]
    )
    preparation = ClinicalKnowledgePreparation()
    preparation.livertox_matcher = LiverToxMatcher(frame, drugs_catalog_df=catalog)

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name="Tylenol", source="therapy")]),
            clinical_context="",
            pattern_score=None,
        )
    )

    assert prepared is not None
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["resolution_decision"]
    assert payload["rxnav_candidates"]
    assert payload["livertox_candidates"]
    assert payload["accepted_rxnav_rxcui"] == "161"
    assert payload["accepted_livertox_nbk_id"] == "NBK100"
    assert payload["requires_human_review"] is False


###############################################################################
@pytest.mark.parametrize(
    ("extracted_name", "generic_name", "brand_name"),
    [
        ("Furadantin retard", "Nitrofurantoin", "Furadantin"),
        ("Pregabalin Pfizer", "Pregabalin", "Pregabalin"),
        ("Quviviq", "Daridorexant", "Quviviq"),
        ("Xarelto", "Rivaroxaban", "Xarelto"),
        ("Benerva", "Vitamin B1", "Benerva"),
        ("Neurontin", "Gabapentin", "Neurontin"),
    ],
)
def test_brand_and_qualified_names_resolve_through_catalog_normalization(
    extracted_name: str,
    generic_name: str,
    brand_name: str,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK-BRAND",
                "drug_name": generic_name,
                "excerpt": f"{generic_name} excerpt.",
                "synonyms": brand_name,
                "ingredient": generic_name,
                "brand_name": brand_name,
            }
        ]
    )
    catalog = pd.DataFrame(
        [
            {
                "rxcui": "12345",
                "term_type": "IN",
                "raw_name": generic_name,
                "name": generic_name,
                "brand_names": brand_name,
                "synonyms": f'["{brand_name}"]',
            }
        ]
    )
    preparation = ClinicalKnowledgePreparation()
    preparation.livertox_matcher = LiverToxMatcher(frame, drugs_catalog_df=catalog)

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name=extracted_name, source="therapy")]),
            clinical_context="",
            pattern_score=None,
        )
    )

    assert prepared is not None
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["accepted_livertox_nbk_id"] == "NBK-BRAND"
    assert payload["match_confidence"] >= 0.9


###############################################################################
def test_brand_qualified_duplicate_mentions_are_merged() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK-PREGABALIN",
                "drug_name": "Pregabalin",
                "excerpt": "Pregabalin excerpt.",
                "synonyms": "Pregabalin",
                "ingredient": "Pregabalin",
                "brand_name": "Pregabalin",
            }
        ]
    )
    preparation = ClinicalKnowledgePreparation()
    preparation.livertox_matcher = LiverToxMatcher(frame)

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(
                entries=[
                    DrugEntry(name="Pregabalin", source="therapy"),
                    DrugEntry(name="Pregabalin Pfizer", source="therapy"),
                ]
            ),
            clinical_context="",
            pattern_score=None,
        )
    )

    assert prepared is not None
    assert len(prepared.resolved_drugs) == 1
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["raw_mentions"] == ["Pregabalin", "Pregabalin Pfizer"]
