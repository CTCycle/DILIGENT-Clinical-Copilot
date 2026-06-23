from __future__ import annotations

import pandas as pd

from domain.clinical import DrugEntry, PatientDrugs
from services.clinical.drug_resolution import DrugResolutionService
from services.clinical.matches_core import LiverToxMatcher


def _livertox_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nbk_id": "NBK001",
                "drug_name": "Acetaminophen",
                "excerpt": "Acetaminophen can cause dose-related liver injury.",
                "synonyms": "Tylenol; Paracetamol",
                "ingredient": "Acetaminophen",
                "brand_name": "Tylenol",
                "monograph_key": "acetaminophen",
            },
            {
                "nbk_id": "NBK002",
                "drug_name": "Amoxicillin-Clavulanate",
                "excerpt": "Amoxicillin-clavulanate is a common cause of cholestatic DILI.",
                "synonyms": "Co-amoxi; Amoxicillin clavulanate",
                "ingredient": "Amoxicillin Clavulanate",
                "brand_name": "Co-amoxi",
                "monograph_key": "amoxicillin-clavulanate",
            },
            {
                "nbk_id": "NBK003",
                "drug_name": "Amoxicillin",
                "excerpt": "Amoxicillin monograph.",
                "synonyms": "",
                "ingredient": "Amoxicillin",
                "brand_name": "",
                "monograph_key": "amoxicillin",
            },
            {
                "nbk_id": "NBK004",
                "drug_name": "Clavulanate",
                "excerpt": "Clavulanate monograph.",
                "synonyms": "",
                "ingredient": "Clavulanate",
                "brand_name": "",
                "monograph_key": "clavulanate",
            },
            {
                "nbk_id": "NBK005",
                "drug_name": "Metformin",
                "excerpt": "Metformin monograph.",
                "synonyms": "SharedAlias",
                "ingredient": "Metformin",
                "brand_name": "",
                "monograph_key": "metformin",
            },
            {
                "nbk_id": "NBK006",
                "drug_name": "Metformin XR",
                "excerpt": "Metformin XR monograph.",
                "synonyms": "SharedAlias",
                "ingredient": "Metformin",
                "brand_name": "",
                "monograph_key": "metformin-xr",
            },
        ]
    )


def _catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rxcui": "161",
                "term_type": "IN",
                "raw_name": "Acetaminophen",
                "name": "Acetaminophen",
                "brand_names": "Tylenol",
                "synonyms": '["Tylenol", "Paracetamol"]',
            },
            {
                "rxcui": "999",
                "term_type": "SBD",
                "raw_name": "amoxicillin clavulanate Oral Tablet",
                "name": "amoxicillin clavulanate",
                "brand_names": "Co-amoxi",
                "synonyms": '["Co-amoxi", "Amoxicillin Clavulanate"]',
            },
        ]
    )


def _resolver() -> DrugResolutionService:
    return DrugResolutionService(
        LiverToxMatcher(_livertox_frame(), drugs_catalog_df=_catalog_frame())
    )


def _resolve_one(name: str) -> dict:
    resolved = _resolver().resolve(PatientDrugs(entries=[DrugEntry(name=name)]))
    assert resolved
    return next(iter(resolved.values()))


def test_exact_generic_drug_gets_resolution_decision() -> None:
    payload = _resolve_one("Acetaminophen")

    assert payload["decision_status"] == "accepted_rxnav_validated"
    assert payload["accepted_rxnav_rxcui"] == "161"
    assert payload["accepted_livertox_nbk_id"] == "NBK001"
    assert payload["resolution_decision"]["accepted_livertox_name"] == "Acetaminophen"


def test_brand_alias_maps_to_rxcui_and_livertox_monograph() -> None:
    payload = _resolve_one("Tylenol")

    assert payload["decision_status"] == "accepted_rxnav_validated"
    assert payload["rxnav_validation_status"] in {"brand_to_rxcui", "alias_to_rxcui"}
    assert payload["accepted_livertox_name"] == "Acetaminophen"


def test_combination_product_preserves_parent_and_components() -> None:
    resolved = _resolver().resolve(
        PatientDrugs(entries=[DrugEntry(name="Amoxicillin-Clavulanate")])
    )

    parent = resolved["amoxicillin clavulanate"]
    assert parent["decision_status"] == "accepted_rxnav_validated"
    assert parent["accepted_livertox_name"] == "Amoxicillin-Clavulanate"
    assert set(parent["regimen_components"]) == {"amoxicillin", "clavulanate"}
    assert any(
        item["accepted_livertox_name"] == "Amoxicillin"
        for item in resolved.values()
    )


def test_ambiguous_alias_requires_review() -> None:
    payload = _resolve_one("SharedAlias")

    assert payload["decision_status"] == "ambiguous_requires_review"
    assert payload["requires_human_review"] is True
    assert payload["accepted_livertox_name"] is None


def test_broad_category_is_not_concrete_drug_match() -> None:
    payload = _resolve_one("vitamins")

    assert payload["decision_status"] in {"ambiguous_requires_review", "missing_livertox"}
    assert payload["accepted_rxnav_rxcui"] is None
    assert payload["accepted_livertox_name"] is None


def test_false_positive_lab_text_is_rejected_before_matching() -> None:
    payload = _resolve_one("ALT")

    assert payload["decision_status"] == "rejected_false_positive"
    assert payload["missing_livertox"] is True
