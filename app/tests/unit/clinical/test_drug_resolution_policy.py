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
            {
                "nbk_id": "NBK007",
                "drug_name": "Abiraterone",
                "excerpt": "Abiraterone monograph.",
                "synonyms": "",
                "ingredient": "Abiraterone",
                "brand_name": "Zytiga",
                "monograph_key": "abiraterone",
            },
            {
                "nbk_id": "NBK008",
                "drug_name": "Corticosteroids",
                "excerpt": "Corticosteroids monograph.",
                "synonyms": "",
                "ingredient": "Prednisone; Prednisolone",
                "brand_name": "Deltasone",
                "monograph_key": "corticosteroids",
            },
            {
                "nbk_id": "NBK009",
                "drug_name": "Tamsulosin",
                "excerpt": "Tamsulosin monograph.",
                "synonyms": "",
                "ingredient": "Tamsulosin",
                "brand_name": "Flomax",
                "monograph_key": "tamsulosin",
            },
            {
                "nbk_id": "NBK010",
                "drug_name": "Vitamin D",
                "excerpt": "Vitamin D monograph.",
                "synonyms": "",
                "ingredient": "Vitamin D; Calcitriol",
                "brand_name": "Citrical plus D",
                "monograph_key": "vitamin-d",
            },
            {
                "nbk_id": "NBK011",
                "drug_name": "Leuprolide",
                "excerpt": "Leuprolide monograph.",
                "synonyms": "",
                "ingredient": "Leuprolide",
                "brand_name": "Eligard",
                "monograph_key": "leuprolide",
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
            {
                "rxcui": "1007",
                "term_type": "SBD",
                "raw_name": "Abirateron Sandoz",
                "name": "Abiraterone",
                "brand_names": "Abirateron Sandoz",
                "synonyms": '["Abirateron"]',
            },
            {
                "rxcui": "1008",
                "term_type": "SBD",
                "raw_name": "Prednison Spirig HC",
                "name": "Prednisone",
                "brand_names": "Prednison Spirig HC",
                "synonyms": '["Prednison"]',
            },
            {
                "rxcui": "1009",
                "term_type": "SBD",
                "raw_name": "Pradif T",
                "name": "Tamsulosin",
                "brand_names": "Pradif T",
                "synonyms": "[]",
            },
            {
                "rxcui": "1010",
                "term_type": "SBD",
                "raw_name": "Vi-De3 4500 IU/ml",
                "name": "Vitamin D",
                "brand_names": "Vi-De3",
                "synonyms": "[]",
            },
            {
                "rxcui": "1011",
                "term_type": "SBD",
                "raw_name": "Calcimagon D3",
                "name": "Vitamin D",
                "brand_names": "Calcimagon D3; Calcimagon",
                "synonyms": "[]",
            },
            {
                "rxcui": "1012",
                "term_type": "IN",
                "raw_name": "Leuprorelina",
                "name": "Leuprolide",
                "brand_names": "",
                "synonyms": '["Leuprorelina", "Leuprorelin"]',
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


def test_catalog_backed_product_labels_resolve_to_livertox_monographs_with_excerpts() -> None:
    expected = {
        "Abirateron Sandoz": "Abiraterone",
        "Prednison Spirig HC": "Corticosteroids",
        "Pradif T": "Tamsulosin",
        "Vi-De3 4'500 IU/ml": "Vitamin D",
        "Calcimagon D3": "Vitamin D",
        "leuprorelina": "Leuprolide",
    }

    for label, accepted_name in expected.items():
        payload = _resolve_one(label)

        assert payload["accepted_livertox_name"] == accepted_name
        assert payload["missing_livertox"] is False
        assert payload["extracted_excerpts"]


def test_opaque_product_label_is_not_guessed_without_source_backing() -> None:
    payload = _resolve_one("Unmapped Pharma Brand")

    assert payload["accepted_livertox_name"] is None
    assert payload["missing_livertox"] is True


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
