from __future__ import annotations

import asyncio

import pandas as pd
from domain.clinical import DrugEntry, PatientDrugs
from services.clinical.match_resolution import conservative_fuzzy_livertox_match
from services.clinical.matches_core import LiverToxMatcher
from services.clinical.preparation import ClinicalKnowledgePreparation

###############################################################################
def _build_livertox_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nbk_id": "NBK0001",
                "drug_name": "Acetaminophen",
                "excerpt": "Acetaminophen can cause dose-related liver injury.",
                "synonyms": "Paracetamol; Tylenol",
                "ingredient": "Acetaminophen",
                "brand_name": "Tylenol",
            },
            {
                "nbk_id": "NBK0002",
                "drug_name": "Omeprazole",
                "excerpt": "Omeprazole has rare liver injury reports.",
                "synonyms": "Losec",
                "ingredient": "Omeprazole",
                "brand_name": "Losec",
            },
        ]
    )

###############################################################################
def test_conservative_fuzzy_livertox_match_high_threshold() -> None:
    assert (
        conservative_fuzzy_livertox_match(
            ["acetaminophenn"],
            ["Acetaminophen", "Omeprazole"],
        )
        == "Acetaminophen"
    )
    assert (
        conservative_fuzzy_livertox_match(
            ["zzzzz"],
            ["Acetaminophen", "Omeprazole"],
        )
        is None
    )

###############################################################################
def test_prepare_inputs_resolves_direct_livertox_alias() -> None:
    prep = ClinicalKnowledgePreparation()
    prep.livertox_matcher = LiverToxMatcher(_build_livertox_df())
    prepared = asyncio.run(
        prep.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name="Tylenol")]),
            clinical_context="",
            pattern_score=None,
        )
    )
    assert prepared is not None
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["accepted_livertox_name"] == "Acetaminophen"
    assert payload["decision_status"] == "accepted_livertox_without_rxnav"
