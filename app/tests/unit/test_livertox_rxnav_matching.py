from __future__ import annotations

import asyncio

import pandas as pd
from domain.clinical.entities import DrugEntry
from services.clinical.match_resolution import conservative_fuzzy_livertox_match
from services.clinical.preparation import ClinicalKnowledgePreparation


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


def test_conservative_fuzzy_livertox_match_high_threshold() -> None:
    assert conservative_fuzzy_livertox_match(
        ["acetaminophenn"],
        ["Acetaminophen", "Omeprazole"],
    ) == "Acetaminophen"
    assert conservative_fuzzy_livertox_match(
        ["zzzzz"],
        ["Acetaminophen", "Omeprazole"],
    ) is None


def test_resolve_livertox_match_for_drug_direct() -> None:
    prep = ClinicalKnowledgePreparation()
    from services.clinical.matches_core import LiverToxMatcher

    prep.livertox_matcher = LiverToxMatcher(_build_livertox_df())
    result = asyncio.run(
        prep.resolve_livertox_match_for_drug(DrugEntry(name="Tylenol"))
    )
    assert result is not None
    assert result.matched_livertox_name == "Acetaminophen"
    assert result.match_strategy == "direct_livertox"
