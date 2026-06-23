from __future__ import annotations

import asyncio

import pandas as pd

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
