from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from domain.clinical import (
    DrugEntry,
    DrugIdentityProposal,
    DrugIdentityProposalBatch,
    PatientDrugs,
)
from services.clinical.matches_core import LiverToxMatcher
from services.clinical.preparation import ClinicalKnowledgePreparation
from repository_fixtures import build_repository_graph


###############################################################################
def build_preparation() -> ClinicalKnowledgePreparation:
    graph = build_repository_graph()
    return ClinicalKnowledgePreparation(
        knowledge_repository=graph.knowledge_repository,
        drug_catalog_repository=graph.drug_catalog_repository,
    )


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
    preparation = build_preparation()
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
    preparation = build_preparation()
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
    preparation = build_preparation()
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


###############################################################################
def test_catalog_missing_medication_label_reaches_resolution_policy() -> None:
    preparation = build_preparation()
    preparation.livertox_matcher = LiverToxMatcher(
        pd.DataFrame(
            [
                {
                    "nbk_id": "NBK-KNOWN",
                    "drug_name": "Known Drug",
                    "excerpt": "Known excerpt.",
                    "synonyms": "",
                    "ingredient": "Known Drug",
                    "brand_name": "",
                }
            ]
        )
    )

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name="OpaqueBrand", source="therapy")]),
            clinical_context="",
            pattern_score=None,
        )
    )

    assert prepared is not None
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["raw_mentions"] == ["OpaqueBrand"]
    assert payload["decision_status"] == "missing_livertox"
    assert payload["resolution_decision"]


###############################################################################
def test_two_edit_international_spelling_resolves_uniquely() -> None:
    preparation = build_preparation()
    preparation.livertox_matcher = LiverToxMatcher(
        pd.DataFrame(
            [
                {
                    "nbk_id": "NBK-LOOP",
                    "drug_name": "Loop Diuretics",
                    "excerpt": "Loop diuretic excerpt.",
                    "synonyms": "",
                    "ingredient": "Torsemide",
                    "brand_name": "",
                }
            ]
        )
    )

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name="Torasemid", source="therapy")]),
            clinical_context="",
            pattern_score=None,
        )
    )

    assert prepared is not None
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["accepted_livertox_name"] == "Loop Diuretics"
    assert payload["extracted_excerpts"] == ["Loop diuretic excerpt."]


###############################################################################
class IdentityClientStub:
    # -------------------------------------------------------------------------
    def __init__(self, batch: DrugIdentityProposalBatch) -> None:
        self.batch = batch
        self.calls = 0

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["schema"] is DrugIdentityProposalBatch
        self.calls += 1
        return self.batch


###############################################################################
def test_llm_combination_identity_candidates_remain_ambiguous() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK-VALERIAN",
                "drug_name": "Valerian",
                "excerpt": "Valerian excerpt.",
                "synonyms": "",
                "ingredient": "Valerian",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK-HOPS",
                "drug_name": "Hops",
                "excerpt": "Hops excerpt.",
                "synonyms": "",
                "ingredient": "Hops",
                "brand_name": "",
            },
        ]
    )
    client = IdentityClientStub(
        DrugIdentityProposalBatch(
            proposals=[
                DrugIdentityProposal(
                    original_mention="SleepBrand",
                    proposed_canonical_name=None,
                    ingredients=["Valerian", "Hops"],
                    confidence=0.86,
                    rationale="Known combination product ingredients.",
                )
            ]
        )
    )
    preparation = build_preparation()
    preparation.livertox_matcher = LiverToxMatcher(frame)

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name="SleepBrand", source="therapy")]),
            clinical_context="",
            pattern_score=None,
            identity_resolution_client=client,
            identity_resolution_model="test-model",
        )
    )

    assert prepared is not None
    assert client.calls == 1
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["decision_status"] == "ambiguous_requires_review"
    assert payload["requires_human_review"] is True
    assert set(payload["match_candidates"]) == {"Valerian", "Hops"}
    assert payload["accepted_livertox_name"] is None


###############################################################################
def test_single_llm_identity_is_accepted_only_after_local_validation() -> None:
    client = IdentityClientStub(
        DrugIdentityProposalBatch(
            proposals=[
                DrugIdentityProposal(
                    original_mention="InternationalName",
                    proposed_canonical_name=None,
                    alternate_names=["Acetaminophen"],
                    ingredients=[],
                    confidence=0.98,
                    rationale="International generic synonym.",
                )
            ]
        )
    )
    preparation = build_preparation()
    preparation.livertox_matcher = LiverToxMatcher(
        pd.DataFrame(
            [
                {
                    "nbk_id": "NBK-ACETAMINOPHEN",
                    "drug_name": "Acetaminophen",
                    "excerpt": "Acetaminophen excerpt.",
                    "synonyms": "",
                    "ingredient": "Acetaminophen",
                    "brand_name": "",
                }
            ]
        )
    )

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(
                entries=[DrugEntry(name="InternationalName", source="therapy")]
            ),
            clinical_context="",
            pattern_score=None,
            identity_resolution_client=client,
            identity_resolution_model="test-model",
        )
    )

    assert prepared is not None
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["accepted_livertox_name"] == "Acetaminophen"
    assert payload["missing_livertox"] is False
    assert payload["matched_livertox_row"]
    assert "InternationalName" in payload["raw_mentions"]
    assert payload["match_reason"] == "Exact normalized LiverTox primary name accepted"
    assert payload["match_notes"] == [
        "Exact normalized LiverTox primary name accepted",
        "identity proposed by configured LLM",
        "identity accepted only after unique local evidence resolution",
    ]
    assert payload["resolution_decision"]["reasons"] == payload["match_notes"]


###############################################################################
def test_unvalidated_llm_identity_remains_unresolved() -> None:
    client = IdentityClientStub(
        DrugIdentityProposalBatch(
            proposals=[
                DrugIdentityProposal(
                    original_mention="OpaqueBrand",
                    proposed_canonical_name="Invented Ingredient",
                    ingredients=[],
                    confidence=0.95,
                    rationale="Proposed identity.",
                )
            ]
        )
    )
    preparation = build_preparation()
    preparation.livertox_matcher = LiverToxMatcher(
        pd.DataFrame(
            [
                {
                    "nbk_id": "NBK-KNOWN",
                    "drug_name": "Known Drug",
                    "excerpt": "Known excerpt.",
                    "synonyms": "",
                    "ingredient": "Known Drug",
                    "brand_name": "",
                }
            ]
        )
    )

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name="OpaqueBrand", source="therapy")]),
            clinical_context="",
            pattern_score=None,
            identity_resolution_client=client,
            identity_resolution_model="test-model",
        )
    )

    assert prepared is not None
    payload = next(iter(prepared.resolved_drugs.values()))
    assert payload["decision_status"] == "missing_livertox"
    assert payload["accepted_livertox_name"] is None
    assert payload["identity_candidates"][0]["accepted_local_names"] == []
