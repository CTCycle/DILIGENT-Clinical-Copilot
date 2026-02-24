from __future__ import annotations

import asyncio

import pandas as pd

from DILIGENT.server.entities.clinical import DrugEntry, PatientDrugs
from DILIGENT.server.services.clinical.matches import LiverToxMatcher
from DILIGENT.server.services.clinical.preparation import ClinicalKnowledgePreparation
from DILIGENT.server.services.text.normalization import normalize_drug_query_name


def build_livertox_df() -> pd.DataFrame:
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
            {
                "nbk_id": "NBK0003",
                "drug_name": "Esomeprazole",
                "excerpt": "Esomeprazole has uncommon liver injury reports.",
                "synonyms": "Nexium",
                "ingredient": "Esomeprazole",
                "brand_name": "Nexium",
            },
        ]
    )


def test_exact_canonical_match_works() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["Acetaminophen 500 mg tablets"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Acetaminophen"
    assert result.reason == "exact_canonical"


def test_alias_resolution_works() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["Tylenol"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Acetaminophen"
    assert result.reason == "exact_alias"


def test_typo_resolves_via_fuzzy_above_threshold() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["Acetaminophenn"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Acetaminophen"
    assert result.reason == "fuzzy"


def test_multiple_fuzzy_candidates_are_marked_ambiguous() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["meprazole"])[0]

    assert result.status == "ambiguous"
    assert "Omeprazole" in result.candidate_names
    assert "Esomeprazole" in result.candidate_names


def test_no_match_is_safe_and_explicit() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["zzzzdrug"])[0]

    assert result.status == "missing"
    assert result.matched_name is None
    assert result.confidence is None


def test_excerpt_attached_only_for_valid_match_confidence() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    queries = ["Acetaminophen", "meprazole", "zzzzdrug"]
    matches = matcher.match_drug_names(queries)
    mapping = matcher.build_drugs_to_excerpt_mapping(queries, matches)

    matched = mapping[0]
    ambiguous = mapping[1]
    missing = mapping[2]

    assert matched["missing_livertox"] is False
    assert matched["ambiguous_match"] is False
    assert matched["extracted_excerpts"]

    assert ambiguous["ambiguous_match"] is True
    assert ambiguous["missing_livertox"] is True
    assert ambiguous["extracted_excerpts"] == []

    assert missing["missing_livertox"] is True
    assert missing["extracted_excerpts"] == []


def test_duplicate_drugs_from_sources_are_merged_by_canonical_name() -> None:
    preparation = ClinicalKnowledgePreparation()
    drugs = PatientDrugs(
        entries=[
            DrugEntry(name="Acetaminophen 500 mg", source="therapy"),
            DrugEntry(name="acetaminophen", source="anamnesis"),
        ]
    )

    candidates = preparation.build_drug_candidates(drugs)

    assert len(candidates) == 1
    assert candidates[0]["canonical_name"] == "acetaminophen"
    assert candidates[0]["origins"] == ["therapy", "anamnesis"]


def test_prepare_inputs_handles_empty_drugs_without_crashing() -> None:
    preparation = ClinicalKnowledgePreparation()

    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[]),
            clinical_context="",
            pattern_score=None,
        )
    )

    assert prepared is None


def test_matcher_keeps_matching_when_nbk_id_is_missing() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": None,
                "drug_name": "Diazepam",
                "excerpt": "Diazepam can cause rare cholestatic injury.",
                "synonyms": "Valium",
                "ingredient": "Diazepam",
                "brand_name": "Valium",
            }
        ]
    )
    matcher = LiverToxMatcher(frame)

    result = matcher.match_drug_names(["Valium"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Diazepam"
    assert result.nbk_id == "synthetic::diazepam"


def test_related_excerpt_is_used_when_matched_monograph_excerpt_is_missing() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0010",
                "drug_name": "ursodiol",
                "excerpt": None,
                "synonyms": "De-Ursil",
                "ingredient": "ursodiol",
                "brand_name": "De-Ursil",
            },
            {
                "nbk_id": "NBK0011",
                "drug_name": "Ursodiol (Ursodeoxycholic Acid)",
                "excerpt": "Ursodiol is generally safe and is not linked to severe DILI.",
                "synonyms": "",
                "ingredient": "ursodiol",
                "brand_name": "",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)

    matches = matcher.match_drug_names(["De-Ursil"])
    mapping = matcher.build_drugs_to_excerpt_mapping(["De-Ursil"], matches)

    entry = mapping[0]
    assert entry["match_status"] == "matched"
    assert entry["missing_livertox"] is False
    assert entry["extracted_excerpts"]
    assert "ursodiol is generally safe" in entry["extracted_excerpts"][0].lower()
    assert "fallback_excerpt_from_related_monograph" in entry["match_notes"]


def test_query_normalization_handles_brands_and_manufacturers() -> None:
    assert normalize_drug_query_name("Levetiracetam Desitin 500 mg cpr") == "levetiracetam"
    assert normalize_drug_query_name("Amlodipin axapharm cpr 5 mg") == "amlodipine"
    assert normalize_drug_query_name("Acido folico Streuli 5 mg cpr") == "folic acid"
    assert normalize_drug_query_name("Pantozol 20 mg cpr") == "pantoprazole"
    assert normalize_drug_query_name("Levetiracetam dal 27.08.2024") == "levetiracetam"
    assert normalize_drug_query_name("Nozinan dal 11.09.2024") == "levomepromazine"
    assert normalize_drug_query_name("Morfina gtt 5 3/die") == "morphine"
