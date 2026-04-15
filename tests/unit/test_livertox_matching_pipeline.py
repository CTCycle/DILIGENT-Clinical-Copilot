from __future__ import annotations

import asyncio

import pandas as pd

from DILIGENT.server.domain.clinical import DrugEntry, PatientDrugs
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

    assert ambiguous["match_status"] == "ambiguous_match"
    assert ambiguous["ambiguous_match"] is True
    assert ambiguous["missing_livertox"] is True
    assert ambiguous["extracted_excerpts"] == []

    assert missing["match_status"] == "missing_match"
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
    assert entry["match_status"] == "matched_with_excerpt"
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


def test_mapping_prefers_excerpt_row_for_duplicate_normalized_drug() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0001",
                "drug_name": "Diazepam",
                "excerpt": None,
                "synonyms": "Valium",
                "ingredient": "Diazepam",
                "brand_name": "Valium",
                "include_in_livertox": False,
                "reference_count": 900,
            },
            {
                "nbk_id": "NBK0002",
                "drug_name": "diazepam",
                "excerpt": "Useful diazepam excerpt from preferred row.",
                "synonyms": "Valium",
                "ingredient": "Diazepam",
                "brand_name": "Valium",
                "include_in_livertox": False,
                "reference_count": 1,
            },
        ]
    )
    matcher = LiverToxMatcher(frame)

    record = matcher.data.records[0]
    match = matcher.lookup.create_match(
        record=record,
        confidence=1.0,
        reason="exact_canonical",
        notes=[],
    )
    mapping = matcher.build_drugs_to_excerpt_mapping(["Valium"], [match])

    entry = mapping[0]
    assert entry["match_status"] == "matched_with_excerpt"
    assert entry["missing_livertox"] is False
    assert entry["matched_livertox_row"] is not None
    assert entry["matched_livertox_row"]["nbk_id"] == "NBK0002"
    assert "useful diazepam excerpt" in entry["extracted_excerpts"][0].lower()


def test_query_normalization_high_value_aliases_are_deterministic() -> None:
    assert normalize_drug_query_name("Co-amoxi 1g") == "amoxicillin clavulanate"
    assert normalize_drug_query_name("Metformina 1000 mg") == "metformin"
    assert normalize_drug_query_name("Bactrim") == "trimethoprim sulfamethoxazole"
    assert normalize_drug_query_name("Quetiapina") == "quetiapine"
    assert normalize_drug_query_name("Fluvastatina") == "fluvastatin"
    assert normalize_drug_query_name("amoxicillin/clavulanate") == "amoxicillin clavulanate"


def test_matcher_prefers_combo_for_bactrim_brand_disambiguation() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0101",
                "drug_name": "Trimethoprim-Sulfamethoxazole",
                "excerpt": "Combination can cause cholestatic injury in rare cases.",
                "synonyms": "Bactrim",
                "ingredient": "Trimethoprim-Sulfamethoxazole",
                "brand_name": "Bactrim",
            },
            {
                "nbk_id": "NBK0102",
                "drug_name": "Trimethoprim",
                "excerpt": "Trimethoprim monotherapy excerpt.",
                "synonyms": "",
                "ingredient": "Trimethoprim",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0103",
                "drug_name": "Sulfamethoxazole",
                "excerpt": "Sulfamethoxazole monotherapy excerpt.",
                "synonyms": "",
                "ingredient": "Sulfamethoxazole",
                "brand_name": "",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)

    result = matcher.match_drug_names(["Bactrim"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Trimethoprim-Sulfamethoxazole"
    assert result.reason in {"exact_canonical", "exact_alias_ranked", "exact_alias"}
    assert not result.rejected_candidate_names or "Trimethoprim-Sulfamethoxazole" not in result.rejected_candidate_names


def test_matcher_handles_known_italian_aliases_without_fuzzy() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0201",
                "drug_name": "Metformin",
                "excerpt": "Metformin has rare hepatotoxicity reports.",
                "synonyms": "",
                "ingredient": "Metformin",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0202",
                "drug_name": "Quetiapine",
                "excerpt": "Quetiapine excerpt.",
                "synonyms": "",
                "ingredient": "Quetiapine",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0203",
                "drug_name": "Fluvastatin",
                "excerpt": "Fluvastatin excerpt.",
                "synonyms": "",
                "ingredient": "Fluvastatin",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0204",
                "drug_name": "Amoxicillin clavulanate",
                "excerpt": "Combination beta-lactam excerpt.",
                "synonyms": "Co-amoxi",
                "ingredient": "Amoxicillin clavulanate",
                "brand_name": "Co-amoxi",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)

    metformina = matcher.match_drug_names(["Metformina"])[0]
    quetiapina = matcher.match_drug_names(["Quetiapina"])[0]
    fluvastatina = matcher.match_drug_names(["Fluvastatina"])[0]
    co_amoxi = matcher.match_drug_names(["Co-amoxi"])[0]

    assert metformina.status == "matched"
    assert metformina.matched_name == "Metformin"
    assert metformina.reason != "fuzzy"
    assert quetiapina.status == "matched"
    assert quetiapina.matched_name == "Quetiapine"
    assert quetiapina.reason != "fuzzy"
    assert fluvastatina.status == "matched"
    assert fluvastatina.matched_name == "Fluvastatin"
    assert fluvastatina.reason != "fuzzy"
    assert co_amoxi.status == "matched"
    assert co_amoxi.matched_name == "Amoxicillin clavulanate"
    assert co_amoxi.reason != "fuzzy"


def test_mapping_classifies_matched_no_excerpt_separately_from_missing_match() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0301",
                "drug_name": "Piperacillin Tazobactam",
                "excerpt": None,
                "synonyms": "Piperacillin-Tazobactam",
                "ingredient": "Piperacillin Tazobactam",
                "brand_name": "",
            }
        ]
    )
    matcher = LiverToxMatcher(frame)

    queries = ["Piperacillin-Tazobactam", "UnknownDrugZZ"]
    mapping = matcher.build_drugs_to_excerpt_mapping(queries, matcher.match_drug_names(queries))

    no_excerpt = mapping[0]
    missing = mapping[1]
    assert no_excerpt["match_status"] == "matched_no_excerpt"
    assert no_excerpt["missing_livertox"] is True
    assert no_excerpt["chosen_candidate"] == "Piperacillin Tazobactam"
    assert missing["match_status"] == "missing_match"
    assert missing["missing_livertox"] is True


def test_preparation_expands_regimen_into_multiple_components() -> None:
    preparation = ClinicalKnowledgePreparation()
    drugs = PatientDrugs(
        entries=[
            DrugEntry(name="Encorafenib + Binimetinib", source="therapy"),
            DrugEntry(name="Dabrafenib + Trametinib", source="therapy"),
        ]
    )

    candidates = preparation.build_drug_candidates(drugs)
    canonical_names = {candidate["canonical_name"] for candidate in candidates}

    assert "encorafenib" in canonical_names
    assert "binimetinib" in canonical_names
    assert "dabrafenib" in canonical_names
    assert "trametinib" in canonical_names
    for candidate in candidates:
        if candidate["canonical_name"] in {"encorafenib", "binimetinib"}:
            assert "binimetinib|encorafenib" in candidate["regimen_group_ids"]
        if candidate["canonical_name"] in {"dabrafenib", "trametinib"}:
            assert "dabrafenib|trametinib" in candidate["regimen_group_ids"]
