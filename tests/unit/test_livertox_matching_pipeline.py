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
            {
                "nbk_id": "NBK0004",
                "drug_name": "Naproxen",
                "excerpt": "Naproxen has rare liver injury reports.",
                "synonyms": "",
                "ingredient": "Naproxen",
                "brand_name": "",
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


def test_small_typo_resolves_to_unique_authoritative_match() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["Acetaminophenn"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Acetaminophen"
    assert result.reason == "spelling_correction"


def test_one_sided_name_fragment_is_not_joined_to_evidence() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["meprazole"])[0]

    assert result.status in {"missing", "ambiguous"}
    assert result.matched_name is None


def test_small_typo_stays_ambiguous_when_multiple_authoritative_candidates_exist() -> (
    None
):
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0801",
                "drug_name": "Metformin",
                "excerpt": "Metformin excerpt.",
                "synonyms": "",
                "ingredient": "Metformin",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0802",
                "drug_name": "Metforman",
                "excerpt": "Metforman excerpt.",
                "synonyms": "",
                "ingredient": "Metforman",
                "brand_name": "",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)

    result = matcher.match_drug_names(["Metformon"])[0]

    assert result.status == "ambiguous"
    assert result.reason == "ambiguous_spelling_correction"
    assert result.candidate_names == ["Metforman", "Metformin"]


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
    unresolved = mapping[1]
    missing = mapping[2]

    assert matched["missing_livertox"] is False
    assert matched["ambiguous_match"] is False
    assert matched["extracted_excerpts"]

    assert unresolved["match_status"] in {"ambiguous_match", "missing_match"}
    assert unresolved["missing_livertox"] is True
    assert unresolved["extracted_excerpts"] == []

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
    assert result.nbk_id is None


def test_repeated_nbk_ids_are_not_collapsed_across_monographs() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBKSHARED",
                "drug_name": "Alphaquine",
                "excerpt": "Alphaquine monograph.",
                "synonyms": "SharedBrand",
                "ingredient": "Alphaquine",
                "brand_name": "",
            },
            {
                "nbk_id": "NBKSHARED",
                "drug_name": "Betazole",
                "excerpt": "Betazole monograph.",
                "synonyms": "SharedBrand",
                "ingredient": "Betazole",
                "brand_name": "",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)

    exact = matcher.match_drug_names(["Alphaquine"])[0]
    shared_alias = matcher.match_drug_names(["SharedBrand"])[0]

    assert exact.status == "matched"
    assert exact.matched_name == "Alphaquine"
    assert exact.nbk_id == "NBKSHARED"
    assert shared_alias.status == "ambiguous"
    assert shared_alias.candidate_names == ["Alphaquine", "Betazole"]


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
    assert (
        normalize_drug_query_name("Levetiracetam Desitin 500 mg cpr") == "levetiracetam"
    )
    assert normalize_drug_query_name("Amlodipin axapharm cpr 5 mg") == "amlodipin"
    assert normalize_drug_query_name("Acido folico Streuli 5 mg cpr") == "acido folico"
    assert normalize_drug_query_name("Pantozol 20 mg cpr") == "pantoprazole"
    assert normalize_drug_query_name("Levetiracetam dal 27.08.2024") == "levetiracetam"
    assert normalize_drug_query_name("Nozinan dal 11.09.2024") == "levomepromazine"
    assert normalize_drug_query_name("Morfina gtt 5 3/die") == "morfina"


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
    assert normalize_drug_query_name("Bactrim") == "trimethoprim sulfamethoxazole"
    assert (
        normalize_drug_query_name("amoxicillin/clavulanate")
        == "amoxicillin clavulanate"
    )


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
    assert (
        not result.rejected_candidate_names
        or "Trimethoprim-Sulfamethoxazole" not in result.rejected_candidate_names
    )


def test_matcher_handles_source_backed_spelling_aliases() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0201",
                "drug_name": "Metformin",
                "excerpt": "Metformin has rare hepatotoxicity reports.",
                "synonyms": "Metformina",
                "ingredient": "Metformin",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0202",
                "drug_name": "Quetiapine",
                "excerpt": "Quetiapine excerpt.",
                "synonyms": "Quetiapina",
                "ingredient": "Quetiapine",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0203",
                "drug_name": "Fluvastatin",
                "excerpt": "Fluvastatin excerpt.",
                "synonyms": "Fluvastatina",
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
    assert metformina.reason in {"exact_alias", "normalized_exact"}
    assert quetiapina.status == "matched"
    assert quetiapina.matched_name == "Quetiapine"
    assert quetiapina.reason in {"exact_alias", "normalized_exact"}
    assert fluvastatina.status == "matched"
    assert fluvastatina.matched_name == "Fluvastatin"
    assert fluvastatina.reason in {"exact_alias", "normalized_exact"}
    assert co_amoxi.status == "matched"
    assert co_amoxi.matched_name == "Amoxicillin clavulanate"
    assert co_amoxi.reason in {
        "exact_canonical",
        "exact_alias_ranked",
        "exact_alias",
        "normalized_exact",
    }


def test_matcher_accepts_small_authoritative_name_misspellings() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0401",
                "drug_name": "Amlodipine",
                "excerpt": "Amlodipine excerpt.",
                "synonyms": "",
                "ingredient": "Amlodipine",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0402",
                "drug_name": "Atorvastatin",
                "excerpt": "Atorvastatin excerpt.",
                "synonyms": "",
                "ingredient": "Atorvastatin",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0403",
                "drug_name": "Esomeprazole",
                "excerpt": "Esomeprazole excerpt.",
                "synonyms": "",
                "ingredient": "Esomeprazole",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0404",
                "drug_name": "Naproxen",
                "excerpt": "Naproxen excerpt.",
                "synonyms": "",
                "ingredient": "Naproxen",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0405",
                "drug_name": "Morphine",
                "excerpt": "Morphine excerpt.",
                "synonyms": "",
                "ingredient": "Morphine",
                "brand_name": "",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)

    amlodipina = matcher.match_drug_names(["Amlodipina"])[0]
    atorvastatina = matcher.match_drug_names(["Atorvastatina"])[0]
    esomeprazolo = matcher.match_drug_names(["Esomeprazolox"])[0]
    morfina = matcher.match_drug_names(["Morfina"])[0]

    assert amlodipina.status == "matched"
    assert amlodipina.matched_name == "Amlodipine"
    assert atorvastatina.status == "matched"
    assert atorvastatina.matched_name == "Atorvastatin"
    assert esomeprazolo.status == "matched"
    assert esomeprazolo.matched_name == "Esomeprazole"
    assert morfina.status == "missing"


def test_matcher_keeps_unsafe_multilingual_fallbacks_unresolved() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0501",
                "drug_name": "Naproxen",
                "excerpt": "Naproxen excerpt.",
                "synonyms": "",
                "ingredient": "Naproxen",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0502",
                "drug_name": "Folic Acid",
                "excerpt": "Folic acid excerpt.",
                "synonyms": "vitamin; basal; schema interno",
                "ingredient": "Folic Acid",
                "brand_name": "",
            },
        ]
    )
    catalog = pd.DataFrame(
        [
            {
                "rxcui": "1",
                "term_type": "IN",
                "raw_name": "Folic Acid",
                "name": "Folic Acid",
                "synonyms": "vitamin; basal; schema interno",
                "brand_names": "",
            }
        ]
    )
    matcher = LiverToxMatcher(frame, drugs_catalog_df=catalog)

    esomeprazolo = matcher.match_drug_names(["Esomeprazolo"])[0]
    insulin = matcher.match_drug_names(["Insulina basal-bolus secondo schema interno"])[
        0
    ]

    assert esomeprazolo.status in {"missing", "ambiguous"}
    assert esomeprazolo.matched_name != "Naproxen"
    assert insulin.status in {"missing", "ambiguous"}
    assert insulin.matched_name != "Folic Acid"


def test_known_italian_drug_aliases_normalize_before_matching() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0701",
                "drug_name": "Esomeprazole",
                "excerpt": "Esomeprazole excerpt.",
                "synonyms": "Esomeprazolo",
                "ingredient": "Esomeprazole",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0702",
                "drug_name": "Bromelain",
                "excerpt": "Bromelain excerpt.",
                "synonyms": "Bromelina",
                "ingredient": "Bromelain",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0703",
                "drug_name": "Sulfamethoxazole Trimethoprim",
                "excerpt": "Cotrimoxazole excerpt.",
                "synonyms": "Cotrimossazolo",
                "ingredient": "Sulfamethoxazole Trimethoprim",
                "brand_name": "",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)

    results = matcher.match_drug_names(["Esomeprazolo", "Bromelina", "Cotrimossazolo"])

    assert [item.status for item in results] == ["matched", "matched", "matched"]
    assert [item.matched_name for item in results] == [
        "Esomeprazole",
        "Bromelain",
        "Sulfamethoxazole Trimethoprim",
    ]


def test_formulation_words_are_removed_from_livertox_query() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0704",
                "drug_name": "Boswellia Serrata",
                "excerpt": "Boswellia excerpt.",
                "synonyms": "",
                "ingredient": "Boswellia Serrata",
                "brand_name": "",
            }
        ]
    )
    matcher = LiverToxMatcher(frame)

    result = matcher.match_drug_names(["Boswellia serrata estratto secco"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Boswellia Serrata"


def test_matcher_prefers_full_latin_script_combination_before_components() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0601",
                "drug_name": "Amoxicillin clavulanate",
                "excerpt": "Combination amoxicillin clavulanate excerpt.",
                "synonyms": "Amoxicillina acido clavulanico",
                "ingredient": "Amoxicillin clavulanate",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0602",
                "drug_name": "Piperacillin Tazobactam",
                "excerpt": "Piperacillin tazobactam excerpt.",
                "synonyms": "Piperacillina tazobactam",
                "ingredient": "Piperacillin Tazobactam",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0603",
                "drug_name": "Piperacillin",
                "excerpt": "Piperacillin component excerpt.",
                "synonyms": "",
                "ingredient": "Piperacillin",
                "brand_name": "",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)

    amoxicillin = matcher.match_drug_names(["Amoxicillina acido clavulanico"])[0]
    piperacillin = matcher.match_drug_names(["Piperacillina tazobactam"])[0]

    assert amoxicillin.status == "matched"
    assert amoxicillin.matched_name == "Amoxicillin clavulanate"
    assert piperacillin.status == "matched"
    assert piperacillin.matched_name == "Piperacillin Tazobactam"


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
    mapping = matcher.build_drugs_to_excerpt_mapping(
        queries, matcher.match_drug_names(queries)
    )

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

    assert "encorafenib binimetinib" in canonical_names
    assert "dabrafenib trametinib" in canonical_names
    assert "encorafenib" in canonical_names
    assert "binimetinib" in canonical_names
    assert "dabrafenib" in canonical_names
    assert "trametinib" in canonical_names
    for candidate in candidates:
        if candidate["canonical_name"] in {"encorafenib", "binimetinib"}:
            assert "binimetinib|encorafenib" in candidate["regimen_group_ids"]
        if candidate["canonical_name"] in {"dabrafenib", "trametinib"}:
            assert "dabrafenib|trametinib" in candidate["regimen_group_ids"]
