from __future__ import annotations

import asyncio

import pandas as pd
from domain.clinical import DrugEntry, PatientDrugs
from services.clinical.drug_identity import DrugIdentityResolver
from services.clinical.drug_resolution import DrugResolutionService
from services.clinical.match_resolution import conservative_fuzzy_livertox_match
from services.clinical.matches_core import LiverToxMatcher
from services.clinical.preparation import ClinicalKnowledgePreparation
from services.text.normalization import normalize_drug_query_name


###############################################################################
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


###############################################################################
def test_exact_canonical_match_works() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["Acetaminophen 500 mg tablets"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Acetaminophen"
    assert result.reason == "exact_canonical"


###############################################################################
def test_alias_resolution_works() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["Tylenol"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Acetaminophen"
    assert result.reason == "exact_alias"


###############################################################################
def test_small_typo_resolves_to_unique_authoritative_match() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["Acetaminophenn"])[0]

    assert result.status == "matched"
    assert result.matched_name == "Acetaminophen"
    assert result.reason == "spelling_correction"


###############################################################################
def test_one_sided_name_fragment_is_not_joined_to_evidence() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["meprazole"])[0]

    assert result.status in {"missing", "ambiguous"}
    assert result.matched_name is None


###############################################################################
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


###############################################################################
def test_no_match_is_safe_and_explicit() -> None:
    matcher = LiverToxMatcher(build_livertox_df())
    result = matcher.match_drug_names(["zzzzdrug"])[0]

    assert result.status == "missing"
    assert result.matched_name is None
    assert result.confidence is None


###############################################################################
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


###############################################################################
def test_duplicate_drugs_from_sources_are_merged_by_canonical_name() -> None:
    drugs = PatientDrugs(
        entries=[
            DrugEntry(name="Acetaminophen 500 mg", source="therapy"),
            DrugEntry(name="acetaminophen", source="anamnesis"),
        ]
    )

    resolved = DrugResolutionService(LiverToxMatcher(build_livertox_df())).resolve(
        drugs
    )

    assert len(resolved) == 1
    payload = next(iter(resolved.values()))
    assert payload["accepted_livertox_name"] == "Acetaminophen"
    assert payload["origins"] == ["therapy", "anamnesis"]


###############################################################################
def test_preparation_resolves_catalog_identity_candidates_before_livertox_lookup() -> (
    None
):
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0901",
                "drug_name": "Prednisone",
                "excerpt": "Prednisone excerpt.",
                "synonyms": "",
                "ingredient": "Prednisone",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0902",
                "drug_name": "Abiraterone",
                "excerpt": "Abiraterone excerpt.",
                "synonyms": "",
                "ingredient": "Abiraterone",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0903",
                "drug_name": "Tamsulosin",
                "excerpt": "Tamsulosin excerpt.",
                "synonyms": "",
                "ingredient": "Tamsulosin",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0904",
                "drug_name": "Leuprolide",
                "excerpt": "Leuprolide excerpt.",
                "synonyms": "Leuprorelin",
                "ingredient": "Leuprolide",
                "brand_name": "",
            },
        ]
    )
    catalog = pd.DataFrame(
        [
            {
                "rxcui": "1",
                "term_type": "SCD",
                "raw_name": "prednisone 1 MG Oral Tablet",
                "name": "prednisone oral",
                "brand_names": "",
                "synonyms": '["Prednisone", "Prednisone Oral", "Oral"]',
            },
            {
                "rxcui": "2",
                "term_type": "SCD",
                "raw_name": "abiraterone acetate 250 MG Oral Tablet",
                "name": "abiraterone acetate oral",
                "brand_names": "Zytiga",
                "synonyms": '["Abiraterone Acetate", "Zytiga", "Oral"]',
            },
            {
                "rxcui": "3",
                "term_type": "SBD",
                "raw_name": "tamsulosin hydrochloride 0.4 MG Oral Capsule",
                "name": "tamsulosin hydrochloride oral",
                "brand_names": "Pradif T",
                "synonyms": '["Tamsulosin", "Pradif T", "Oral"]',
            },
            {
                "rxcui": "4",
                "term_type": "SCD",
                "raw_name": "leuprolide acetate Prefilled Syringe",
                "name": "leuprolide acetate prefilled syringe",
                "brand_names": "Lupron",
                "synonyms": '["Leuprolide Acetate", "Leuprorelin", "Prefilled Syringe"]',
            },
        ]
    )
    drugs = PatientDrugs(
        entries=[
            DrugEntry(name="Prednisone TrialCo", source="therapy"),
            DrugEntry(name="Abiraterone TrialPharm", source="therapy"),
            DrugEntry(name="Pradif T", source="therapy"),
            DrugEntry(name="leuprorelina", source="therapy"),
        ]
    )

    resolved = DrugResolutionService(
        LiverToxMatcher(frame, drugs_catalog_df=catalog)
    ).resolve(drugs)
    canonical_names = {
        payload["accepted_livertox_name"] for payload in resolved.values()
    }

    assert {"Prednisone", "Abiraterone", "Tamsulosin", "Leuprolide"} <= canonical_names


###############################################################################
def test_component_splitting_keeps_units_and_drops_noise() -> None:
    assert DrugIdentityResolver.split_components("Trialmed 4500 IU/ml") == [
        "Trialmed 4500 IU per ml"
    ]
    assert DrugIdentityResolver.split_components("Trialmed 500/800") == [
        "Trialmed 500/800"
    ]
    assert DrugIdentityResolver.split_components("Alphamed + Betamed") == [
        "Alphamed",
        "Betamed",
    ]
    assert DrugIdentityResolver.split_components("Alphamed-Betamed") == [
        "Alphamed",
        "Betamed",
    ]


###############################################################################
def test_catalog_alias_quality_rejects_formulation_pollution() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK1001",
                "drug_name": "Diazepam",
                "excerpt": "Diazepam excerpt.",
                "synonyms": "",
                "ingredient": "Diazepam",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK1002",
                "drug_name": "Cholecalciferol",
                "excerpt": "Cholecalciferol excerpt.",
                "synonyms": "",
                "ingredient": "Cholecalciferol",
                "brand_name": "",
            },
        ]
    )
    catalog = pd.DataFrame(
        [
            {
                "rxcui": "10",
                "term_type": "SCD",
                "raw_name": "diazepam 5 MG Oral Tablet",
                "name": "diazepam oral",
                "brand_names": "",
                "synonyms": '["Oral", "Tablet", "Diazepam Oral"]',
            }
        ]
    )
    matcher = LiverToxMatcher(frame, drugs_catalog_df=catalog)

    cholecalciferol = matcher.match_drug_names(["Cholecalciferol"])[0]
    calcium = matcher.match_drug_names(["Calcium Carbonate"])[0]

    assert cholecalciferol.status == "matched"
    assert cholecalciferol.matched_name == "Cholecalciferol"
    assert calcium.matched_name != "Diazepam"


###############################################################################
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


###############################################################################
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


###############################################################################
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


###############################################################################
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
    assert any(
        note in entry["match_notes"]
        for note in (
            "fallback_excerpt_from_related_monograph",
            "deterministic_disambiguation_applied",
        )
    )


###############################################################################
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


###############################################################################
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


###############################################################################
def test_query_normalization_high_value_aliases_are_deterministic() -> None:
    assert normalize_drug_query_name("Co-amoxi 1g") == "amoxicillin clavulanate"
    assert normalize_drug_query_name("Bactrim") == "trimethoprim sulfamethoxazole"
    assert (
        normalize_drug_query_name("amoxicillin/clavulanate")
        == "amoxicillin clavulanate"
    )
    assert normalize_drug_query_name("Paspertin") == "metoclopramide"
    assert normalize_drug_query_name("Buscopan") == "scopolamine"
    assert normalize_drug_query_name("Imodium lingual") == "loperamide"
    assert normalize_drug_query_name("Dafalgan") == "acetaminophen"
    assert normalize_drug_query_name("Rivotril") == "clonazepam"
    assert (
        normalize_drug_query_name("nozione di terapia antibiotica con co amoxicillina")
        == "amoxicillin clavulanate"
    )
    assert normalize_drug_query_name("dal") == ""
    assert normalize_drug_query_name("entrambi e il") == ""
    assert normalize_drug_query_name("rialzo a") == ""


###############################################################################
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


###############################################################################
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


###############################################################################
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


###############################################################################
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


###############################################################################
def test_identity_resolution_does_not_manufacture_prefix_or_stem_queries() -> None:
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0503",
                "drug_name": "Diazepam",
                "excerpt": "Diazepam excerpt.",
                "synonyms": "thyroid; levothyroxine",
                "ingredient": "Diazepam",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0504",
                "drug_name": "Morphine",
                "excerpt": "Morphine excerpt.",
                "synonyms": "morphin",
                "ingredient": "Morphine",
                "brand_name": "",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)
    resolver = DrugIdentityResolver(matcher)

    levothyroxine = resolver.resolve("levothyroxine sodium")
    morfina = resolver.resolve("morfina")

    assert all(
        candidate.canonical_candidate != "levothyroxine" for candidate in levothyroxine
    )
    assert all(candidate.canonical_candidate != "morphin" for candidate in morfina)
    assert all(
        candidate.canonical_candidate != "diazepam" for candidate in levothyroxine
    )


###############################################################################
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


###############################################################################
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


###############################################################################
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


###############################################################################
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


###############################################################################
def test_preparation_expands_regimen_into_multiple_components() -> None:
    drugs = PatientDrugs(
        entries=[
            DrugEntry(name="Encorafenib + Binimetinib", source="therapy"),
            DrugEntry(name="Dabrafenib + Trametinib", source="therapy"),
        ]
    )

    resolved = DrugResolutionService(LiverToxMatcher(build_livertox_df())).resolve(
        drugs
    )
    canonical_names = {payload["canonical_name"] for payload in resolved.values()}

    assert "encorafenib binimetinib" in canonical_names
    assert "dabrafenib trametinib" in canonical_names
    assert "encorafenib" in canonical_names
    assert "binimetinib" in canonical_names
    assert "dabrafenib" in canonical_names
    assert "trametinib" in canonical_names
    for payload in resolved.values():
        if payload["canonical_name"] in {"encorafenib", "binimetinib"}:
            assert "binimetinib|encorafenib" in payload["regimen_group_ids"]
        if payload["canonical_name"] in {"dabrafenib", "trametinib"}:
            assert "dabrafenib|trametinib" in payload["regimen_group_ids"]


###############################################################################
def test_catalog_retry_resolves_ambiguous_match_via_catalog_alias() -> None:
    """When an alias matches multiple LiverTox records, the catalog-backed
    retry provides a more specific alias that resolves the ambiguity."""
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0501",
                "drug_name": "DrugA",
                "excerpt": "DrugA excerpt.",
                "synonyms": "CommonName",
                "ingredient": "DrugA",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0502",
                "drug_name": "DrugB",
                "excerpt": "DrugB excerpt.",
                "synonyms": "CommonName",
                "ingredient": "DrugB",
                "brand_name": "",
            },
        ]
    )
    catalog = pd.DataFrame(
        [
            {
                "rxcui": "500",
                "term_type": "SCD",
                "raw_name": "DrugA 500 MG Oral Tablet",
                "name": "DrugA oral",
                "brand_names": "",
                "synonyms": '["DrugA", "CommonName"]',
            },
        ]
    )
    matcher = LiverToxMatcher(frame, drugs_catalog_df=catalog)
    result = matcher.match_drug_names(["CommonName"])[0]

    assert result.status == "matched"
    assert result.matched_name == "DrugA"
    assert result.reason in {
        "exact_canonical",
        "exact_alias_ranked",
        "exact_alias",
        "normalized_exact_ranked",
    }


###############################################################################
def test_ambiguous_retry_preserves_original_when_catalog_does_not_help() -> None:
    """When catalog aliases still cannot resolve an ambiguous match,
    the original ambiguous result is preserved with all candidates."""
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0601",
                "drug_name": "DrugAlpha",
                "excerpt": "DrugAlpha excerpt.",
                "synonyms": "SharedName",
                "ingredient": "DrugAlpha",
                "brand_name": "",
            },
            {
                "nbk_id": "NBK0602",
                "drug_name": "DrugBeta",
                "excerpt": "DrugBeta excerpt.",
                "synonyms": "SharedName",
                "ingredient": "DrugBeta",
                "brand_name": "",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)
    result = matcher.match_drug_names(["SharedName"])[0]

    assert result.status == "ambiguous"
    assert "DrugAlpha" in result.candidate_names
    assert "DrugBeta" in result.candidate_names


###############################################################################
def _build_rxnav_livertox_df() -> pd.DataFrame:
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
    prep.livertox_matcher = LiverToxMatcher(_build_rxnav_livertox_df())
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


###############################################################################
def test_livertox_match_audit_flags_missing_ambiguous_and_low_confidence() -> None:
    preparation = ClinicalKnowledgePreparation()
    issues = preparation.build_match_audit_issues(
        {
            "drug-a": {
                "raw_mentions": ["Drug A"],
                "missing_livertox": True,
                "match_status": "missing_match",
                "rxnav_validated": False,
                "rxnav_rxcui": None,
            },
            "drug-b": {
                "raw_mentions": ["Drug B"],
                "ambiguous_match": True,
                "match_confidence": 0.4,
                "rxnav_validated": True,
                "rxnav_rxcui": "123",
            },
        }
    )

    codes = {issue.code for issue in issues}
    assert "livertox_match_missing" in codes
    assert "livertox_match_ambiguous" in codes
    assert "livertox_match_low_confidence" in codes
    assert "rxnav_alias_not_validated" in codes
