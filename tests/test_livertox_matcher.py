import asyncio
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DILIGENT.app.utils.services.pharma import LiverToxMatcher


def build_matcher() -> LiverToxMatcher:
    monographs = pd.DataFrame(
        [
            {
                "nbk_id": "NBK1",
                "drug_name": "Rivaroxaban",
                "excerpt": "Example excerpt",
                "synonyms": "Xarelto, Rivaroxaban tablets",
            },
            {
                "nbk_id": "NBK2",
                "drug_name": "Guselkumab",
                "excerpt": "Another excerpt",
                "synonyms": "In; For",
            },
            {
                "nbk_id": "NBK3",
                "drug_name": "Dabigatran",
                "excerpt": "Third excerpt",
                "synonyms": "Dabigatran Etexilate",
            },
        ]
    )
    master = pd.DataFrame(
        [
            {
                "ingredient": "Rivaroxaban",
                "brand_name": "Xarelto",
                "drug_name": "Rivaroxaban",
            },
            {
                "ingredient": "Dabigatran Etexilate",
                "brand_name": "Pradaxa",
                "drug_name": "Dabigatran Etexilate",
            },
        ]
    )
    return LiverToxMatcher(monographs, master)


def run_match(matcher: LiverToxMatcher, drugs: list[str]):
    return asyncio.run(matcher.match_drug_names(drugs))


def test_direct_monograph_match():
    matcher = build_matcher()
    matches = run_match(matcher, ["Rivaroxaban"])
    match = matches[0]
    assert match is not None
    assert match.nbk_id == "NBK1"
    assert match.reason == "monograph_name"


def test_master_list_brand_lookup():
    matcher = build_matcher()
    matches = run_match(matcher, ["Xarelto"])
    match = matches[0]
    assert match is not None
    assert match.nbk_id == "NBK1"
    assert match.reason == "brand_drug_name"


def test_synonym_lookup_from_delimited_string():
    matcher = build_matcher()
    matches = run_match(matcher, ["Rivaroxaban tablets"])
    match = matches[0]
    assert match is not None
    assert match.nbk_id == "NBK1"
    assert match.reason == "synonym_match"


def test_dictionary_synonym_lookup():
    monographs = pd.DataFrame(
        [
            {
                "nbk_id": "NBK_DICT",
                "drug_name": "Sample Drug",
                "excerpt": "Sample excerpt",
                "synonyms": '{"other": ["Sample Alias"]}',
            }
        ]
    )
    matcher = LiverToxMatcher(monographs)
    matches = run_match(matcher, ["Sample Alias"])
    match = matches[0]
    assert match is not None
    assert match.nbk_id == "NBK_DICT"
    assert match.reason == "synonym_match"


def test_master_list_drug_synonym_resolution():
    matcher = build_matcher()
    matches = run_match(matcher, ["Pradaxa"])
    match = matches[0]
    assert match is not None
    assert match.nbk_id == "NBK3"
    assert match.reason == "brand_drug_synonym"


def test_synonym_partial_and_stopword_filter():
    matcher = build_matcher()
    matches = run_match(matcher, ["Etexilate", "In"])
    partial_match, stopword_match = matches
    assert partial_match is not None
    assert partial_match.nbk_id == "NBK3"
    assert partial_match.reason == "partial_synonym"
    assert stopword_match is None


def test_fuzzy_typo_resolution():
    matcher = build_matcher()
    matches = run_match(matcher, ["Xaretlo"])
    match = matches[0]
    assert match is not None
    assert match.nbk_id == "NBK1"
    assert match.reason == "fuzzy_synonym"


def test_partial_match_with_duplicate_nbk_ids():
    monographs = pd.DataFrame(
        [
            {
                "nbk_id": "NBK_DUP",
                "drug_name": "Alpha Drug",
                "excerpt": "Alpha excerpt",
                "synonyms": "Alpha Tablet",
            },
            {
                "nbk_id": "NBK_DUP",
                "drug_name": "Beta Drug",
                "excerpt": "Beta excerpt",
                "synonyms": "Beta Tablet",
            },
        ]
    )
    matcher = LiverToxMatcher(monographs)
    matches = run_match(matcher, ["Alpha Tablet"])
    match = matches[0]
    assert match is not None
    assert match.matched_name == "Alpha Drug"


def test_build_patient_mapping_uses_monograph_rows():
    matcher = build_matcher()
    matches = run_match(matcher, ["Xarelto"])
    mapping = matcher.build_patient_mapping(["Xarelto"], matches)
    assert mapping[0]["matched_livertox_row"]["drug_name"] == "Rivaroxaban"
    assert mapping[0]["matched_livertox_row"]["excerpt"] == "Example excerpt"


def test_build_patient_mapping_with_duplicate_nbk_ids():
    monographs = pd.DataFrame(
        [
            {
                "nbk_id": "NBK_DUP",
                "drug_name": "Alpha Drug",
                "excerpt": "Alpha excerpt",
            },
            {
                "nbk_id": "NBK_DUP",
                "drug_name": "Beta Drug",
                "excerpt": "Beta excerpt",
            },
        ]
    )
    matcher = LiverToxMatcher(monographs)
    matches = run_match(matcher, ["Alpha Drug", "Beta Drug"])
    mapping = matcher.build_patient_mapping(["Alpha Drug", "Beta Drug"], matches)
    assert mapping[0]["matched_livertox_row"]["excerpt"] == "Alpha excerpt"
    assert mapping[1]["matched_livertox_row"]["excerpt"] == "Beta excerpt"
    assert mapping[0]["matched_livertox_row"]["drug_name"] == "Alpha Drug"
    assert mapping[1]["matched_livertox_row"]["drug_name"] == "Beta Drug"


def test_unified_dataset_brand_lookup_without_explicit_master_list():
    dataset = pd.DataFrame(
        [
            {
                "nbk_id": "NBK100",
                "drug_name": "Unified Drug",
                "excerpt": "Example excerpt",
                "synonyms": "",
                "ingredient": "Unified Ingredient",
                "brand_name": "UnifiedBrand",
            }
        ]
    )
    matcher = LiverToxMatcher(dataset)
    matches = run_match(matcher, ["UnifiedBrand"])
    match = matches[0]
    assert match is not None
    assert match.nbk_id == "NBK100"
    assert match.reason.startswith("brand_")
