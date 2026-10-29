# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import pytest

from services.terminology.normalization import (
    Strength,
    language_variants,
    latin_to_english_variants,
    normalize_drug_input,
    orthographic_skeleton,
)


###############################################################################
@pytest.mark.parametrize(
    ("text", "key", "strengths", "qualifiers"),
    [
        ("Paracetamol 500 mg", "paracetamol", (Strength(500.0, "mg"),), ()),
        ("Metoprolol succinate ER 50mg", "metoprolol succinate", (Strength(50.0, "mg"),), ("er",)),
        ("Fentanyl 25 mcg/h", "fentanyl", (Strength(25.0, "mcg/hr"),), ()),
        ("Ibuprofen 600 mg Tabl", "ibuprofen", (Strength(600.0, "mg"),), ("tabl",)),
        ("Vitamin D3 1000 UI", "vitamin d3", (Strength(1000.0, "iu"),), ()),
    ],
)
def test_strengths_and_qualifiers_are_kept_out_of_the_key(
    text: str, key: str, strengths: tuple[Strength, ...], qualifiers: tuple[str, ...]
) -> None:
    normalized = normalize_drug_input(text)

    assert normalized.key == key
    assert normalized.strengths == strengths
    assert normalized.qualifiers == qualifiers
    # The salt is identity-relevant and therefore stays inside the key.
    assert normalize_drug_input("metoprolol tartrate").key == "metoprolol tartrate"


###############################################################################
def test_combination_separators_produce_component_keys() -> None:
    assert normalize_drug_input("Amoxicillin/Clavulansäure 875/125 mg").component_keys == (
        "amoxicillin",
        "clavulansaure",
    )
    assert normalize_drug_input("trimethoprim + sulfamethoxazole").component_keys == (
        "trimethoprim",
        "sulfamethoxazole",
    )
    # Short hyphenated prefixes are names, not combinations.
    assert normalize_drug_input("Co-Amoxi").component_keys == ()


###############################################################################
def test_latin_pharmacopoeial_names_map_to_english_ingredients() -> None:
    assert "valproic acid" in latin_to_english_variants("acidum valproicum")
    assert "metformin hydrochloride" in latin_to_english_variants("metformini hydrochloridum")
    assert "pantoprazole sodium" in latin_to_english_variants("pantoprazolum natricum")


###############################################################################
def test_romance_language_endings_and_orthographic_skeleton() -> None:
    assert "metformin" in language_variants("metformina")
    assert "omeprazole" in language_variants("omeprazolo")
    assert "amoxicillin" in language_variants("amoxicilline")
    assert orthographic_skeleton("phenytoin") == orthographic_skeleton("fenitoin")
    assert orthographic_skeleton("amoxicillin") == orthographic_skeleton("amoxicilin")
    # Different drugs keep different skeletons.
    assert orthographic_skeleton("hydroxyzine") != orthographic_skeleton("hydralazine")
