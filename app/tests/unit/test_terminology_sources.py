# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import zipfile
from pathlib import Path

import openpyxl

from services.updater.terminology.sources.chembl import ChemblSource
from services.updater.terminology.sources.drugcentral import (
    DrugCentralConnection,
    DrugCentralSource,
)
from services.updater.terminology.sources.rxnorm import RxNormPrescribableSource
from services.updater.terminology.sources.swissmedic import SwissmedicSource


# -----------------------------------------------------------------------------
def conso(rxcui: str, tty: str, text: str) -> str:
    fields = [""] * 18
    fields[0], fields[11], fields[12], fields[14], fields[16] = rxcui, "RXNORM", tty, text, "N"
    return "|".join(fields)


# -----------------------------------------------------------------------------
def rel(rxcui1: str, rela: str, rxcui2: str) -> str:
    fields = [""] * 16
    fields[0], fields[4], fields[7], fields[10] = rxcui1, rxcui2, rela, "RXNORM"
    return "|".join(fields)


###############################################################################
def test_rxnorm_archive_builds_hierarchy_combinations_and_brands(tmp_path: Path) -> None:
    archive = tmp_path / "rxnorm.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "rrf/RXNCONSO.RRF",
            "\n".join(
                [
                    conso("6918", "IN", "metoprolol"),
                    conso("221124", "PIN", "metoprolol succinate"),
                    conso("723", "IN", "amoxicillin"),
                    conso("48203", "IN", "clavulanate"),
                    conso("19711", "MIN", "amoxicillin / clavulanate"),
                    conso("151392", "BN", "Augmentin"),
                    conso("617296", "SBD", "amoxicillin 875 MG / clavulanate 125 MG Oral Tablet [Augmentin]"),
                ]
            ),
        )
        bundle.writestr(
            "rrf/RXNREL.RRF",
            "\n".join(
                [
                    rel("6918", "form_of", "221124"),
                    rel("19711", "part_of", "723"),
                    rel("19711", "part_of", "48203"),
                ]
            ),
        )

    snapshot = RxNormPrescribableSource(tmp_path).parse_archive(
        archive, version="2026-10", should_stop=lambda: False
    )

    assert snapshot.concepts["221124"].parent_key == "6918"
    assert snapshot.concepts["221124"].concept_type == "precise_ingredient"
    assert snapshot.concepts["19711"].component_keys == ("48203", "723")
    brands = [(item.concept_key, item.alias) for item in snapshot.aliases if item.alias_kind == "brand"]
    assert brands == [("19711", "Augmentin")]
    product = snapshot.products[0]
    assert [(item.concept_key, item.strength_value, item.strength_unit) for item in product.ingredients] == [
        ("723", 875.0, "mg"),
        ("48203", 125.0, "mg"),
    ]


###############################################################################
def test_drugcentral_never_relates_prodrugs_to_their_active_moiety() -> None:
    snapshot = DrugCentralSource(DrugCentralConnection()).build_snapshot(
        version="2023",
        structures=[(1, "droxidopa", "QXWYKJLNLSIPIN-JGVFFNPUSA-N"), (2, "norepinephrine", None)],
        synonyms=[(1, "L-DOPS")],
        identifiers=[(1, "UNII", "J7A92W69L7"), (1, "INN_ID", "5521")],
        rxnorm=[(1, "1489913")],
        atc=[(1, "C01CA27")],
    )

    droxidopa = snapshot.concepts["1"]
    assert droxidopa.parent_key is None
    assert snapshot.concepts["2"].parent_key is None
    assert droxidopa.identifiers == {
        "unii": "J7A92W69L7",
        "inn_id": "5521",
        "drugcentral": "1",
        "inchikey": "QXWYKJLNLSIPIN-JGVFFNPUSA-N",
    }
    assert (droxidopa.rxcui, droxidopa.preferred_inn, droxidopa.atc_codes) == ("1489913", "droxidopa", ("C01CA27",))
    assert {(item.alias, item.alias_kind) for item in snapshot.aliases if item.concept_key == "1"} == {
        ("droxidopa", "inn"),
        ("L-DOPS", "synonym"),
    }


###############################################################################
def test_chembl_salts_become_precise_forms_of_their_parent() -> None:
    snapshot = ChemblSource().build_snapshot(
        [
            {
                "molecule_chembl_id": "CHEMBL13",
                "pref_name": "METOPROLOL",
                "molecule_hierarchy": {"parent_chembl_id": "CHEMBL13"},
                "molecule_synonyms": [{"molecule_synonym": "Metoprolol", "syn_type": "INN"}],
            },
            {
                "molecule_chembl_id": "CHEMBL1200",
                "pref_name": "METOPROLOL SUCCINATE",
                "molecule_hierarchy": {"parent_chembl_id": "CHEMBL13"},
                "molecule_synonyms": [{"molecule_synonym": "Toprol-XL", "syn_type": "TRADE_NAME"}],
            },
        ],
        version="CHEMBL_35",
    )

    salt = snapshot.concepts["CHEMBL1200"]
    assert (salt.name, salt.concept_type, salt.parent_key) == ("metoprolol succinate", "precise_ingredient", "CHEMBL13")
    assert snapshot.concepts["CHEMBL13"].preferred_inn == "metoprolol"
    assert ("Toprol-XL", "trade_name") in {(item.alias, item.alias_kind) for item in snapshot.aliases}


###############################################################################
def test_swissmedic_workbook_header_detection_and_latin_lookup(tmp_path: Path) -> None:
    path = tmp_path / "packs.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Zugelassene Packungen, Stand 01.10.2026"])
    sheet.append(
        [
            "Zulassungs-Nummer",
            "Dosisstärke-nummer",
            "Bezeichnung des Arzneimittels",
            "ATC-Code",
            "Wirkstoff(e)",
            "Zusammensetzung",
        ]
    )
    sheet.append(
        [
            "54321",
            "01",
            "Dafalgan 500 mg, Filmtabletten",
            "N02BE01",
            "paracetamolum",
            "paracetamolum 500 mg, excipiens pro compresso obducto",
        ]
    )
    workbook.save(path)

    snapshot = SwissmedicSource(tmp_path).parse_workbook(path, content_hash="abc123def4567", should_stop=lambda: False)

    assert snapshot.version == "2026-10-01 sha256:abc123def456"
    product = snapshot.products[0]
    assert (product.product_id, product.brand_name, product.country, product.atc_code) == (
        "54321-01",
        "Dafalgan",
        "CH",
        "N02BE01",
    )
    ingredient = product.ingredients[0]
    assert "paracetamol" in ingredient.lookup_names
    assert (ingredient.strength_value, ingredient.strength_unit) == (500.0, "mg")
