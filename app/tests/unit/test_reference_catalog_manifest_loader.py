from __future__ import annotations

import json

import pytest
from services.catalogs.manifest_loader import load_catalog_manifest, validate_manifest


def test_validate_manifest_rejects_duplicate_normalized_values() -> None:
    payload = {
        "manifest": "x",
        "version": 1,
        "entries": [
            {
                "domain": "d",
                "category": "c",
                "key": "k",
                "values": ["PCK", "pck"],
            }
        ],
    }
    with pytest.raises(ValueError):
        validate_manifest(payload)


def test_load_catalog_manifest_parses_entries(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "manifest": "demo",
                "version": 1,
                "description": "demo",
                "entries": [
                    {
                        "domain": "clinical_extraction",
                        "category": "drug_dosage_units",
                        "key": "mass_units",
                        "locale": "und",
                        "values": ["mg", "mcg"],
                        "metadata": {"case_sensitive": False, "match_mode": "token"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_catalog_manifest(path)
    assert loaded.manifest == "demo"
    assert loaded.version == 1
    assert len(loaded.entries) == 2
    assert {entry.value for entry in loaded.entries} == {"mg", "mcg"}
