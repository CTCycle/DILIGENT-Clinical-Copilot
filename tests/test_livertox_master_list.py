import os
import sys
from unittest.mock import patch

import json
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DILIGENT.app.utils.repository.serializer import (  # noqa: E402
    DRUGS_CATALOG_COLUMNS,
    DataSerializer,
    LIVERTOX_COLUMNS,
)
from DILIGENT.app.utils.updater.livertox import LiverToxUpdater  # noqa: E402
from DILIGENT.app.utils.updater.rxnav import (  # noqa: E402
    DrugsCatalogUpdater,
    RxNormCandidate,
)


###############################################################################
@pytest.fixture()
def updater() -> LiverToxUpdater:
    return LiverToxUpdater(
        ".",
        redownload=False,
        serializer=DataSerializer(),
        database_client=None,
    )


###############################################################################
def test_build_unified_dataset_merges_sources(updater: LiverToxUpdater) -> None:
    monographs = pd.DataFrame(
        [
            {
                "drug_name": "Valid Drug",
                "nbk_id": "NBK100",
                "excerpt": "Example excerpt",
            },
            {
                "drug_name": "Archive Only",
                "nbk_id": "NBK200",
                "excerpt": "Standalone excerpt",
            },
        ]
    )
    master_frame = pd.DataFrame(
        [
            {
                "chapter_title": "Valid Drug",
                "likelihood_score": "A",
                "last_update": "2024-01-01",
            },
            {
                "chapter_title": "Master Only",
                "likelihood_score": "B",
                "last_update": "2024-01-02",
            },
        ]
    )
    metadata = {"source_url": "http://example.com", "last_modified": "2024-01-05"}

    unified = updater.build_unified_dataset(monographs, master_frame, metadata)

    expected_columns = [
        "drug_name",
        "likelihood_score",
        "last_update",
        "reference_count",
        "year_approved",
        "agent_classification",
        "primary_classification",
        "secondary_classification",
        "include_in_livertox",
        "source_url",
        "source_last_modified",
        "nbk_id",
        "excerpt",
    ]
    assert list(unified.columns) == expected_columns
    assert set(unified["drug_name"]) == {"Valid Drug", "Archive Only", "Master Only"}
    valid_row = unified[unified["drug_name"] == "Valid Drug"].iloc[0]
    assert valid_row["likelihood_score"] == "A"
    assert valid_row["source_url"] == "http://example.com"
    archive_row = unified[unified["drug_name"] == "Archive Only"].iloc[0]
    assert archive_row["nbk_id"] == "NBK200"


###############################################################################
def test_sanitization_rules_drop_invalid_values(updater: LiverToxUpdater) -> None:
    raw = pd.DataFrame(
        [
            {
                "drug_name": "12345",
                "nbk_id": "NBK001",
                "excerpt": "",
            },
            {
                "drug_name": "Valid Name",
                "nbk_id": "NBK002",
                "excerpt": "  ",
            },
            {
                "drug_name": "Another Valid",
                "nbk_id": "NBK003",
                "excerpt": "Excerpt",
            },
        ]
    )

    sanitized = updater.sanitize_unified_dataset(raw)

    assert list(sanitized["drug_name"]) == ["Valid Name", "Another Valid"]
    assert sanitized.iloc[0]["excerpt"] == "Not available"
    assert sanitized.iloc[1]["excerpt"] == "Excerpt"


###############################################################################
def test_finalize_dataset_fills_missing_entries(updater: LiverToxUpdater) -> None:
    dataset = pd.DataFrame(
        [
            {
                "drug_name": "Final Drug",
                "nbk_id": pd.NA,
                "excerpt": pd.NA,
                "likelihood_score": pd.NA,
            }
        ]
    )
    finalized = updater.finalize_dataset(dataset)
    row = finalized.iloc[0]
    assert row["excerpt"] == "Not available"
    assert row["likelihood_score"] == "Not available"


###############################################################################
def test_serializer_roundtrip_uses_unified_table() -> None:
    serializer = DataSerializer()
    livertox_frame = pd.DataFrame(
        [
            {
                "drug_name": "Roundtrip",
                "nbk_id": "NBK999",
                "excerpt": "Excerpt",
                "likelihood_score": "A",
                "last_update": "2024-01-01",
                "reference_count": "2",
                "year_approved": "2023",
                "agent_classification": "Class",
                "primary_classification": "Primary",
                "secondary_classification": "Secondary",
                "include_in_livertox": "Yes",
                "source_url": "http://example.com",
                "source_last_modified": "2024-01-01",
            }
        ]
    )
    catalog_frame = pd.DataFrame(
        [
            {
                "rxcui": "1",
                "full_name": "Catalog Drug",
                "term_type": "SCD",
                "ingredient": "[\"Ingredient\"]",
                "brand_name": "[\"Brand\"]",
                "synonyms": "[\"Catalog Drug\", \"Brand\"]",
            }
        ]
    )

    with patch(
        "DILIGENT.app.utils.repository.serializer.database.save_into_database"
    ) as save_mock:
        serializer.save_livertox_records(livertox_frame)
        saved_frame = save_mock.call_args.args[0]
        assert list(saved_frame.columns) == LIVERTOX_COLUMNS

    def loader(table_name: str) -> pd.DataFrame:
        if table_name == "LIVERTOX_DATA":
            return livertox_frame
        if table_name == "DRUGS_CATALOG":
            return catalog_frame
        raise AssertionError(table_name)

    with patch(
        "DILIGENT.app.utils.repository.serializer.database.load_from_database",
        side_effect=loader,
    ):
        monographs = serializer.get_livertox_records()
        assert list(monographs.columns) == LIVERTOX_COLUMNS
        catalog = serializer.get_drugs_catalog()
        assert list(catalog.columns) == DRUGS_CATALOG_COLUMNS
        master = serializer.get_livertox_master_list()
        assert list(master.columns) == DRUGS_CATALOG_COLUMNS


###############################################################################
class StubCatalogClient:
    TIMEOUT = 0.01

    def __init__(self, mapping: dict[str, dict[str, RxNormCandidate]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    ###########################################################################
    def get_candidates(self, alias: str) -> dict[str, RxNormCandidate]:
        self.calls.append(alias)
        return self.mapping.get(alias, {})

    ###########################################################################
    def standardize_term(self, value: str) -> str:
        return value


###############################################################################
def test_drugs_catalog_enrichment_collects_synonyms() -> None:
    mapping = {
        "Primary Drug": {
            "primary drug": RxNormCandidate(
                value="primary drug",
                kind="original",
                display="Primary Drug",
            ),
            "primary ingredient": RxNormCandidate(
                value="primary ingredient",
                kind="ingredient",
                display="Primary Ingredient",
            ),
            "brandname": RxNormCandidate(
                value="brandname",
                kind="brand",
                display="BrandName",
            ),
        }
    }
    rx_client = StubCatalogClient(mapping)
    updater = DrugsCatalogUpdater(rx_client=rx_client, serializer=DataSerializer())
    concepts = [
        {
            "rxcui": "1",
            "full_name": "Primary Drug",
            "term_type": "SCD",
            "aliases": ["Primary Drug"],
        }
    ]

    records = updater.enrich_batch(concepts)
    assert len(records) == 1
    record = records[0]
    assert json.loads(record["ingredient"]) == ["Primary Ingredient"]
    assert json.loads(record["brand_name"]) == ["BrandName"]
    synonyms = json.loads(record["synonyms"])
    assert "Primary Drug" in synonyms
    assert "BrandName" in synonyms
    assert rx_client.calls == ["Primary Drug"]
