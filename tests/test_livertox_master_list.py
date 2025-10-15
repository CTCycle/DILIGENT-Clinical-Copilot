import os
import sys
import threading
import time
from unittest.mock import patch

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DILIGENT.app.utils.repository.serializer import DataSerializer, LIVERTOX_COLUMNS
from DILIGENT.app.utils.updater.livertox import LiverToxUpdater


###############################################################################
class StubRxClient:
    UNIT_STOPWORDS = {"mg", "ml", "ng"}

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    ###########################################################################
    def fetch_drug_terms(self, name: str) -> list[str]:
        return self.mapping.get(name, [])


###############################################################################
class RateCheckingRxClient:
    UNIT_STOPWORDS = StubRxClient.UNIT_STOPWORDS

    def __init__(self, mapping: dict[str, list[str]], delay: float = 0.01) -> None:
        self.mapping = mapping
        self.delay = delay
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.call_count = 0

    ###########################################################################
    def fetch_drug_terms(self, name: str) -> list[str]:
        with self.lock:
            self.active += 1
            self.call_count += 1
            if self.active > self.max_active:
                self.max_active = self.active
        try:
            time.sleep(self.delay)
            return self.mapping.get(name, [])
        finally:
            with self.lock:
                self.active -= 1


###############################################################################
@pytest.fixture()
def updater() -> LiverToxUpdater:
    return LiverToxUpdater(
        ".",
        redownload=False,
        rx_client=StubRxClient({}),
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
                "ingredient": "Valid Ingredient",
                "brand_name": "ValidBrand",
                "likelihood_score": "A",
            },
            {
                "chapter_title": "Master Only",
                "ingredient": "Master Ingredient",
                "brand_name": "MasterBrand",
                "likelihood_score": "B",
            },
        ]
    )
    metadata = {"source_url": "http://example.com", "last_modified": "2024-01-01"}

    unified = updater.build_unified_dataset(monographs, master_frame, metadata)

    assert set(unified["drug_name"]) == {"Valid Drug", "Archive Only", "Master Only"}
    valid_row = unified[unified["drug_name"] == "Valid Drug"].iloc[0]
    assert valid_row["ingredient"] == "Valid Ingredient"
    assert valid_row["source_url"] == "http://example.com"
    archive_row = unified[unified["drug_name"] == "Archive Only"].iloc[0]
    assert pd.isna(archive_row["ingredient"])


###############################################################################
def test_sanitization_rules_drop_invalid_values(updater: LiverToxUpdater) -> None:
    raw = pd.DataFrame(
        [
            {
                "drug_name": "12345",
                "ingredient": "Ingredient",
                "brand_name": "Brand",
                "excerpt": "",
            },
            {
                "drug_name": "Valid Name",
                "ingredient": "Ingr#dient",
                "brand_name": "Brand",
                "excerpt": "  ",
            },
            {
                "drug_name": "Another Valid",
                "ingredient": "Good Ingredient",
                "brand_name": "ValidBrand",
                "excerpt": "Excerpt",
            },
        ]
    )
    sanitized = updater.sanitize_unified_dataset(raw)

    assert list(sanitized["drug_name"]) == ["Another Valid"]
    assert sanitized.iloc[0]["excerpt"] == "Excerpt"


###############################################################################
def test_enrichment_uses_all_aliases() -> None:
    mapping = {
        "Primary Drug": [
            "Primary Synonym",
            "mg",
            "Multi Word Term",
            "Invalid@Term",
            "Abc",
        ],
        "Primary Ingredient": ["Ingredient Synonym"],
        "PrimaryBrand": ["Brand Synonym", "5"],
    }
    updater = LiverToxUpdater(
        ".",
        redownload=False,
        rx_client=StubRxClient(mapping),
        serializer=DataSerializer(),
        database_client=None,
    )
    dataset = pd.DataFrame(
        [
            {
                "drug_name": "Primary Drug",
                "ingredient": "Primary Ingredient",
                "brand_name": "PrimaryBrand",
                "excerpt": "Example",
                "nbk_id": "NBK123",
            }
        ]
    )

    enriched = updater.enrich_records(updater.sanitize_unified_dataset(dataset))
    synonyms = enriched.iloc[0]["synonyms"]

    assert "Primary Synonym" in synonyms
    assert "Ingredient Synonym" in synonyms
    assert "Brand Synonym" in synonyms
    assert "mg" not in synonyms
    assert "Invalid@Term" not in synonyms
    assert "Abc" not in synonyms


###############################################################################
def test_enrichment_respects_max_workers() -> None:
    mapping: dict[str, list[str]] = {}
    rows: list[dict[str, str | pd.NA]] = []
    for index in range(12):
        drug = f"Drug {index}"
        ingredient = f"Ingredient {index}"
        brand = f"Brand {index}"
        mapping[drug] = [f"{drug} Alias"]
        mapping[ingredient] = [f"{ingredient} Alias"]
        mapping[brand] = [f"{brand} Alias"]
        rows.append(
            {
                "drug_name": drug,
                "ingredient": ingredient,
                "brand_name": brand,
                "synonyms": pd.NA,
            }
        )

    rx_client = RateCheckingRxClient(mapping, delay=0.01)
    updater = LiverToxUpdater(
        ".",
        redownload=False,
        rx_client=rx_client,
        serializer=DataSerializer(),
        database_client=None,
    )
    updater.RXNAV_MAX_WORKERS = 4

    dataset = pd.DataFrame(rows)
    enriched = updater.enrich_records(dataset)

    assert rx_client.call_count == len(mapping)
    assert rx_client.max_active <= updater.RXNAV_MAX_WORKERS
    assert enriched["synonyms"].notna().all()


###############################################################################
def test_finalize_dataset_fills_missing_entries(updater: LiverToxUpdater) -> None:
    dataset = pd.DataFrame(
        [
            {
                "drug_name": "Final Drug",
                "ingredient": pd.NA,
                "brand_name": pd.NA,
                "nbk_id": pd.NA,
                "excerpt": pd.NA,
                "synonyms": pd.NA,
            }
        ]
    )
    finalized = updater.finalize_dataset(dataset)
    row = finalized.iloc[0]
    assert row["ingredient"] == "Not available"
    assert row["excerpt"] == "Not available"
    assert row["synonyms"] == "Not available"


###############################################################################
def test_serializer_roundtrip_uses_unified_table() -> None:
    serializer = DataSerializer()
    frame = pd.DataFrame(
        [
            {
                "drug_name": "Roundtrip",
                "ingredient": "Base",
                "brand_name": "Brand",
                "nbk_id": "NBK999",
                "excerpt": "Excerpt",
                "synonyms": "Alias",
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

    with patch(
        "DILIGENT.app.utils.repository.serializer.database.save_into_database"
    ) as save_mock:
        serializer.save_livertox_records(frame)
        assert save_mock.called
        saved_frame = save_mock.call_args.args[0]
        assert "drug_name" in saved_frame.columns
        assert "ingredient" in saved_frame.columns

    with patch(
        "DILIGENT.app.utils.repository.serializer.database.load_from_database",
        return_value=frame,
    ):
        monographs = serializer.get_livertox_records()
        assert list(monographs.columns) == LIVERTOX_COLUMNS
        master = serializer.get_livertox_master_list()
        expected_master_cols = [
            "drug_name",
            "ingredient",
            "brand_name",
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
        ]
        assert list(master.columns) == expected_master_cols
        assert master.iloc[0]["drug_name"] == "Roundtrip"
