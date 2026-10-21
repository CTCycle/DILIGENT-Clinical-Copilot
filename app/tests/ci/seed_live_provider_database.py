# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Seed isolated clinical catalogs for the live-provider browser gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SERVER_ROOT = Path(__file__).resolve().parents[2] / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from configurations.startup import get_server_settings
from repositories.context import RepositoryContext
from repositories.database.initializer import initialize_sqlite_database
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository


def seed_live_provider_database() -> tuple[int, int]:
    database_settings = get_server_settings().database
    if database_settings.backend != "sqlite":
        raise RuntimeError("Live-provider browser seeding requires SQLite.")

    initialize_sqlite_database(database_settings)
    context = RepositoryContext.create()
    DrugCatalogRepository(context).upsert_drugs_catalog_records(
        [
            {
                "rxcui": "723",
                "raw_name": "amoxicillin",
                "term_type": "IN",
                "name": "amoxicillin",
                "brand_names": "Amoxil",
                "synonyms": "amoxicillin",
            }
        ]
    )
    KnowledgeRepository(context).save_livertox_records(
        pd.DataFrame(
            [
                {
                    "drug_name": "amoxicillin",
                    "nbk_id": "NBK548517",
                    "synonyms": "amoxicillin",
                    "excerpt": "Synthetic release-gate catalog evidence for amoxicillin.",
                    "include_in_livertox": True,
                    "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK548517/",
                }
            ]
        )
    )

    _, rxnav_count = DrugCatalogRepository(context).list_rxnav_catalog(
        search=None, offset=0, limit=1
    )
    _, livertox_count = KnowledgeRepository(context).list_livertox_catalog(
        search=None, offset=0, limit=1
    )
    if rxnav_count < 1 or livertox_count < 1:
        raise RuntimeError(
            "Live-provider browser catalog fixture did not persist "
            f"(RxNav={rxnav_count}, LiverTox={livertox_count})."
        )
    return rxnav_count, livertox_count


if __name__ == "__main__":
    rxnav_count, livertox_count = seed_live_provider_database()
    print(
        "Seeded the live-provider browser database "
        f"(RxNav={rxnav_count}, LiverTox={livertox_count})."
    )
