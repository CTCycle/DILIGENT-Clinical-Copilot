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

from configurations.startup import get_server_settings  # noqa: E402
from repositories.context import RepositoryContext  # noqa: E402
from repositories.database.initializer import initialize_sqlite_database  # noqa: E402
from repositories.drug_catalog_repository import DrugCatalogRepository  # noqa: E402
from repositories.knowledge_repository import KnowledgeRepository  # noqa: E402
from repositories.terminology_repository import TerminologyRepository  # noqa: E402
from services.updater.terminology.snapshot import (  # noqa: E402
    StagedConcept,
    TerminologySnapshot,
)


def _seed_rxnorm_fixture(repository: TerminologyRepository) -> None:
    snapshot = TerminologySnapshot(
        source="rxnorm", version="live-provider-fixture", license="Public domain"
    )
    snapshot.concepts["723"] = StagedConcept(key="723", name="amoxicillin", rxcui="723")
    snapshot.add_alias("723", "amoxicillin", "canonical", term_type="IN")
    snapshot.add_alias("723", "Amoxil", "brand", term_type="BN")
    run_id = repository.start_run(source="rxnorm", mode="full", license_text=snapshot.license)
    stats = repository.persist_snapshot(snapshot, run_id)
    repository.finish_run(
        run_id, status="completed", version=snapshot.version, stats=stats.to_dict()
    )


def seed_live_provider_database() -> tuple[int, int]:
    database_settings = get_server_settings().database
    if database_settings.backend != "sqlite":
        raise RuntimeError("Live-provider browser seeding requires SQLite.")

    initialize_sqlite_database(database_settings)
    context = RepositoryContext.create()
    terminology = TerminologyRepository(context)
    _seed_rxnorm_fixture(terminology)
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
    terminology.rebuild_livertox_links()

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
