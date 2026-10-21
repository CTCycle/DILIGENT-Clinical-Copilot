# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Create synthetic upgrade data using the published baseline's repositories."""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

# The harness supplies a git-archived source directory from the published baseline.
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))

from configurations.startup import get_server_settings  # noqa: E402
from repositories.clinical_session_repository import ClinicalSessionRepository  # noqa: E402
from repositories.context import RepositoryContext  # noqa: E402
from repositories.database.initializer import initialize_sqlite_database  # noqa: E402
from repositories.serialization.application_configuration import (  # noqa: E402
    ApplicationConfigurationSerializer,
)


def seed(baseline_version: str) -> None:
    settings = get_server_settings().database
    if settings.backend != "sqlite":
        raise RuntimeError("Upgrade seeding requires isolated SQLite storage.")
    initialize_sqlite_database(settings, seed_catalogs=False)
    context = RepositoryContext.create()
    repository = ClinicalSessionRepository(context)
    session_id = repository.save_clinical_session(
        {
            "patient_name": "Synthetic MSI Upgrade Subject",
            "patient_visit_date": date(2026, 10, 18),
            "session_timestamp": datetime(2026, 10, 18, 9),
            "session_status": "successful",
            "anamnesis": "Synthetic upgrade fixture; no real patient data.",
            "drugs": "Amoxicillin 500 mg BID",
            "laboratory_analysis": "ALT 210 U/L; ALP 130 U/L.",
            "session_result_payload": {
                "report": f"Synthetic v{baseline_version} upgrade preservation report.",
                "issues": [],
            },
        }
    )
    if session_id is None:
        raise RuntimeError("The baseline application did not persist the fixture.")
    configuration = ApplicationConfigurationSerializer(
        engine=context.engine,
        session_factory=context.session_factory,
    )
    payload = configuration.load() or {}
    payload["polling_interval"] = 2.0
    configuration.save(payload)
    context.engine.dispose()


if __name__ == "__main__":
    seed(sys.argv[2])
