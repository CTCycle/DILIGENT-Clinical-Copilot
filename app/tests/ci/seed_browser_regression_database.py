# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Seed the smallest repository-backed dataset required by browser regression."""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[2] / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from configurations.startup import get_server_settings  # noqa: E402
from repositories.clinical_session_repository import ClinicalSessionRepository  # noqa: E402
from repositories.context import RepositoryContext  # noqa: E402
from repositories.database.initializer import initialize_sqlite_database  # noqa: E402
from repositories.session_timeline_repository import SessionTimelineRepository  # noqa: E402

SEED_PATIENT_NAME = "CI Browser Regression Subject"
SEED_TIMESTAMP = datetime(2026, 9, 29, 8, 0)


###############################################################################
def seed_browser_regression_database() -> tuple[int, int]:
    database_settings = get_server_settings().database
    if database_settings.backend != "sqlite":
        raise RuntimeError("Browser regression seeding requires the SQLite backend.")

    initialize_sqlite_database(database_settings)
    context = RepositoryContext.create()
    sessions = ClinicalSessionRepository(context)
    timelines = SessionTimelineRepository(context)

    existing_rows, _ = sessions.list_sessions(
        search=SEED_PATIENT_NAME,
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=100,
    )
    for row in existing_rows:
        session_id = row.get("session_id")
        if isinstance(session_id, int):
            sessions.delete_session(session_id)

    session_id = sessions.save_clinical_session(
        {
            "patient_name": SEED_PATIENT_NAME,
            "patient_visit_date": date(2026, 9, 29),
            "session_timestamp": SEED_TIMESTAMP,
            "session_status": "successful",
            "session_kind": "browser_regression",
            "anamnesis": "Synthetic browser regression session with a persisted timeline.",
            "drugs": "Amoxicillin 500 mg BID",
            "laboratory_analysis": "ALT 210 U/L; AST 180 U/L; ALP 130 U/L.",
            "session_result_payload": {
                "report": "Synthetic browser regression report.",
                "issues": [],
            },
        }
    )
    if session_id is None:
        raise RuntimeError("Browser regression session seed did not persist.")

    timeline = timelines.create_session_timeline_record(
        session_id,
        {
            "session_id": session_id,
            "generated_at": SEED_TIMESTAMP,
            "generation_status": "llm_generated",
            "generation_note": "Deterministic browser regression fixture.",
            "source_model": "ci-fake-ollama",
            "source_kind": "local",
            "model_provider": "ollama",
            "events": [
                {
                    "event_id": "browser-regression-baseline",
                    "title": "Browser regression baseline",
                    "description": "Persisted synthetic baseline event.",
                    "event_type": "disease",
                    "timing_type": "explicit_date",
                    "event_date": "2026-09-29",
                    "date_precision": "day",
                    "date_certainty": "explicit",
                    "extracted_timing_text": "September 29, 2026",
                    "source_evidence": "Synthetic browser regression fixture.",
                    "source": "browser-regression-seed",
                    "confidence": 1.0,
                    "confidence_rationale": "Deterministic fixture.",
                    "sort_order": 0,
                }
            ],
        },
    )
    if timeline is None or not isinstance(timeline.get("timeline_id"), int):
        raise RuntimeError("Browser regression timeline seed did not persist.")
    return session_id, int(timeline["timeline_id"])


###############################################################################
if __name__ == "__main__":
    seeded_session_id, seeded_timeline_id = seed_browser_regression_database()
    print(
        "Seeded browser regression session "
        f"{seeded_session_id} with timeline {seeded_timeline_id}."
    )
