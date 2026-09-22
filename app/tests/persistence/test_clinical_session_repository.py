from __future__ import annotations

from datetime import date

from repositories.schemas.clinical import ClinicalLabObservation
from repository_fixtures import build_repository_graph
from sqlalchemy import select


def test_persisted_lab_sample_date_populates_observation_at(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    repository = build_repository_graph(
        engine=persistence_engine
    ).clinical_session_repository
    session_id = repository.save_clinical_session(
        {
            "patient_name": "Synthetic sample-date persistence",
            "session_status": "successful",
            "session_result_payload": {
                "lab_timeline": [
                    {
                        "marker_name": "ALT",
                        "sample_date": "2026-03-01",
                        "value": 22,
                        "unit": "U/L",
                        "upper_limit_normal": 40,
                    }
                ]
            },
        }
    )
    assert session_id is not None

    context = build_repository_graph(engine=persistence_engine).context
    with context.session_factory() as db_session:
        observation = db_session.scalar(
            select(ClinicalLabObservation).where(
                ClinicalLabObservation.session_id == session_id
            )
        )

    assert observation is not None
    assert observation.observation_at is not None
    assert observation.observation_at.date() == date(2026, 3, 1)
    assert observation.metadata_json is not None
    assert observation.metadata_json["sample_date"] == "2026-03-01"
