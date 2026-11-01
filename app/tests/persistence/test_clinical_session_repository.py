# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import date

from repositories.schemas.clinical import ClinicalLabObservation
from repository_fixtures import build_repository_graph
from sqlalchemy import select

###############################################################################
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

###############################################################################
def test_repeated_lab_representations_persist_one_observation(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    repository = build_repository_graph(
        engine=persistence_engine
    ).clinical_session_repository
    entry = {"marker_name": "ALT", "value": 37, "unit": "U/L"}
    session_id = repository.save_clinical_session(
        {
            "patient_name": "Synthetic duplicate labs",
            "session_status": "successful",
            "session_result_payload": {"lab_timeline": [entry, dict(entry)]},
        }
    )
    assert session_id is not None

    context = build_repository_graph(engine=persistence_engine).context
    with context.session_factory() as db_session:
        rows = db_session.scalars(
            select(ClinicalLabObservation).where(
                ClinicalLabObservation.session_id == session_id
            )
        ).all()
    assert len(rows) == 1

###############################################################################
def test_attachments_are_idempotent_version_scoped_and_cascade(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    repository = graph.clinical_session_repository
    session_id = repository.save_clinical_session(
        {"patient_name": "Synthetic attachments", "session_status": "successful"}
    )
    assert session_id is not None
    version = graph.session_revision_repository.get_version_record_for_session(
        session_id
    )
    assert version is not None
    upload = {
        "version_id": version["version_id"],
        "kind": "json",
        "file_name": "a.json",
        "media_type": "application/json",
        "content": b'{"a": 1}',
    }

    first = repository.add_attachment(session_id, **upload)
    repeated = repository.add_attachment(session_id, **upload)
    assert first is not None and repeated is not None
    assert repeated["attachment_id"] == first["attachment_id"]
    assert first["version_id"] == version["version_id"]

    fresh = build_repository_graph(engine=persistence_engine).clinical_session_repository
    stored = fresh.get_attachment_content(session_id, first["attachment_id"])
    assert stored is not None and stored[1] == b'{"a": 1}'
    assert fresh.get_attachment_content(session_id + 1, first["attachment_id"]) is None

    assert repository.delete_session(session_id) is True
    assert fresh.list_attachments(session_id) == []
