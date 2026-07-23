from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from repositories.schemas.clinical import ClinicalSession

###############################################################################
def test_session_pagination_has_stable_timestamp_and_id_order(persistence_session) -> None:  # type: ignore[no-untyped-def]
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    ids: list[int] = []
    for index in range(3):
        session = ClinicalSession(
            patient_name=f"Page {index}",
            session_timestamp=timestamp + timedelta(minutes=index),
        )
        persistence_session.add(session)
        persistence_session.flush()
        ids.append(session.id)
    persistence_session.commit()

    statement = (
        select(ClinicalSession.id)
        .order_by(ClinicalSession.session_timestamp.desc(), ClinicalSession.id.desc())
        .offset(1)
        .limit(1)
    )
    page = persistence_session.scalars(statement).one()
    assert page == ids[-2], (
        f"Expected second-last inserted id ({ids[-2]}) at offset 1, "
        f"got {page}. Insert order: {ids}"
    )
