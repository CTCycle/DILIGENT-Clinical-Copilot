from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from repositories.schemas.clinical import ClinicalSession

###############################################################################
def test_session_pagination_has_stable_timestamp_and_id_order(persistence_session) -> None:  # type: ignore[no-untyped-def]
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(3):
        persistence_session.add(
            ClinicalSession(
                patient_name=f"Page {index}",
                session_timestamp=timestamp + timedelta(minutes=index),
            )
        )
    persistence_session.commit()

    statement = (
        select(ClinicalSession.id)
        .order_by(ClinicalSession.session_timestamp.desc(), ClinicalSession.id.desc())
        .offset(1)
        .limit(1)
    )
    page = persistence_session.scalars(statement).all()
    expected = persistence_session.scalar(
        select(ClinicalSession.id)
        .order_by(ClinicalSession.session_timestamp.desc(), ClinicalSession.id.desc())
        .offset(1)
        .limit(1)
    )
    assert page == [expected]
