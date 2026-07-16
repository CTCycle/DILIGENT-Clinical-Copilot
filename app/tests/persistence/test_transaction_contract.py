from __future__ import annotations

from sqlalchemy import select

from repositories.schemas.configuration import ApplicationConfiguration

###############################################################################
def test_configuration_singleton_rolls_back(persistence_session) -> None:  # type: ignore[no-untyped-def]
    persistence_session.add(
        ApplicationConfiguration(payload={"clinical_model": "contract"})
    )
    persistence_session.rollback()
    assert persistence_session.scalar(select(ApplicationConfiguration)) is None

###############################################################################
def test_configuration_singleton_is_unique(persistence_session) -> None:  # type: ignore[no-untyped-def]
    persistence_session.add(
        ApplicationConfiguration(payload={"clinical_model": "contract"})
    )
    persistence_session.commit()
    persistence_session.add(
        ApplicationConfiguration(payload={"clinical_model": "duplicate"})
    )
    try:
        persistence_session.commit()
    except Exception:
        persistence_session.rollback()
    assert persistence_session.scalar(select(ApplicationConfiguration)).payload == {
        "clinical_model": "contract"
    }
