from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from repositories.schemas.configuration import ApplicationConfiguration
from repositories.schemas.knowledge import (
    Drug,
    DrugIdentifier,
)

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

###############################################################################
def test_drug_identifier_composite_unique_constraint(persistence_session) -> None:  # type: ignore[no-untyped-def]
    drug = Drug(canonical_name="Test Drug", canonical_name_norm="test drug")
    persistence_session.add(drug)
    persistence_session.flush()
    persistence_session.add_all(
        [
            DrugIdentifier(
                drug_id=drug.id,
                identifier_system="rxnorm",
                identifier_value="test-1",
            ),
            DrugIdentifier(
                drug_id=drug.id,
                identifier_system="rxnorm",
                identifier_value="test-1",
            ),
        ]
    )
    with pytest.raises(IntegrityError):
        persistence_session.commit()
