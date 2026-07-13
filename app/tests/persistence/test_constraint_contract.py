from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from repositories.schemas.models import (
    ApplicationConfiguration,
    Drug,
    DrugIdentifier,
)


###############################################################################
def test_singleton_and_identifier_constraints_are_enforced(persistence_session) -> None:  # type: ignore[no-untyped-def]
    persistence_session.add(ApplicationConfiguration(payload={"revision": 1}))
    persistence_session.commit()

    persistence_session.add(ApplicationConfiguration(payload={"revision": 2}))
    with pytest.raises(IntegrityError):
        persistence_session.commit()
    persistence_session.rollback()

    drug = Drug(canonical_name="Constraint Drug", canonical_name_norm="constraint drug")
    persistence_session.add(drug)
    persistence_session.flush()
    persistence_session.add_all(
        [
            DrugIdentifier(
                drug_id=drug.id,
                identifier_system="rxnorm",
                identifier_value="constraint-1",
            ),
            DrugIdentifier(
                drug_id=drug.id,
                identifier_system="rxnorm",
                identifier_value="constraint-1",
            ),
        ]
    )
    with pytest.raises(IntegrityError):
        persistence_session.commit()
