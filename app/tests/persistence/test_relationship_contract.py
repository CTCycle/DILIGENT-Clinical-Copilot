from __future__ import annotations

from sqlalchemy import select

from repositories.schemas.clinical import (
    ClinicalDrugMention,
    ClinicalLabObservation,
    ClinicalSession,
    ClinicalSessionVersion,
)
from repositories.schemas.knowledge import (
    Drug,
    DrugIdentifier,
)


###############################################################################
def test_session_children_cascade_and_drug_mentions_set_null(
    persistence_session,
) -> None:  # type: ignore[no-untyped-def]
    drug = Drug(canonical_name="Contract Drug", canonical_name_norm="contract drug")
    persistence_session.add(drug)
    persistence_session.flush()
    session = ClinicalSession(patient_name="Contract Patient")
    persistence_session.add(session)
    persistence_session.flush()
    persistence_session.add_all(
        [
            ClinicalSessionVersion(
                session_id=session.id,
                root_session_id=session.id,
                version_number=1,
                version_status="current",
                revision_kind="original",
                llm_qa_status="not_run",
                clinical_review_status="not_reviewed",
            ),
            ClinicalLabObservation(
                session_id=session.id,
                marker_code="ALT",
                value_numeric=42.0,
            ),
            ClinicalDrugMention(
                session_id=session.id,
                mention_ordinal=0,
                raw_name="Contract Drug",
                normalized_name="contract drug",
                drug_id=drug.id,
                match_status="matched",
            ),
            DrugIdentifier(
                drug_id=drug.id,
                identifier_system="rxnorm",
                identifier_value="123",
            ),
        ]
    )
    persistence_session.commit()

    persistence_session.delete(drug)
    persistence_session.commit()
    mention = persistence_session.scalar(select(ClinicalDrugMention))
    assert mention is not None
    assert mention.drug_id is None
    assert persistence_session.scalar(select(DrugIdentifier)) is None

    persistence_session.delete(session)
    persistence_session.commit()
    assert persistence_session.scalar(select(ClinicalLabObservation)) is None
    assert persistence_session.scalar(select(ClinicalSessionVersion)) is None
    assert persistence_session.scalar(select(ClinicalDrugMention)) is None
