from __future__ import annotations

from repositories.schemas.models import (
    Base,
    ClinicalSession,
    Drug,
    DrugAlias,
    Patient,
    ReferenceCatalogEntry,
)
from repositories.serialization.data import DataSerializer
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


def test_session_learning_promotes_only_direct_high_confidence_aliases() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    serializer = DataSerializer(engine=engine, session_factory=factory)

    db_session = factory()
    try:
        atorvastatin = Drug(
            canonical_name="atorvastatin",
            canonical_name_norm="atorvastatin",
            rxnorm_rxcui=None,
            livertox_nbk_id=None,
        )
        loop_diuretics = Drug(
            canonical_name="Loop Diuretics",
            canonical_name_norm="loop diuretics",
            rxnorm_rxcui=None,
            livertox_nbk_id=None,
        )
        patient = Patient(name="Test", visit_date=None)
        db_session.add_all([atorvastatin, loop_diuretics, patient])
        db_session.flush()
        session = ClinicalSession(
            patient_id=int(patient.id), session_status="successful"
        )
        db_session.add(session)
        db_session.flush()

        serializer.persist_session_drugs(
            db_session,
            int(session.id),
            {
                "matched_drugs": [
                    {
                        "raw_drug_name": "Atorvastatina",
                        "matched_drug_name": "atorvastatin",
                        "match_reason": "exact_canonical",
                        "match_confidence": 1.0,
                    },
                    {
                        "raw_drug_name": "Furosemide",
                        "matched_drug_name": "Loop Diuretics",
                        "match_reason": "exact_alias_ranked",
                        "match_confidence": 0.92,
                    },
                    {
                        "raw_drug_name": "Unknown Herb",
                        "matched_drug_name": None,
                        "match_reason": "no_match",
                        "match_confidence": None,
                    },
                ]
            },
        )
        db_session.commit()

        aliases = db_session.execute(
            select(DrugAlias.alias, DrugAlias.drug_id).where(
                DrugAlias.source == "session"
            )
        ).all()
        terms = db_session.execute(
            select(ReferenceCatalogEntry.category, ReferenceCatalogEntry.value).where(
                ReferenceCatalogEntry.manifest == "runtime_observations"
            )
        ).all()
    finally:
        db_session.close()

    assert aliases == [("Atorvastatina", int(atorvastatin.id))]
    assert sorted(terms) == [
        ("observed_unpromoted_query", "Furosemide"),
        ("observed_unresolved_query", "Unknown Herb"),
    ]

