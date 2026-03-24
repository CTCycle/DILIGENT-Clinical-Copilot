from __future__ import annotations

import time
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.repositories.schemas.models import (
    Base,
    ClinicalSession,
    ClinicalSessionDrug,
    Drug,
    DrugAlias,
    DrugRxnormCode,
    LiverToxMonograph,
)
from DILIGENT.server.repositories.serialization.data import DataSerializer
from DILIGENT.server.services.inspection import DataInspectionService
from DILIGENT.server.services.jobs import JobManager


###############################################################################
class QueryStub:
    def __init__(self, engine: Any) -> None:
        self.database = SimpleNamespace(backend=SimpleNamespace(engine=engine))


# -----------------------------------------------------------------------------
def build_serializer() -> tuple[DataSerializer, Any]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(queries=QueryStub(engine))
    return serializer, engine


# -----------------------------------------------------------------------------
def save_session(
    serializer: DataSerializer,
    *,
    patient_name: str,
    timestamp: datetime,
    status: str | None,
    report: str,
    anamnesis: str,
) -> None:
    serializer.save_clinical_session(
        {
            "patient_name": patient_name,
            "session_timestamp": timestamp,
            "session_status": status,
            "anamnesis": anamnesis,
            "drugs": "acetaminophen",
            "final_report": report,
            "detected_drugs": ["acetaminophen"],
            "session_result_payload": {
                "report": report,
                "issues": [],
            },
        }
    )


# -----------------------------------------------------------------------------
def test_session_list_filters_and_search() -> None:
    serializer, _ = build_serializer()
    save_session(
        serializer,
        patient_name="Alice Example",
        timestamp=datetime(2025, 1, 1, 8, 30),
        status="successful",
        report="Alpha report",
        anamnesis="Mild alpha finding",
    )
    save_session(
        serializer,
        patient_name="Bob Failure",
        timestamp=datetime(2025, 1, 2, 8, 30),
        status="failed",
        report="Failure report",
        anamnesis="Critical warning",
    )
    save_session(
        serializer,
        patient_name="Carol Legacy",
        timestamp=datetime(2025, 1, 3, 8, 30),
        status=None,
        report="Legacy report",
        anamnesis="Unremarkable",
    )

    items, total = serializer.list_sessions(
        search="bob",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["patient_name"] == "Bob Failure"
    assert items[0]["status"] == "failed"

    items, total = serializer.list_sessions(
        search=None,
        status_filter="successful",
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    assert total == 2
    assert {item["patient_name"] for item in items} == {"Alice Example", "Carol Legacy"}

    items, total = serializer.list_sessions(
        search="failure report",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["patient_name"] == "Bob Failure"

    items, total = serializer.list_sessions(
        search=None,
        status_filter=None,
        date_mode="exact",
        filter_date=date(2025, 1, 2),
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["patient_name"] == "Bob Failure"

    items, total = serializer.list_sessions(
        search=None,
        status_filter=None,
        date_mode="before",
        filter_date=date(2025, 1, 2),
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["patient_name"] == "Alice Example"

    items, total = serializer.list_sessions(
        search=None,
        status_filter=None,
        date_mode="after",
        filter_date=date(2025, 1, 2),
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["patient_name"] == "Carol Legacy"


# -----------------------------------------------------------------------------
def test_catalog_search_and_drug_delete_cleanup() -> None:
    serializer, engine = build_serializer()
    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db_session:
        drug = serializer.ensure_drug(
            db_session,
            canonical_name="Acetaminophen",
            canonical_name_norm="acetaminophen",
            rxnorm_rxcui="161",
            livertox_nbk_id="NBK100",
            rxnav_last_update="2025-01-05",
        )
        serializer.upsert_drug_alias(
            db_session,
            drug_id=int(drug.id),
            alias="Tylenol",
            alias_kind="synonym",
            source="rxnorm",
            term_type="SCD",
        )
        db_session.add(
            LiverToxMonograph(
                drug_id=int(drug.id),
                excerpt="Severe liver injury risk profile",
                last_update="2025-01-06",
            )
        )
        clinical_session = ClinicalSession(
            patient_name="Drug Link",
            session_timestamp=datetime(2025, 1, 4, 10, 0),
            session_status="successful",
        )
        db_session.add(clinical_session)
        db_session.flush()
        db_session.add(
            ClinicalSessionDrug(
                session_id=int(clinical_session.id),
                raw_drug_name="Acetaminophen",
                raw_drug_name_norm="acetaminophen",
                drug_id=int(drug.id),
            )
        )
        db_session.commit()

    rxnav_items, rxnav_total = serializer.list_rxnav_catalog(
        search="tylenol",
        offset=0,
        limit=10,
    )
    assert rxnav_total == 1
    assert rxnav_items[0]["drug_name"] == "Acetaminophen"

    livertox_items, livertox_total = serializer.list_livertox_catalog(
        search="injury",
        offset=0,
        limit=10,
    )
    assert livertox_total == 1
    assert livertox_items[0]["drug_name"] == "Acetaminophen"

    aliases = serializer.get_rxnav_alias_groups(rxnav_items[0]["drug_id"])
    assert aliases is not None
    sources = {group["source"] for group in aliases["groups"]}
    assert "rxnorm" in sources

    excerpt = serializer.get_livertox_excerpt(rxnav_items[0]["drug_id"])
    assert excerpt is not None
    assert "injury" in excerpt["excerpt"]

    assert serializer.delete_drug_with_cleanup(rxnav_items[0]["drug_id"]) is True

    with session_factory() as db_session:
        assert db_session.execute(select(Drug)).scalars().all() == []
        assert db_session.execute(select(DrugAlias)).scalars().all() == []
        assert db_session.execute(select(DrugRxnormCode)).scalars().all() == []
        assert db_session.execute(select(LiverToxMonograph)).scalars().all() == []
        session_drugs = db_session.execute(select(ClinicalSessionDrug)).scalars().all()
        assert len(session_drugs) == 1
        assert session_drugs[0].drug_id is None


# -----------------------------------------------------------------------------
def test_update_job_lifecycle_with_cooperative_cancel() -> None:
    serializer, _ = build_serializer()
    jobs = JobManager()
    service = DataInspectionService(serializer=serializer, jobs=jobs)

    def fast_rxnav_runner(job_id: str) -> dict[str, Any]:
        jobs.update_progress(job_id, 50)
        jobs.update_result(job_id, {"progress_message": "halfway"})
        return {"summary": {"records": 2}}

    def slow_livertox_runner(job_id: str) -> dict[str, Any]:
        for _ in range(120):
            if jobs.should_stop(job_id):
                return {}
            time.sleep(0.005)
        return {"summary": {"records": 4}}

    service.run_rxnav_update_job = fast_rxnav_runner  # type: ignore[method-assign]
    service.run_livertox_update_job = slow_livertox_runner  # type: ignore[method-assign]

    started = service.start_update_job(service.RXNAV_JOB_TYPE)
    rxnav_job_id = str(started["job_id"])
    for _ in range(80):
        payload = service.get_job_status(rxnav_job_id, expected_type=service.RXNAV_JOB_TYPE)
        if payload and payload["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    final_rxnav = service.get_job_status(rxnav_job_id, expected_type=service.RXNAV_JOB_TYPE)
    assert final_rxnav is not None
    assert final_rxnav["status"] == "completed"

    started = service.start_update_job(service.LIVERTOX_JOB_TYPE)
    livertox_job_id = str(started["job_id"])
    assert service.cancel_job(livertox_job_id, expected_type=service.LIVERTOX_JOB_TYPE) is True
    for _ in range(120):
        payload = service.get_job_status(livertox_job_id, expected_type=service.LIVERTOX_JOB_TYPE)
        if payload and payload["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    final_livertox = service.get_job_status(
        livertox_job_id,
        expected_type=service.LIVERTOX_JOB_TYPE,
    )
    assert final_livertox is not None
    assert final_livertox["status"] == "cancelled"
