from __future__ import annotations

import time
from datetime import date, datetime
from inspect import signature
from typing import Any

from domain.patient_timeline import (
    PatientTimeline,
    PatientTimelineEvent,
)
from repositories.schemas.base import Base
from repositories.schemas.clinical import ClinicalDrugMention, ClinicalSession
from repositories.schemas.knowledge import (
    Drug,
    DrugAlias,
    DrugRxnormCode,
    KbMatchCache,
    LiverToxMonograph,
)
from repository_fixtures import build_repository_graph
from services.clinical.knowledge import ClinicalKnowledgeComposer
from services.clinical.preparation import ClinicalKnowledgePreparation
from services.inspection import DataInspectionService
from services.llm.cloud import LLMError
from services.runtime.jobs import JobManager
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


###############################################################################
def build_repository_graph_for_test() -> tuple[Any, Any]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return build_repository_graph(engine=engine), engine


###############################################################################
def build_service(
    repository_graph: Any, *, jobs: JobManager, timeline_extractor: Any | None = None
) -> DataInspectionService:
    graph = build_repository_graph(
        engine=repository_graph.context.engine,
        session_factory=repository_graph.context.session_factory,
    )
    return DataInspectionService(
        clinical_session_repository=graph.clinical_session_repository,
        drug_catalog_repository=graph.drug_catalog_repository,
        knowledge_repository=graph.knowledge_repository,
        session_timeline_repository=graph.session_timeline_repository,
        session_revision_repository=graph.session_revision_repository,
        timeline_extractor=timeline_extractor,
        jobs=jobs,
    )


###############################################################################
def save_session(
    repository_graph: Any,
    *,
    patient_name: str,
    timestamp: datetime,
    status: str | None,
    report: str,
    anamnesis: str,
    payload: dict[str, Any] | None = None,
) -> None:
    repository_graph.clinical_session_repository.save_clinical_session(
        {
            "patient_name": patient_name,
            "session_timestamp": timestamp,
            "session_status": status,
            "anamnesis": anamnesis,
            "drugs": "acetaminophen",
            "final_report": report,
            "detected_drugs": ["acetaminophen"],
            "session_result_payload": payload
            or {
                "report": report,
                "issues": [],
            },
        }
    )


###############################################################################
def test_clinical_services_receive_only_their_repository_capabilities() -> None:
    repository_graph, _ = build_repository_graph_for_test()

    composer = ClinicalKnowledgeComposer(
        knowledge_repository=repository_graph.knowledge_repository
    )
    preparation = ClinicalKnowledgePreparation(
        knowledge_repository=repository_graph.knowledge_repository,
        drug_catalog_repository=repository_graph.drug_catalog_repository,
    )

    assert composer.knowledge_repository is repository_graph.knowledge_repository
    assert preparation.knowledge_repository is repository_graph.knowledge_repository
    assert (
        preparation.drug_catalog_repository is repository_graph.drug_catalog_repository
    )
    assert set(signature(ClinicalKnowledgeComposer).parameters) == {
        "knowledge_repository",
    }
    assert set(signature(ClinicalKnowledgePreparation).parameters) == {
        "knowledge_repository",
        "drug_catalog_repository",
    }


###############################################################################
def test_session_list_filters_and_search() -> None:
    repository_graph, _ = build_repository_graph_for_test()
    save_session(
        repository_graph,
        patient_name="Alice Example",
        timestamp=datetime(2025, 1, 1, 8, 30),
        status="successful",
        report="Alpha report",
        anamnesis="Mild alpha finding",
    )
    save_session(
        repository_graph,
        patient_name="Bob Failure",
        timestamp=datetime(2025, 1, 2, 8, 30),
        status="failed",
        report="Failure report",
        anamnesis="Critical warning",
    )
    save_session(
        repository_graph,
        patient_name="Carol Archive",
        timestamp=datetime(2025, 1, 3, 8, 30),
        status=None,
        report="Archive report",
        anamnesis="Unremarkable",
    )

    items, total = repository_graph.clinical_session_repository.list_sessions(
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

    items, total = repository_graph.clinical_session_repository.list_sessions(
        search=None,
        status_filter="successful",
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    assert total == 2
    assert {item["patient_name"] for item in items} == {
        "Alice Example",
        "Carol Archive",
    }

    items, total = repository_graph.clinical_session_repository.list_sessions(
        search="failure report",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["patient_name"] == "Bob Failure"

    items, total = repository_graph.clinical_session_repository.list_sessions(
        search=None,
        status_filter=None,
        date_mode="exact",
        filter_date=date(2025, 1, 2),
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["patient_name"] == "Bob Failure"

    items, total = repository_graph.clinical_session_repository.list_sessions(
        search=None,
        status_filter=None,
        date_mode="before",
        filter_date=date(2025, 1, 2),
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["patient_name"] == "Alice Example"

    items, total = repository_graph.clinical_session_repository.list_sessions(
        search=None,
        status_filter=None,
        date_mode="after",
        filter_date=date(2025, 1, 2),
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["patient_name"] == "Carol Archive"


###############################################################################
def test_session_report_and_text_use_result_payload_only() -> None:
    repository_graph, _ = build_repository_graph_for_test()
    repository_graph.clinical_session_repository.save_clinical_session(
        {
            "patient_name": "Payload Only",
            "session_timestamp": datetime(2025, 1, 1, 8, 30),
            "anamnesis": "Original anamnesis",
            "drugs": "acetaminophen",
            "final_report": "Section report",
            "session_result_payload": {
                "report": "Payload report",
                "original_session_text": "Payload session text",
            },
        }
    )
    repository_graph.clinical_session_repository.save_clinical_session(
        {
            "patient_name": "Payload Missing Text",
            "session_timestamp": datetime(2025, 1, 2, 8, 30),
            "anamnesis": "Section anamnesis",
            "drugs": "ibuprofen",
            "final_report": "Section report",
            "session_result_payload": {},
        }
    )

    items, _ = repository_graph.clinical_session_repository.list_sessions(
        search=None,
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    items_by_name = {item["patient_name"]: item for item in items}
    assert items_by_name["Payload Only"]["has_report"] is True
    assert items_by_name["Payload Missing Text"]["has_report"] is False

    payload_detail = repository_graph.clinical_session_repository.get_session_detail(
        int(items_by_name["Payload Only"]["session_id"])
    )
    assert payload_detail is not None
    assert payload_detail["report"] == "Payload report"
    assert payload_detail["session_text"] == "Payload session text"

    missing_text_detail = (
        repository_graph.clinical_session_repository.get_session_detail(
            int(items_by_name["Payload Missing Text"]["session_id"])
        )
    )
    assert missing_text_detail is not None
    assert missing_text_detail["report"] is None
    assert missing_text_detail["session_text"] == ""
    assert missing_text_detail["sections"]["anamnesis"] == "Section anamnesis"


###############################################################################
def test_catalog_search_and_drug_delete_cleanup() -> None:
    repository_graph, engine = build_repository_graph_for_test()
    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db_session:
        drug = repository_graph.drug_catalog_repository.ensure_drug(
            db_session,
            canonical_name="Acetaminophen",
            canonical_name_norm="acetaminophen",
            rxnorm_rxcui="161",
            livertox_nbk_id="NBK100",
            rxnav_last_update="2025-01-05",
        )
        repository_graph.drug_catalog_repository.upsert_drug_alias(
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
                monograph_key="acetaminophen|NBK100|unit",
                drug_name_norm="acetaminophen",
                nbk_id="NBK100",
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
            ClinicalDrugMention(
                session_id=int(clinical_session.id),
                mention_ordinal=0,
                raw_name="Acetaminophen",
                normalized_name="acetaminophen",
                drug_id=int(drug.id),
                match_status="matched",
            )
        )
        db_session.commit()

    rxnav_items, rxnav_total = (
        repository_graph.drug_catalog_repository.list_rxnav_catalog(
            search="tylenol",
            offset=0,
            limit=10,
        )
    )
    assert rxnav_total == 1
    assert rxnav_items[0]["drug_name"] == "Acetaminophen"

    livertox_items, livertox_total = (
        repository_graph.knowledge_repository.list_livertox_catalog(
            search="injury",
            offset=0,
            limit=10,
        )
    )
    assert livertox_total == 1
    assert livertox_items[0]["drug_name"] == "Acetaminophen"

    aliases = repository_graph.drug_catalog_repository.get_rxnav_alias_groups(
        rxnav_items[0]["drug_id"]
    )
    assert aliases is not None
    sources = {group["source"] for group in aliases["groups"]}
    assert "rxnorm" in sources

    excerpt = repository_graph.knowledge_repository.get_livertox_excerpt(
        rxnav_items[0]["drug_id"]
    )
    assert excerpt is not None
    assert "injury" in excerpt["excerpt"]

    assert (
        repository_graph.drug_catalog_repository.delete_drug_with_cleanup(
            rxnav_items[0]["drug_id"]
        )
        is True
    )

    with session_factory() as db_session:
        assert db_session.execute(select(Drug)).scalars().all() == []
        assert db_session.execute(select(DrugAlias)).scalars().all() == []
        assert db_session.execute(select(DrugRxnormCode)).scalars().all() == []
        assert db_session.execute(select(LiverToxMonograph)).scalars().all() == []
        assert db_session.execute(select(KbMatchCache)).scalars().all() == []
        mentions = db_session.execute(select(ClinicalDrugMention)).scalars().all()
        assert len(mentions) == 1
        assert mentions[0].drug_id is None


###############################################################################
def test_update_job_lifecycle_with_cooperative_cancel() -> None:
    repository_graph, _ = build_repository_graph_for_test()
    jobs = JobManager()
    service = build_service(repository_graph, jobs=jobs)

    def fast_rxnav_runner(
        job_id: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = overrides
        jobs.update_progress(job_id, 50)
        jobs.update_result(job_id, {"progress_message": "halfway"})
        return {"summary": {"records": 2}}

    def slow_livertox_runner(
        job_id: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = overrides
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
        payload = service.get_job_status(
            rxnav_job_id, expected_type=service.RXNAV_JOB_TYPE
        )
        if payload and payload["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    final_rxnav = service.get_job_status(
        rxnav_job_id, expected_type=service.RXNAV_JOB_TYPE
    )
    assert final_rxnav is not None
    assert final_rxnav["status"] == "completed"

    started = service.start_update_job(service.LIVERTOX_JOB_TYPE)
    livertox_job_id = str(started["job_id"])
    assert (
        service.cancel_job(livertox_job_id, expected_type=service.LIVERTOX_JOB_TYPE)
        is True
    )
    for _ in range(120):
        payload = service.get_job_status(
            livertox_job_id, expected_type=service.LIVERTOX_JOB_TYPE
        )
        if payload and payload["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    final_livertox = service.get_job_status(
        livertox_job_id,
        expected_type=service.LIVERTOX_JOB_TYPE,
    )
    assert final_livertox is not None
    assert final_livertox["status"] == "cancelled"


###############################################################################
class FakeTimelineExtractor:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.call_count = 0
        self.last_runtime_settings: dict[str, Any] | None = None

    # -------------------------------------------------------------------------
    async def extract_timeline(
        self,
        *,
        session_id: int,
        source_payload: dict[str, Any],
        runtime_settings: dict[str, Any] | None = None,
    ) -> PatientTimeline:
        _ = source_payload
        self.call_count += 1
        self.last_runtime_settings = runtime_settings
        return PatientTimeline(
            session_id=session_id,
            generated_at=datetime(2026, 1, 1, 12, 0),
            events=[
                PatientTimelineEvent(
                    event_id=f"event-{session_id}-1",
                    title="Therapy started",
                    event_type="therapy",
                    event_date="2025-01-10",
                    source="drugs",
                )
            ],
        )


###############################################################################
class FailingTimelineExtractor:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.timeout_s = 1.0

    # -------------------------------------------------------------------------
    async def extract_timeline(
        self,
        *,
        session_id: int,
        source_payload: dict[str, Any],
        runtime_settings: dict[str, Any] | None = None,
    ) -> PatientTimeline:
        _ = session_id, source_payload, runtime_settings
        raise LLMError("structured extraction failed", error_code="invalid_response")


###############################################################################
def test_timeline_generation_persists_history_and_reuses_latest_when_not_forced() -> (
    None
):
    repository_graph, _ = build_repository_graph_for_test()
    save_session(
        repository_graph,
        patient_name="Timeline Patient",
        timestamp=datetime(2025, 1, 1, 8, 30),
        status="successful",
        report="Timeline report",
        anamnesis="Symptoms started in January 2025.",
    )
    session_rows, _ = repository_graph.clinical_session_repository.list_sessions(
        search="Timeline Patient",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    session_id = int(session_rows[0]["session_id"])
    extractor = FakeTimelineExtractor()
    service = build_service(
        repository_graph, timeline_extractor=extractor, jobs=JobManager()
    )

    generated = service.generate_session_timeline(session_id)
    assert generated is not None
    assert generated.timeline_id is not None
    assert generated.events[0].title == "Therapy started"
    assert extractor.call_count == 1
    assert extractor.last_runtime_settings is not None
    assert "text_extraction_model" in extractor.last_runtime_settings
    assert "clinical_model" in extractor.last_runtime_settings

    previews = repository_graph.session_timeline_repository.list_session_timelines(
        session_id
    )
    assert len(previews) == 1
    assert previews[0]["timeline_id"] == generated.timeline_id
    assert previews[0]["event_count"] == 1

    persisted = (
        repository_graph.session_timeline_repository.get_session_timeline_record(
            session_id, int(generated.timeline_id)
        )
    )
    assert persisted is not None
    assert persisted["timeline_id"] == generated.timeline_id

    cached = service.get_session_timeline(session_id)
    assert cached is not None
    assert cached.timeline_id == generated.timeline_id
    assert cached.events[0].event_type == "therapy"

    reused = service.generate_session_timeline(session_id)
    assert reused is not None
    assert reused.timeline_id == generated.timeline_id
    assert reused.events[0].title == "Therapy started"
    assert extractor.call_count == 1

    regenerated = service.generate_session_timeline(session_id, force_regenerate=True)
    assert regenerated is not None
    assert regenerated.timeline_id is not None
    assert regenerated.timeline_id != generated.timeline_id
    assert extractor.call_count == 2

    history = repository_graph.session_timeline_repository.list_session_timelines(
        session_id
    )
    assert len(history) == 2
    assert history[0]["timeline_id"] == regenerated.timeline_id
    assert history[1]["timeline_id"] == generated.timeline_id


###############################################################################
def test_timeline_generation_marks_fallback_payload() -> None:
    repository_graph, _ = build_repository_graph_for_test()
    save_session(
        repository_graph,
        patient_name="Fallback Timeline Patient",
        timestamp=datetime(2025, 1, 1, 8, 30),
        status="successful",
        report="Fallback timeline report",
        anamnesis="Symptoms started in January 2025.",
    )
    session_rows, _ = repository_graph.clinical_session_repository.list_sessions(
        search="Fallback Timeline Patient",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    session_id = int(session_rows[0]["session_id"])
    service = build_service(
        repository_graph,
        timeline_extractor=FailingTimelineExtractor(),
        jobs=JobManager(),
    )

    generated = service.generate_session_timeline(session_id, force_regenerate=True)
    assert generated is not None
    assert generated.timeline_id is not None
    assert generated.generation_status == "fallback"
    assert generated.generation_note is not None
    assert generated.generation_error_code == "invalid_response"
    assert "invalid structured data" in generated.generation_note
    assert generated.events
    assert {event.source for event in generated.events} == {"fallback_parser"}

    cached = service.get_session_timeline(session_id)
    assert cached is not None
    assert cached.timeline_id == generated.timeline_id
    assert cached.generation_status == "fallback"

    history = repository_graph.session_timeline_repository.list_session_timelines(
        session_id
    )
    assert len(history) == 1
    assert history[0]["timeline_id"] == generated.timeline_id
    assert history[0]["generation_status"] == "fallback"
    assert history[0]["generation_error_code"] == "invalid_response"
    assert all(event.event_date is None for event in generated.events)
    assert all(event.extracted_timing_text is None for event in generated.events)
    assert all(event.timing_type == "uncertain" for event in generated.events)


###############################################################################
def test_timeline_generation_does_not_mutate_persisted_runtime_settings() -> None:
    repository_graph, _ = build_repository_graph_for_test()
    original_runtime_settings = {
        "use_cloud_services": False,
        "llm_provider": "openai",
        "text_extraction_model": "baseline-parser",
        "clinical_model": "baseline-clinical",
    }
    save_session(
        repository_graph,
        patient_name="Stable Runtime Patient",
        timestamp=datetime(2025, 1, 1, 8, 30),
        status="successful",
        report="Stable runtime report",
        anamnesis="Stable runtime context.",
        payload={
            "report": "Stable runtime report",
            "issues": [],
            "runtime_settings": original_runtime_settings,
        },
    )
    session_rows, _ = repository_graph.clinical_session_repository.list_sessions(
        search="Stable Runtime Patient",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    session_id = int(session_rows[0]["session_id"])
    service = build_service(
        repository_graph, timeline_extractor=FakeTimelineExtractor(), jobs=JobManager()
    )

    _ = service.generate_session_timeline(session_id, force_regenerate=True)

    source_after = (
        repository_graph.session_timeline_repository.get_session_timeline_source(
            session_id
        )
    )
    assert source_after is not None
    assert (
        source_after["session_result_payload"]["runtime_settings"]
        == original_runtime_settings
    )


###############################################################################
def test_timeline_generation_passes_persisted_opencode_go_settings_to_extractor() -> (
    None
):
    repository_graph, _ = build_repository_graph_for_test()
    persisted_runtime_settings = {
        "use_cloud_services": True,
        "llm_provider": "opencode_go",
        "cloud_model": "deepseek-v4-flash",
        "text_extraction_model": "deepseek-v4-flash",
        "clinical_model": "deepseek-v4-flash",
    }
    save_session(
        repository_graph,
        patient_name="OpenCode Go Timeline Patient",
        timestamp=datetime(2025, 1, 1, 8, 30),
        status="successful",
        report="OpenCode Go timeline report",
        anamnesis="OpenCode Go timeline context.",
        payload={
            "report": "OpenCode Go timeline report",
            "issues": [],
            "runtime_settings": persisted_runtime_settings,
        },
    )
    session_rows, _ = repository_graph.clinical_session_repository.list_sessions(
        search="OpenCode Go Timeline Patient",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    session_id = int(session_rows[0]["session_id"])
    extractor = FakeTimelineExtractor()
    service = build_service(
        repository_graph, timeline_extractor=extractor, jobs=JobManager()
    )

    generated = service.generate_session_timeline(session_id, force_regenerate=True)

    assert generated is not None
    assert extractor.last_runtime_settings is not None
    assert extractor.last_runtime_settings["use_cloud_services"] is True
    assert extractor.last_runtime_settings["llm_provider"] == "opencode_go"
    assert extractor.last_runtime_settings["cloud_model"] == "deepseek-v4-flash"


###############################################################################
def test_session_payload_timeline_is_not_read_as_history_record() -> None:
    repository_graph, _ = build_repository_graph_for_test()
    payload_timeline = PatientTimeline(
        session_id=1,
        generated_at=datetime(2025, 1, 2, 9, 15),
        generation_status="llm_generated",
        events=[
            PatientTimelineEvent(
                event_id="payload-1",
                title="Payload-only timeline event",
                event_type="other",
                timing_type="relative",
                sort_order=10,
            )
        ],
    )
    save_session(
        repository_graph,
        patient_name="Payload Timeline Patient",
        timestamp=datetime(2025, 1, 1, 8, 30),
        status="successful",
        report="Payload timeline report",
        anamnesis="Payload timeline context.",
        payload={"patient_timeline": payload_timeline.model_dump(mode="json")},
    )
    session_rows, _ = repository_graph.clinical_session_repository.list_sessions(
        search="Payload Timeline Patient",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    session_id = int(session_rows[0]["session_id"])
    service = build_service(
        repository_graph, timeline_extractor=FakeTimelineExtractor(), jobs=JobManager()
    )

    latest = service.get_session_timeline(session_id)
    assert latest is None

    previews = repository_graph.session_timeline_repository.list_session_timelines(
        session_id
    )
    assert previews == []
