# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app as server_app_module
from domain.clinical.robustness import ClinicalInputPreflightResult
from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository
from services.llm.runtime_config import LLMRuntimeConfig
from services.runtime.jobs import get_job_manager
from services.session.scheduler import get_clinical_scheduler
from services.session.session_service import ClinicalSessionService

TERMINAL = {"completed", "failed", "cancelled"}


###############################################################################
def clinical_input(marker: str) -> str:
    return (
        "## Anamnesis\n"
        f"Patient {marker} reports fatigue, pruritus, dark urine, mild nausea, and "
        "recent appetite reduction without alcohol excess, viral prodrome, travel "
        "exposure, hypotension, biliary colic, or known chronic liver disease. "
        "Symptoms began gradually after medication exposure and are documented "
        "here only as lifecycle test content.\n\n"
        "## Therapy\n"
        "Amoxicillin 500 mg three times daily for seven days.\n\n"
        "## Laboratory Analysis\n"
        "ALT 220 U/L with ULN 50, ALP 180 U/L with ULN 120."
    )


###############################################################################
def launch_payload(marker: str) -> dict[str, Any]:
    return {
        "name": f"Patient {marker}",
        "visit_date": "2026-06-19",
        "clinical_input": clinical_input(marker),
        "evidence_sources": ["livertox"],
    }


###############################################################################
class FakeAnalyses:
    """Stand-in pipeline whose runs can be held open and released per patient."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.gates: dict[str, threading.Event] = {}
        self.started: list[str] = []
        self.lock = threading.Lock()
        self.released = False

    # -------------------------------------------------------------------------
    def gate(self, marker: str) -> threading.Event:
        with self.lock:
            event = self.gates.setdefault(marker, threading.Event())
            if self.released:
                event.set()
            return event

    # -------------------------------------------------------------------------
    def release_all(self) -> None:
        with self.lock:
            self.released = True
            for event in self.gates.values():
                event.set()

    # -------------------------------------------------------------------------
    async def process(
        self, service: ClinicalSessionService, payload: Any, **kwargs: Any
    ) -> dict[str, Any]:
        marker = str(payload.name).removeprefix("Patient ")
        with self.lock:
            self.started.append(marker)
        progress = kwargs["progress_callback"]
        stop_check = kwargs["stop_check"]
        progress("clinical", 16.0, "drugs.extracting")
        gate = self.gate(marker)
        while not gate.wait(0.02):
            stop_check()
        # The service and its diagnostics belong to this run only.
        service.stage_elapsed_ms[marker] = 1
        progress("clinical", 94.0, "report.generating")
        result_payload = {
            "report": f"Report for {marker}",
            "diagnostics": dict(service.stage_elapsed_ms),
        }
        await asyncio.to_thread(
            service.session_repository.complete_clinical_session,
            kwargs["session_id"],
            {
                "patient_name": payload.name,
                "patient_visit_date": payload.visit_date,
                "session_status": "completed",
                "session_kind": "standard_assessment",
                "final_report": result_payload["report"],
                "session_result_payload": result_payload,
            },
        )
        return result_payload


###############################################################################
@pytest.fixture
def analyses(monkeypatch) -> FakeAnalyses:
    fake = FakeAnalyses()
    monkeypatch.setattr(
        ClinicalSessionService, "apply_persisted_runtime_configuration", lambda self: None
    )
    monkeypatch.setattr(
        ClinicalSessionService, "ensure_submission_requirements", lambda self, p: None
    )
    monkeypatch.setattr(
        LLMRuntimeConfig, "is_cloud_enabled", classmethod(lambda cls: False)
    )
    monkeypatch.setattr(
        ClinicalSessionService,
        "validate_clinical_input",
        lambda self, request: ClinicalInputPreflightResult(ready=True),
    )
    monkeypatch.setattr(
        KnowledgeRepository,
        "list_livertox_catalog",
        lambda self, search, offset, limit: ([{"drug_name": "Amoxicillin"}], 1),
    )
    monkeypatch.setattr(
        DrugCatalogRepository,
        "list_rxnav_catalog",
        lambda self, search, offset, limit: ([{"drug_name": "Amoxicillin"}], 1),
    )
    async def process_single_patient(
        service: ClinicalSessionService, payload: Any, **kwargs: Any
    ) -> dict[str, Any]:
        return await fake.process(service, payload, **kwargs)

    monkeypatch.setattr(
        ClinicalSessionService, "process_single_patient", process_single_patient
    )
    return fake


###############################################################################
@pytest.fixture
def client(analyses: FakeAnalyses, monkeypatch) -> Iterator[TestClient]:
    scheduler = get_clinical_scheduler()
    monkeypatch.setattr(scheduler, "limit_provider", lambda: 2)
    with TestClient(server_app_module.app, raise_server_exceptions=False) as test_client:
        yield test_client
        analyses.release_all()
        wait_until(lambda: not get_job_manager().is_job_running("clinical"))


###############################################################################
def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition not reached in time")


###############################################################################
def create_draft(client: TestClient, marker: str) -> dict[str, Any]:
    response = client.post(
        "/api/clinical/sessions",
        json={"patient_name": f"Patient {marker}", "clinical_input": clinical_input(marker)},
    )
    assert response.status_code == 201, response.text
    return response.json()


###############################################################################
def launch(client: TestClient, marker: str) -> int:
    session_id = create_draft(client, marker)["session_id"]
    response = client.post(
        f"/api/clinical/sessions/{session_id}/launch", json=launch_payload(marker)
    )
    assert response.status_code == 202, response.text
    return session_id


###############################################################################
def status_of(client: TestClient, session_id: int) -> dict[str, Any]:
    response = client.get(f"/api/clinical/sessions/{session_id}/status")
    assert response.status_code == 200, response.text
    return response.json()


###############################################################################
def wait_for_status(
    client: TestClient, session_id: int, expected: set[str]
) -> dict[str, Any]:
    wait_until(lambda: status_of(client, session_id)["status"] in expected)
    return status_of(client, session_id)


###############################################################################
def test_draft_edit_and_launch_reuse_one_session(client, analyses) -> None:
    draft = create_draft(client, "A")
    session_id = draft["session_id"]
    assert draft["status"] == "draft"

    updated = client.put(
        f"/api/clinical/sessions/{session_id}/input",
        json={
            "patient_name": "Patient A",
            "clinical_input": clinical_input("A") + "\nEdited.",
            "expected_updated_at": draft["updated_at"],
        },
    )
    assert updated.status_code == 200, updated.text
    stale = client.put(
        f"/api/clinical/sessions/{session_id}/input",
        json={"clinical_input": "x", "expected_updated_at": draft["updated_at"]},
    )
    assert stale.status_code == 409

    launched = client.post(
        f"/api/clinical/sessions/{session_id}/launch", json=launch_payload("A")
    )
    assert launched.status_code == 202, launched.text
    duplicate = client.post(
        f"/api/clinical/sessions/{session_id}/launch", json=launch_payload("A")
    )
    assert duplicate.status_code == 409
    edit_running = client.put(
        f"/api/clinical/sessions/{session_id}/input",
        json={"clinical_input": "x", "expected_updated_at": updated.json()["updated_at"]},
    )
    assert edit_running.status_code == 409

    analyses.gate("A").set()
    final = wait_for_status(client, session_id, TERMINAL)
    assert final["status"] == "completed"
    assert final["has_report"] is True
    assert final["progress"] == 100.0
    assert analyses.started.count("A") == 1
    detail = client.get(f"/api/inspection/sessions/{session_id}").json()
    assert detail["report"] == "Report for A"


###############################################################################
def test_limit_queues_sessions_and_admits_them_in_order(client, analyses, monkeypatch) -> None:
    monkeypatch.setattr(get_clinical_scheduler(), "limit_provider", lambda: 1)
    first, second, third = (launch(client, marker) for marker in ("Q1", "Q2", "Q3"))
    wait_for_status(client, first, {"running"})
    assert status_of(client, second)["status"] == "queued"
    assert status_of(client, second)["queue_position"] == 1
    assert status_of(client, third)["queue_position"] == 2

    analyses.gate("Q1").set()
    wait_for_status(client, first, {"completed"})
    wait_for_status(client, second, {"running"})
    assert status_of(client, third)["status"] == "queued"

    # Raising the limit admits waiting work without a restart.
    monkeypatch.setattr(get_clinical_scheduler(), "limit_provider", lambda: 2)
    get_clinical_scheduler().dispatch()
    wait_for_status(client, third, {"running"})
    wait_until(lambda: len(analyses.started) == 3)
    assert analyses.started == ["Q1", "Q2", "Q3"]


###############################################################################
def test_cancelling_one_session_leaves_others_running(client, analyses, monkeypatch) -> None:
    keep, stop = launch(client, "K"), launch(client, "S")
    wait_for_status(client, keep, {"running"})
    wait_for_status(client, stop, {"running"})
    queued = launch(client, "W")
    assert status_of(client, queued)["status"] == "queued"

    cancelled_queued = client.post(f"/api/clinical/sessions/{queued}/cancel").json()
    assert cancelled_queued["status"] == "cancelled"

    requested = client.post(f"/api/clinical/sessions/{stop}/cancel").json()
    assert requested["status"] == "running"
    assert requested["stop_requested"] is True
    assert wait_for_status(client, stop, TERMINAL)["status"] == "cancelled"
    assert status_of(client, keep)["status"] == "running"

    analyses.gate("K").set()
    assert wait_for_status(client, keep, TERMINAL)["status"] == "completed"
    assert "W" not in analyses.started
    assert client.post(f"/api/clinical/sessions/{stop}/cancel").status_code == 409


###############################################################################
def test_cancel_before_worker_start_releases_slot_and_admits_next(
    client, analyses, monkeypatch
) -> None:
    monkeypatch.setattr(get_clinical_scheduler(), "limit_provider", lambda: 1)
    manager = get_job_manager()
    original_run_job = manager.run_job
    entered, release = threading.Event(), threading.Event()
    calls: list[str] = []

    def paused_run_job(job_id, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        calls.append(job_id)
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=5)
        original_run_job(job_id, *args, **kwargs)

    monkeypatch.setattr(manager, "run_job", paused_run_job)
    first = launch(client, "P1")
    second = launch(client, "P2")
    assert entered.wait(timeout=5)

    requested = client.post(f"/api/clinical/sessions/{first}/cancel").json()
    assert requested["status"] == "running"
    release.set()

    assert wait_for_status(client, first, TERMINAL)["status"] == "cancelled"
    wait_for_status(client, second, {"running"})
    wait_until(lambda: analyses.started == ["P2"])
    assert len(calls) == 2
    repository = ClinicalSessionRepository(RepositoryContext.create())
    assert repository.count_sessions_with_status("running") == 1


###############################################################################
def test_concurrent_runs_keep_their_own_results(client, analyses) -> None:
    left, right = launch(client, "L"), launch(client, "R")
    wait_for_status(client, left, {"running"})
    wait_for_status(client, right, {"running"})
    analyses.gate("R").set()
    analyses.gate("L").set()
    wait_for_status(client, left, {"completed"})
    wait_for_status(client, right, {"completed"})

    repository = ClinicalSessionRepository(RepositoryContext.create())
    for session_id, marker in ((left, "L"), (right, "R")):
        payload = repository.get_session_result_payload(session_id) or {}
        assert payload["report"] == f"Report for {marker}"
        assert payload["diagnostics"] == {marker: 1}
        assert marker in repository.get_session_input(session_id)["clinical_input"]


###############################################################################
def test_restart_fails_interrupted_runs_and_keeps_queue(client, analyses, monkeypatch) -> None:
    repository = ClinicalSessionRepository(RepositoryContext.create())
    orphan = create_draft(client, "O")["session_id"]
    waiting = create_draft(client, "Z")["session_id"]
    for session_id in (orphan, waiting):
        assert repository.launch_draft(
            session_id,
            patient_name=None,
            visit_date=None,
            clinical_input="text",
            patient_image_base64=None,
            metadata={},
            run_request={},
        )
    # Simulate a crash: one row claimed by a worker that no longer exists.
    monkeypatch.setattr(get_clinical_scheduler(), "limit_provider", lambda: 0)
    assert repository.claim_next_queued("deadbeef") == orphan

    assert get_clinical_scheduler().reconcile_interrupted() == [orphan]
    interrupted = status_of(client, orphan)
    assert interrupted["status"] == "failed"
    assert interrupted["error"]["code"] == "interrupted"
    assert status_of(client, waiting)["status"] == "queued"
    client.post(f"/api/clinical/sessions/{waiting}/cancel")


###############################################################################
def test_active_sessions_cannot_be_deleted(client, analyses) -> None:
    session_id = launch(client, "D")
    wait_for_status(client, session_id, {"running"})
    assert client.delete(f"/api/inspection/sessions/{session_id}").status_code == 409
    analyses.gate("D").set()
    wait_for_status(client, session_id, TERMINAL)
    wait_until(lambda: not get_job_manager().is_job_running("clinical"))
    assert client.delete(f"/api/inspection/sessions/{session_id}").status_code == 200
