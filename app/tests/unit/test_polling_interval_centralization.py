from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

from configurations.startup import server_settings
from domain.clinical import ClinicalSessionRequest
from api import ollama as ollama_routes
from api import session as session_routes
from services.session import session_workflow
from services.clinical import job_progress as clinical_job_progress


def get_route_owner(router: Any, route_path: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", "").endswith(route_path):
            owner = getattr(route.endpoint, "__self__", None)
            if owner is not None:
                return owner
    raise AssertionError(f"Route not found: {route_path}")


###############################################################################
class JobManagerStub:
    def __init__(self) -> None:
        self.job_type: str | None = None

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        return False

    # -------------------------------------------------------------------------
    def start_job(
        self,
        job_type: str,
        runner: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> str:
        self.job_type = job_type
        return "job-1234"

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        if self.job_type is None:
            return None
        return {
            "job_id": job_id,
            "job_type": self.job_type,
            "status": "running",
            "progress": 0.0,
            "result": None,
            "error": None,
        }


# -----------------------------------------------------------------------------
def test_start_clinical_job_uses_centralized_poll_interval(monkeypatch) -> None:
    job_manager_stub = JobManagerStub()
    endpoint = get_route_owner(session_routes.router, "/clinical/jobs")
    monkeypatch.setattr(endpoint.service, "job_manager", job_manager_stub)
    monkeypatch.setattr(
        endpoint.service,
        "validate_clinical_input",
        lambda _: SimpleNamespace(ready=True, blocking_issues=[]),
    )
    monkeypatch.setattr(
        session_workflow.LLMRuntimeConfig,
        "is_cloud_enabled",
        staticmethod(lambda: False),
    )

    async def preprocess_unified_input(request_payload: ClinicalSessionRequest) -> tuple[ClinicalSessionRequest, object]:
        return request_payload, object()

    monkeypatch.setattr(endpoint.service, "preprocess_unified_input", preprocess_unified_input)
    monkeypatch.setattr(endpoint.service, "build_patient_payload", lambda request_payload: object())
    monkeypatch.setattr(endpoint.service, "ensure_submission_requirements", lambda _: None)

    response = endpoint.service.start_clinical_job(
        ClinicalSessionRequest(
            clinical_input=(
                "Anamnesis: Clinical context\n"
                "Drugs: Acetaminophen 500 mg 1 - 0 - 0 - 0 po started from 01/01/2024\n"
                "Lab analysis: ALT 300 U/L"
            ),
            visit_date=date(2026, 4, 24),
            selected_model_providers=["openai"],
        )
    )

    assert response.poll_interval == server_settings.jobs.polling_interval


# -----------------------------------------------------------------------------
def test_start_pull_job_uses_centralized_poll_interval(monkeypatch) -> None:
    job_manager_stub = JobManagerStub()
    endpoint = get_route_owner(ollama_routes.router, "/pull/jobs")
    monkeypatch.setattr(endpoint.service, "job_manager", job_manager_stub)

    response = endpoint.start_pull_job(
        name="llama3.1:8b",
        stream=False,
    )

    assert response.poll_interval == server_settings.jobs.polling_interval


# -----------------------------------------------------------------------------
def test_clinical_progress_callback_raises_when_stop_requested(monkeypatch) -> None:
    class StopRequestedJobManagerStub:
        def should_stop(self, job_id: str) -> bool:
            return True

    monkeypatch.setattr(
        clinical_job_progress,
        "get_job_manager",
        lambda: StopRequestedJobManagerStub(),
    )

    try:
        clinical_job_progress.report_clinical_job_progress(
            "job-cancelled",
            stage="llm_analysis",
            progress=75.0,
        )
    except clinical_job_progress.ClinicalJobCancelled:
        return
    raise AssertionError("Expected ClinicalJobCancelled when stop is requested.")

