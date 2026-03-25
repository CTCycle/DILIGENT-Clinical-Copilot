from __future__ import annotations

from typing import Any

from DILIGENT.server.configurations import server_settings
from DILIGENT.server.domain.clinical import ClinicalSessionRequest
from DILIGENT.server.api import ollama as ollama_routes
from DILIGENT.server.api import session as session_routes


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
    monkeypatch.setattr(session_routes, "job_manager", job_manager_stub)

    response = session_routes.endpoint.start_clinical_job(
        ClinicalSessionRequest(anamnesis="Clinical context")
    )

    assert response.poll_interval == server_settings.jobs.polling_interval


# -----------------------------------------------------------------------------
def test_start_pull_job_uses_centralized_poll_interval(monkeypatch) -> None:
    job_manager_stub = JobManagerStub()
    monkeypatch.setattr(ollama_routes, "job_manager", job_manager_stub)

    response = ollama_routes.endpoint.start_pull_job(
        name="llama3.1:8b",
        stream=False,
    )

    assert response.poll_interval == server_settings.jobs.polling_interval
