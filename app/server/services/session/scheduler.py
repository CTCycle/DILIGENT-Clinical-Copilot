# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any

from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.clinical.entities import ClinicalSessionRequest
from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext
from services.llm.runtime_config import LLMRuntimeConfig
from services.runtime.jobs import JobErrorSanitizer, JobManager, get_job_manager
from services.session.session_shared import execute_clinical_job
from services.session.session_workflow import prepare_clinical_run_workflow

CLINICAL_JOB_TYPE = "clinical"
INTERRUPTED_ERROR = {
    "code": "interrupted",
    "message": (
        "Interrupted because the application stopped while this analysis was "
        "running. Start a new analysis from this session's input."
    ),
}


###############################################################################
class ClinicalSessionScheduler:
    """Admit queued clinical sessions up to the configured concurrency limit.

    The queue is the set of ``queued`` session rows, so it survives restarts.
    Each admitted session runs as one ``clinical`` job with its own service
    instance: extractors hold event-loop-bound LLM clients and per-run
    diagnostics that must never be shared between concurrent analyses.
    """

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        job_manager: JobManager,
        repository_factory: Callable[[], ClinicalSessionRepository],
        service_factory: Callable[[JobManager], Any],
        limit_provider: Callable[[], int],
    ) -> None:
        self.job_manager = job_manager
        self.repository_factory = repository_factory
        self.service_factory = service_factory
        self.limit_provider = limit_provider
        self.lock = threading.Lock()

    # -------------------------------------------------------------------------
    def dispatch(self) -> list[int]:
        """Start queued sessions while capacity is free; returns started ids."""
        started: list[int] = []
        with self.lock:
            if not self.job_manager.accepting_jobs:
                return started
            repository = self.repository_factory()
            limit = max(1, int(self.limit_provider()))
            while repository.count_sessions_with_status("running") < limit:
                job_id = uuid.uuid4().hex[:8]
                session_id = repository.claim_next_queued(job_id)
                if session_id is None:
                    break
                try:
                    self.job_manager.start_job(
                        job_type=CLINICAL_JOB_TYPE,
                        runner=self.run_session,
                        kwargs={"session_id": session_id},
                        scope_key=f"clinical:{session_id}",
                        job_id=job_id,
                        on_cancelled_before_start=partial(
                            self.release_unstarted_session, session_id, job_id
                        ),
                    )
                except RuntimeError:
                    # Shutdown began between the claim and the start: the
                    # session never executed, so it waits in the queue again.
                    repository.transition_session(
                        session_id, from_statuses={"running"}, to_status="queued"
                    )
                    break
                started.append(session_id)
        if started:
            logger.info("Admitted queued clinical sessions %s", started)
        return started

    # -------------------------------------------------------------------------
    def run_session(self, session_id: int, job_id: str) -> dict[str, Any]:
        repository = self.repository_factory()
        try:
            result = self._execute(repository, session_id, job_id)
        except BaseException as exc:
            self._persist_unfinished(repository, session_id, job_id, exc)
            raise
        else:
            self._persist_unfinished(repository, session_id, job_id, None)
            return result
        finally:
            self.dispatch()

    # -------------------------------------------------------------------------
    def release_unstarted_session(self, session_id: int, job_id: str) -> None:
        """Finalize a claimed session whose job was cancelled before its runner."""
        repository = self.repository_factory()
        try:
            self._persist_unfinished(repository, session_id, job_id, None)
        finally:
            self.dispatch()

    # -------------------------------------------------------------------------
    def _execute(
        self, repository: ClinicalSessionRepository, session_id: int, job_id: str
    ) -> dict[str, Any]:
        stored = repository.get_run_request(session_id)
        if stored is None or stored["job_id"] != job_id:
            raise RuntimeError("Clinical session is no longer owned by this job.")
        run_request = stored["run_request"]
        request_payload = ClinicalSessionRequest.model_validate(
            {
                **run_request["request"],
                "patient_image_base64": stored["patient_image_base64"],
            }
        )
        runtime_snapshot = run_request.get("runtime_snapshot")
        runtime_snapshot_hash = run_request.get("runtime_snapshot_hash")
        service = self.service_factory(self.job_manager)
        with LLMRuntimeConfig.override_for_run(runtime_snapshot):
            prepared = prepare_clinical_run_workflow(service, request_payload)
            result = asyncio.run(
                execute_clinical_job(
                    service=service,
                    payload=prepared["patient_payload"],
                    patient_image_base64=request_payload.patient_image_base64,
                    job_id=job_id,
                    session_id=session_id,
                    section_extraction=prepared["section_extraction"],
                    normalized_document=prepared["normalized_document"],
                    report_mode=request_payload.report_mode,
                )
            )
        if not result:
            return {"session_id": session_id}
        if runtime_snapshot is not None:
            result.setdefault("runtime_settings", {}).update(runtime_snapshot)
            artifacts = result.setdefault("pipeline_artifacts", {})
            artifacts["runtime_snapshot"] = runtime_snapshot
            artifacts["runtime_snapshot_hash"] = runtime_snapshot_hash
            result["runtime_settings"]["runtime_snapshot_hash"] = runtime_snapshot_hash
            repository.upsert_session_result_payload(session_id, result)
        # The full payload lives on the session; the job keeps only its link.
        return {"session_id": session_id}

    # -------------------------------------------------------------------------
    def _persist_unfinished(
        self,
        repository: ClinicalSessionRepository,
        session_id: int,
        job_id: str,
        exc: BaseException | None,
    ) -> None:
        """Give a session that is still ``running`` its truthful terminal state."""
        if self.job_manager.should_stop(job_id):
            if self.job_manager.accepting_jobs:
                target, error = "cancelled", None
            else:
                target, error = "failed", INTERRUPTED_ERROR
        elif exc is not None:
            message = (
                JobErrorSanitizer.build_safe_job_error_message(exc)
                if isinstance(exc, Exception)
                else "Clinical analysis stopped unexpectedly."
            )
            target, error = "failed", {"code": "clinical_job_failed", "message": message}
        else:
            target, error = "failed", {
                "code": "clinical_job_failed",
                "message": "Clinical analysis ended without a result.",
            }
        repository.transition_session(
            session_id, from_statuses={"running"}, to_status=target, error=error
        )

    # -------------------------------------------------------------------------
    def cancel(self, session_id: int) -> str:
        """Cancel a queued or running session; returns the resulting status."""
        repository = self.repository_factory()
        state = repository.get_session_state(session_id)
        if state is None:
            raise KeyError(session_id)
        if state["status"] == "queued" and repository.transition_session(
            session_id, from_statuses={"queued"}, to_status="cancelled"
        ):
            return "cancelled"
        state = repository.get_session_state(session_id) or state
        if state["status"] != "running":
            raise SessionNotCancellableError(state["status"])
        job_id = state.get("job_id")
        if job_id and self.job_manager.get_job_status(job_id) is not None:
            self.job_manager.cancel_job(job_id)
            return "running"
        # No live worker owns the row; nothing can still write to it.
        repository.transition_session(
            session_id, from_statuses={"running"}, to_status="cancelled"
        )
        return "cancelled"

    # -------------------------------------------------------------------------
    def reconcile_interrupted(self) -> list[int]:
        """Fail running rows whose worker no longer exists (restart recovery)."""
        active_job_ids = {
            str(job["job_id"])
            for job in self.job_manager.list_jobs(job_type=CLINICAL_JOB_TYPE)
            if job.get("status") in ("pending", "running")
        }
        interrupted = self.repository_factory().fail_interrupted_sessions(
            active_job_ids=active_job_ids, error=INTERRUPTED_ERROR
        )
        if interrupted:
            logger.warning(
                "Marked interrupted clinical sessions as failed: %s", interrupted
            )
        return interrupted


###############################################################################
class SessionNotCancellableError(RuntimeError):

    # -------------------------------------------------------------------------
    def __init__(self, status: str) -> None:
        super().__init__(f"A {status} session cannot be cancelled.")
        self.status = status


###############################################################################
def _build_repository() -> ClinicalSessionRepository:
    return ClinicalSessionRepository(RepositoryContext.create())


###############################################################################
def _build_service(job_manager: JobManager) -> Any:
    from services.session.factory import build_clinical_session_service

    return build_clinical_session_service(job_manager)


###############################################################################
@lru_cache(maxsize=1)
def get_clinical_scheduler() -> ClinicalSessionScheduler:
    return ClinicalSessionScheduler(
        job_manager=get_job_manager(),
        repository_factory=_build_repository,
        service_factory=_build_service,
        limit_provider=lambda: get_server_settings().jobs.max_concurrent_clinical_jobs,
    )
