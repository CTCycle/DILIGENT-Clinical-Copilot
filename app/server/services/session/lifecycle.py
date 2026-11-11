# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

from common.exceptions import ServiceConflictError, ServiceNotFoundError
from configurations.startup import get_server_settings
from domain.clinical.entities import ClinicalSessionRequest
from domain.clinical.lifecycle import (
    ClinicalSessionInput,
    ClinicalSessionInputUpdate,
    ClinicalSessionInputWrite,
    ClinicalSessionState,
)
from repositories.clinical_session_repository import (
    ClinicalSessionRepository,
    SessionStateConflictError,
)
from services.runtime.jobs import JobManager
from services.session.scheduler import (
    ClinicalSessionScheduler,
    SessionNotCancellableError,
)

# Live job versions restart at zero per job; scaling the persisted row version
# keeps the combined snapshot version monotonic for one session.
_ROW_VERSION_SCALE = 1_000_000


###############################################################################
class ClinicalSessionLifecycleService:
    """Session-centred lifecycle: drafts, launch, status, and cancellation."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        clinical_service: Any,
        repository: ClinicalSessionRepository,
        scheduler: ClinicalSessionScheduler,
        job_manager: JobManager,
    ) -> None:
        self.clinical_service = clinical_service
        self.repository = repository
        self.scheduler = scheduler
        self.job_manager = job_manager

    # -------------------------------------------------------------------------
    def create_draft(self, payload: ClinicalSessionInputWrite) -> ClinicalSessionState:
        state = self.repository.create_draft(
            patient_name=payload.patient_name,
            visit_date=payload.visit_date,
            clinical_input=payload.clinical_input,
            patient_image_base64=payload.patient_image_base64,
        )
        return self._to_state(state)

    # -------------------------------------------------------------------------
    def get_input(self, session_id: int) -> ClinicalSessionInput:
        record = self.repository.get_session_input(session_id)
        if record is None:
            raise ServiceNotFoundError("Clinical session not found.")
        return ClinicalSessionInput(**record)

    # -------------------------------------------------------------------------
    def update_draft(
        self, session_id: int, payload: ClinicalSessionInputUpdate
    ) -> ClinicalSessionState:
        try:
            state = self.repository.update_draft_input(
                session_id,
                patient_name=payload.patient_name,
                visit_date=payload.visit_date,
                clinical_input=payload.clinical_input,
                patient_image_base64=payload.patient_image_base64,
                expected_updated_at=payload.expected_updated_at,
            )
        except SessionStateConflictError as exc:
            raise ServiceConflictError(str(exc)) from exc
        if state is None:
            raise ServiceNotFoundError("Clinical session not found.")
        return self._to_state(state)

    # -------------------------------------------------------------------------
    def launch(
        self, session_id: int, request_payload: ClinicalSessionRequest
    ) -> ClinicalSessionState:
        self._require_status(session_id, {"draft"}, "Only a draft session can be launched.")
        launch = self.clinical_service.validate_clinical_launch(request_payload)
        request_payload = launch["request"]
        agent_context = launch["agent_context"]
        patient = launch["patient_payload"]
        run_request = {
            "request": request_payload.model_dump(
                mode="json", exclude={"patient_image_base64"}
            ),
            "runtime_snapshot": launch["runtime_snapshot"],
            "runtime_snapshot_hash": launch["runtime_snapshot_hash"],
        }
        launched = self.repository.launch_draft(
            session_id,
            patient_name=patient.name,
            visit_date=patient.visit_date,
            clinical_input=request_payload.clinical_input or "",
            patient_image_base64=request_payload.patient_image_base64,
            metadata={
                "evidence_sources": list(request_payload.evidence_sources),
                "agent": {
                    key: value
                    for key, value in agent_context.provenance().items()
                    if key != "agent_configuration"
                },
            },
            run_request=run_request,
            agent_revision_id=agent_context.agent_revision_id,
        )
        if not launched:
            raise ServiceConflictError("This session was already launched.")
        self.scheduler.dispatch()
        return self.get_state(session_id)

    # -------------------------------------------------------------------------
    def get_state(self, session_id: int) -> ClinicalSessionState:
        state = self.repository.get_session_state(session_id)
        if state is None:
            raise ServiceNotFoundError("Clinical session not found.")
        return self._to_state(state)

    # -------------------------------------------------------------------------
    def cancel(self, session_id: int) -> ClinicalSessionState:
        try:
            self.scheduler.cancel(session_id)
        except KeyError as exc:
            raise ServiceNotFoundError("Clinical session not found.") from exc
        except SessionNotCancellableError as exc:
            raise ServiceConflictError(str(exc)) from exc
        return self.get_state(session_id)

    # -------------------------------------------------------------------------
    def _require_status(self, session_id: int, allowed: set[str], message: str) -> None:
        state = self.repository.get_session_state(session_id)
        if state is None:
            raise ServiceNotFoundError("Clinical session not found.")
        if state["status"] not in allowed:
            raise ServiceConflictError(message)

    # -------------------------------------------------------------------------
    def _to_state(self, state: dict[str, Any]) -> ClinicalSessionState:
        version = int(state["state_version"]) * _ROW_VERSION_SCALE
        stop_requested = False
        values = {
            key: state[key]
            for key in ClinicalSessionState.model_fields
            if key in state and key not in {"version", "stop_requested"}
        }
        job_id = state.get("job_id")
        if state["status"] == "running" and job_id:
            job = self.job_manager.get_job_status(job_id)
            if job is not None:
                result = job.get("result") or {}
                stop_requested = bool(job.get("stop_requested"))
                version += int(job.get("version") or 0)
                if float(job.get("progress") or 0.0) > 0.0:
                    values["progress"] = float(job["progress"])
                values["progress_stage"] = (
                    result.get("progress_stage") or values.get("progress_stage")
                )
                values["progress_message"] = (
                    result.get("progress_message") or values.get("progress_message")
                )
        return ClinicalSessionState(
            **values,
            version=version,
            stop_requested=stop_requested,
            poll_interval=get_server_settings().jobs.polling_interval,
        )
