from __future__ import annotations

import uuid
from typing import Any

from domain.inspection import SessionRevisionRequest

REVISION_JOB_MISSING_STATUS_MESSAGE = (
    "Revision job worker is no longer available. Reload the persisted revision run "
    "and retry if needed."
)

###############################################################################
class SessionRevisionNotFoundError(ValueError):
    pass

###############################################################################
class SessionRevisionConflictError(ValueError):
    pass

###############################################################################
class SessionRevisionValidationError(ValueError):
    pass

###############################################################################
class InspectionRevisionScaffoldMixin:
    session_revision_repository: Any
    jobs: Any
    revision_agent_runner: Any
    REVISION_JOB_TYPE: str
    get_session_detail: Any
    get_session_version_detail: Any
    get_job_status: Any
    cancel_job: Any

    # -------------------------------------------------------------------------
    def _run_revision_agent(self, **kwargs: Any) -> dict[str, Any]:
        pipeline_run_id = str(kwargs["pipeline_run_id"])
        job_id = str(kwargs["job_id"])
        try:
            result = self.revision_agent_runner.run_agentic(**kwargs)
        except Exception:
            self.session_revision_repository.fail_revision_run(
                pipeline_run_id=pipeline_run_id,
                error={"message": "Revision processing failed. Retry the revision if needed."},
            )
            raise
        if self.jobs.should_stop(job_id):
            self.session_revision_repository.cancel_revision_run(pipeline_run_id=pipeline_run_id)
        return result

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_revision_request(
        request: SessionRevisionRequest | None,
    ) -> SessionRevisionRequest:
        return request if request is not None else SessionRevisionRequest()

    # -------------------------------------------------------------------------
    @staticmethod
    def _session_has_revision_source(session: dict[str, Any]) -> bool:
        for key in ("source_clinical_text", "session_text"):
            if str(session.get(key) or "").strip():
                return True
        sections = session.get("sections")
        if not isinstance(sections, dict):
            return False
        return any(str(value or "").strip() for value in sections.values())

    # -------------------------------------------------------------------------
    def _revision_scope_key(self, root_session_id: int) -> str:
        return f"revision:{int(root_session_id)}"

    # -------------------------------------------------------------------------
    def start_revision_job(
        self,
        session_id: int,
        request: SessionRevisionRequest | None = None,
    ) -> dict[str, Any]:
        revision_request = self._coerce_revision_request(request)
        session = self.get_session_detail(session_id)
        if session is None:
            raise SessionRevisionNotFoundError("Session not found.")
        if not self._session_has_revision_source(session):
            raise SessionRevisionValidationError(
                "Session has no clinical text available for revision."
            )
        source_version = self.session_revision_repository.get_version_record_for_session(session_id)
        if source_version is None:
            raise SessionRevisionValidationError(
                "Session version could not be prepared for revision."
            )
        root_session_id = int(source_version["root_session_id"])
        scope_key = self._revision_scope_key(root_session_id)
        if self.jobs.is_job_running(self.REVISION_JOB_TYPE, scope_key=scope_key):
            raise SessionRevisionConflictError(
                "A revision job is already running for this root session."
            )

        pipeline_run_id = uuid.uuid4().hex
        model_configuration = {
            "pipeline_run_id": pipeline_run_id,
            "revision_agent": "single_model_agentic_revision",
            "revision_mode": "agentic_revision",
            "root_session_id": root_session_id,
            "source_session_id": int(session_id),
            "source_version_id": int(source_version["version_id"]),
            "model_overrides": dict(revision_request.model_overrides or {}),
            "metadata": dict(revision_request.metadata or {}),
        }
        shell = self.session_revision_repository.create_revision_version_shell(
            session_id,
            reviewer_note=revision_request.revision_instruction,
            configuration=model_configuration,
            pipeline_run_id=pipeline_run_id,
            initiated_by="revision_agent",
        )
        if shell is None:
            raise SessionRevisionValidationError(
                "Revision version shell could not be created."
            )
        revision_version_id = int(shell["revision_version_id"])
        model_configuration["revision_version_id"] = revision_version_id
        self.session_revision_repository.create_or_update_revision_run(
            pipeline_run_id=pipeline_run_id,
            session_id=int(session_id),
            root_session_id=root_session_id,
            source_version_id=int(source_version["version_id"]),
            target_revision_version_id=revision_version_id,
            revision_mode="agentic_revision",
            revision_kind="llm_assisted_revision",
            configuration=model_configuration,
            reviewer_note=revision_request.revision_instruction,
            status="running",
            initiated_by="revision_agent",
            actor_source="system",
            actor_confidence="system",
        )
        job_id = self.jobs.start_job(
            job_type=self.REVISION_JOB_TYPE,
            runner=self._run_revision_agent,
            kwargs={
                "pipeline_run_id": pipeline_run_id,
                "revision_version_id": revision_version_id,
                "source_version_id": int(source_version["version_id"]),
                "session": session,
                "request": revision_request,
                "model_configuration": model_configuration,
            },
            scope_key=scope_key,
        )
        model_configuration["job_id"] = job_id
        self.session_revision_repository.create_or_update_revision_run(
            pipeline_run_id=pipeline_run_id,
            session_id=int(session_id),
            root_session_id=root_session_id,
            source_version_id=int(source_version["version_id"]),
            target_revision_version_id=revision_version_id,
            revision_mode="agentic_revision",
            revision_kind="llm_assisted_revision",
            configuration=model_configuration,
            reviewer_note=revision_request.revision_instruction,
            status="running",
            initiated_by="revision_agent",
            actor_source="system",
            actor_confidence="system",
        )
        status_payload = self.jobs.get_job_status(job_id)
        if status_payload is None:
            raise RuntimeError("Failed to initialize revision job.")
        status_payload["poll_interval"] = 1.0
        status_payload["result"] = {
            **(status_payload.get("result") or {}),
            "pipeline_run_id": pipeline_run_id,
            "revision_version_id": revision_version_id,
        }
        self.jobs.update_result(job_id, status_payload["result"])
        return status_payload

    # -------------------------------------------------------------------------
    def retry_revision_job(self, pipeline_run_id: str) -> dict[str, Any]:
        run = self.get_revision_run(pipeline_run_id)
        if run is None:
            raise SessionRevisionNotFoundError("Revision run not found.")
        configuration = run.get("configuration")
        if not isinstance(configuration, dict):
            configuration = {}
        request = SessionRevisionRequest(
            revision_instruction=run.get("reviewer_note"),
            model_overrides=configuration.get("model_overrides") or {},
            metadata=configuration.get("metadata") or {},
        )
        return self.start_revision_job(int(run["session_id"]), request)

    # -------------------------------------------------------------------------
    def get_revision_job_status(self, job_id: str) -> dict[str, Any] | None:
        payload = self.get_job_status(job_id, expected_type=self.REVISION_JOB_TYPE)
        if payload is not None:
            return payload
        run = self.session_revision_repository.get_revision_run_by_job_id(job_id)
        if run is None:
            return None
        self.session_revision_repository.fail_revision_run(
            pipeline_run_id=str(run["pipeline_run_id"]),
            error={"message": REVISION_JOB_MISSING_STATUS_MESSAGE},
        )
        return {
            "job_id": job_id,
            "job_type": self.REVISION_JOB_TYPE,
            "status": "failed",
            "progress": 100.0,
            "result": {
                "recoverable": True,
                "pipeline_run_id": run["pipeline_run_id"],
                "revision_version_id": run.get("target_revision_version_id"),
            },
            "error": REVISION_JOB_MISSING_STATUS_MESSAGE,
            "created_at": None,
            "completed_at": None,
            "version": None,
        }

    # -------------------------------------------------------------------------
    def cancel_revision_job(self, job_id: str) -> bool:
        cancelled = self.cancel_job(job_id, expected_type=self.REVISION_JOB_TYPE)
        if not cancelled:
            return False
        run = self.session_revision_repository.get_revision_run_by_job_id(job_id)
        if run is not None:
            self.session_revision_repository.cancel_revision_run(pipeline_run_id=run["pipeline_run_id"])
        return True

    # -------------------------------------------------------------------------
    def get_revision_run(self, pipeline_run_id: str) -> dict[str, Any] | None:
        return self.session_revision_repository.get_revision_run(pipeline_run_id)

    # -------------------------------------------------------------------------
    def list_revision_steps(self, pipeline_run_id: str) -> list[dict[str, Any]]:
        return self.session_revision_repository.list_revision_steps(pipeline_run_id)

    # -------------------------------------------------------------------------
    def list_revision_artifacts(
        self,
        session_id: int,
        *,
        version_id: int,
    ) -> list[dict[str, Any]]:
        detail = self.get_session_version_detail(session_id, version_id=version_id)
        if detail is None:
            raise SessionRevisionNotFoundError("Revision version not found.")
        return self.session_revision_repository.list_revision_artifacts_for_version(
            revision_version_id=version_id,
        )

    # -------------------------------------------------------------------------
    def list_revision_entities(
        self,
        session_id: int,
        *,
        version_id: int,
    ) -> list[dict[str, Any]]:
        detail = self.get_session_version_detail(session_id, version_id=version_id)
        if detail is None:
            raise SessionRevisionNotFoundError("Revision version not found.")
        return self.session_revision_repository.list_revision_entities_for_version(
            revision_version_id=version_id,
        )

    # -------------------------------------------------------------------------
    def list_revision_reviews(
        self,
        session_id: int,
        *,
        version_id: int,
    ) -> list[dict[str, Any]]:
        detail = self.get_session_version_detail(session_id, version_id=version_id)
        if detail is None:
            raise SessionRevisionNotFoundError("Revision version not found.")
        return self.session_revision_repository.list_revision_reviews_for_version(
            revision_version_id=version_id,
        )

    # -------------------------------------------------------------------------
    def update_revision_clinical_review(
        self,
        session_id: int,
        *,
        version_id: int,
        clinical_review_status: str,
        reviewer_note: str | None,
        reviewed_by: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        detail = self.get_session_version_detail(session_id, version_id=version_id)
        if detail is None:
            raise SessionRevisionNotFoundError("Revision version not found.")
        return self.session_revision_repository.record_revision_review_action(
            revision_version_id=version_id,
            clinical_review_status=clinical_review_status,
            reviewer_note=reviewer_note,
            reviewed_by=reviewed_by,
            metadata=metadata,
        )
