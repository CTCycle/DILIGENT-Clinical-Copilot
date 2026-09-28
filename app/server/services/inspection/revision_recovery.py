from __future__ import annotations

from typing import Any

from repositories.context import RepositoryContext
from repositories.session_revision_repository import SessionRevisionRepository
from services.runtime.jobs import JobManager

REVISION_JOB_MISSING_STATUS_MESSAGE = (
    "Revision job worker is no longer available. Reload the persisted revision run "
    "and retry if needed."
)


def reconcile_interrupted_revision_jobs(
    jobs: JobManager,
    *,
    repository: SessionRevisionRepository | None = None,
) -> list[dict[str, Any]]:
    """Fail persisted revision runs that have no live worker after startup."""

    active_job_ids = {
        str(payload.get("job_id") or "")
        for payload in jobs.list_jobs(job_type="session_revision")
        if str(payload.get("status") or "") in {"pending", "running"}
    }
    revision_repository = repository or SessionRevisionRepository(
        RepositoryContext.create()
    )
    return revision_repository.reconcile_running_revision_runs(
        active_job_ids=active_job_ids,
        error={"message": REVISION_JOB_MISSING_STATUS_MESSAGE},
    )
