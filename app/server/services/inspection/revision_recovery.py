# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

from repositories.context import RepositoryContext
from repositories.session_revision_repository import SessionRevisionRepository
from services.runtime.jobs import JobManager

REVISION_JOB_MISSING_STATUS_MESSAGE = (
    "Revision job worker is no longer available. Reload the persisted revision run "
    "and retry if needed."
)

REVISION_JOB_SHUTDOWN_MESSAGE = (
    "The application shut down before this revision run finished. "
    "Start a new revision run if needed."
)

###############################################################################
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

###############################################################################
def fail_revision_runs_interrupted_by_shutdown(
    *,
    repository: SessionRevisionRepository | None = None,
) -> list[dict[str, Any]]:
    """Fail runs whose workers did not stop before the application exited."""

    revision_repository = repository or SessionRevisionRepository(
        RepositoryContext.create()
    )
    return revision_repository.reconcile_running_revision_runs(
        active_job_ids=set(),
        error={"message": REVISION_JOB_SHUTDOWN_MESSAGE},
    )
