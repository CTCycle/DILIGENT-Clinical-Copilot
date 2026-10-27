# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from time import monotonic, time
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from domain.jobs import JobCancelResponse, JobSummary, JobSummaryListResponse
from services.runtime.jobs import get_job_manager

# Job types whose dedicated cancel route is a plain cooperative stop request.
# Revision and timeline jobs also touch persisted run state, so they are
# cancelled from their own pages.
CANCELLABLE_JOB_TYPES = frozenset(
    {
        "clinical",
        "ollama_pull",
        "rxnav_update",
        "livertox_update",
        "dilirank_update",
        "structured_sources_update",
        "rag_update",
    }
)
ACTIVE_STATUSES = frozenset({"pending", "running"})
MAX_RECENT_TERMINAL_JOBS = 20

router = APIRouter(tags=["jobs"])


###############################################################################
def summarize_job(snapshot: dict[str, Any], *, now_mono: float, now_wall: float) -> JobSummary:
    result = snapshot.get("result") or {}
    job_status = str(snapshot["status"])
    progress = float(snapshot.get("progress") or 0.0)
    # Progress is only reported when the worker actually measured it.
    measurable = job_status == "completed" or (
        job_status == "running" and progress > 0.0
    )
    completed_at = snapshot.get("completed_at")
    phase = result.get("phase") or result.get("stage")
    message = result.get("progress_message") or result.get("message")
    return JobSummary(
        job_id=str(snapshot["job_id"]),
        job_type=str(snapshot["job_type"]),
        status=job_status,
        progress=progress if measurable else None,
        phase=str(phase) if phase else None,
        message=str(message) if message else None,
        error=snapshot.get("error"),
        started_at=now_wall - (now_mono - float(snapshot["created_at"])),
        finished_at=(
            now_wall - (now_mono - float(completed_at))
            if completed_at is not None
            else None
        ),
        stop_requested=bool(snapshot.get("stop_requested")),
        cancellable=(
            snapshot["job_type"] in CANCELLABLE_JOB_TYPES
            and job_status in ACTIVE_STATUSES
            and not snapshot.get("stop_requested")
        ),
    )


###############################################################################
def list_jobs(response: Response) -> JobSummaryListResponse:
    response.headers["Cache-Control"] = "no-store"
    now_mono, now_wall = monotonic(), time()
    summaries = [
        summarize_job(snapshot, now_mono=now_mono, now_wall=now_wall)
        for snapshot in get_job_manager().list_jobs()
    ]
    summaries.sort(key=lambda job: job.started_at, reverse=True)
    active = [job for job in summaries if job.status in ACTIVE_STATUSES]
    recent = [job for job in summaries if job.status not in ACTIVE_STATUSES]
    return JobSummaryListResponse(jobs=active + recent[:MAX_RECENT_TERMINAL_JOBS])


###############################################################################
def cancel_job(job_id: str) -> JobCancelResponse:
    manager = get_job_manager()
    snapshot = manager.get_job_status(job_id)
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    if snapshot["job_type"] not in CANCELLABLE_JOB_TYPES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This job must be cancelled from its own page.",
        )
    if manager.cancel_job(job_id) is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job cannot be cancelled.",
        )
    return JobCancelResponse(job_id=job_id, success=True, message="Cancellation requested")


router.add_api_route(
    "/jobs",
    list_jobs,
    methods=["GET"],
    response_model=JobSummaryListResponse,
    status_code=status.HTTP_200_OK,
)
router.add_api_route(
    "/jobs/{job_id}",
    cancel_job,
    methods=["DELETE"],
    response_model=JobCancelResponse,
    status_code=status.HTTP_200_OK,
)
