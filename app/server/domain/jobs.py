# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

###############################################################################
class JobStartResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    message: str
    poll_interval: float

###############################################################################
class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: float
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float | None = None
    completed_at: float | None = None
    version: int | None = None
    stop_requested: bool = False

###############################################################################
class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]

###############################################################################
class JobSummary(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: float | None = None
    phase: str | None = None
    message: str | None = None
    error: str | None = None
    started_at: float
    finished_at: float | None = None
    stop_requested: bool = False
    cancellable: bool = False

###############################################################################
class JobSummaryListResponse(BaseModel):
    jobs: list[JobSummary]

###############################################################################
class JobCancelResponse(BaseModel):
    job_id: str
    success: bool
    message: str
