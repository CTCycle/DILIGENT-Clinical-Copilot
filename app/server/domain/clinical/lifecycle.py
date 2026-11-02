# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SessionLifecycleStatus = Literal[
    "draft", "queued", "running", "completed", "cancelled", "failed"
]


###############################################################################
class ClinicalSessionInputWrite(BaseModel):
    """Clinician-entered input saved as a draft without running analysis."""

    model_config = ConfigDict(extra="forbid")

    patient_name: str | None = Field(default=None, max_length=200)
    visit_date: date | None = None
    clinical_input: str = Field(default="", max_length=100000)
    patient_image_base64: str | None = Field(default=None, max_length=8_000_000)

    # -------------------------------------------------------------------------
    @field_validator("patient_name", "patient_image_base64", mode="before")
    @classmethod
    def blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value


###############################################################################
class ClinicalSessionInputUpdate(ClinicalSessionInputWrite):
    expected_updated_at: datetime


###############################################################################
class ClinicalSessionInput(BaseModel):
    session_id: int
    status: SessionLifecycleStatus
    patient_name: str | None = None
    visit_date: date | None = None
    clinical_input: str = ""
    patient_image_base64: str | None = None
    updated_at: datetime | None = None


###############################################################################
class ClinicalSessionError(BaseModel):
    code: str
    message: str


###############################################################################
class ClinicalSessionState(BaseModel):
    """Authoritative lifecycle snapshot; the only poll target for a session."""

    session_id: int
    status: SessionLifecycleStatus
    patient_name: str | None = None
    visit_date: date | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    queued_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    queue_position: int | None = None
    progress: float | None = None
    progress_stage: str | None = None
    progress_message: str | None = None
    stop_requested: bool = False
    error: ClinicalSessionError | None = None
    has_report: bool = False
    version: int = 0
    poll_interval: float = 1.0
