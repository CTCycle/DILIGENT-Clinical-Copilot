from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


PatientTimelineEventType = Literal["therapy", "disease", "lab", "other"]


###############################################################################
class PatientTimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1200)
    event_type: PatientTimelineEventType = "other"
    event_date: str | None = Field(default=None, max_length=40)
    relative_time: str | None = Field(default=None, max_length=80)
    source: str | None = Field(default=None, max_length=80)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sort_order: int = Field(default=0, ge=0)

    # -------------------------------------------------------------------------
    @field_validator("event_id", "title", "event_date", "relative_time", "source", mode="before")
    @classmethod
    def strip_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split()).strip()
        return normalized or None

    # -------------------------------------------------------------------------
    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split()).strip()
        return normalized or None


###############################################################################
class PatientTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: int = Field(..., ge=1)
    generated_at: datetime
    events: list[PatientTimelineEvent] = Field(default_factory=list)


###############################################################################
class PatientTimelineExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[PatientTimelineEvent] = Field(default_factory=list)


###############################################################################
class SessionTimelineRegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    force_regenerate: bool = False
