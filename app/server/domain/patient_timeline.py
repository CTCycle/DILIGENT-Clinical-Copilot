from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PatientTimelineEventType = Literal["therapy", "disease", "lab", "other"]
PatientTimelineGenerationStatus = Literal["llm_generated", "fallback"]
PatientTimelineGenerationErrorCode = Literal[
    "network_unavailable",
    "timeout",
    "authentication",
    "rate_limited",
    "upstream_error",
    "invalid_response",
    "configuration",
    "provider_error",
    "unknown",
]
PatientTimelineSourceKind = Literal["local", "cloud"]
PatientTimelineTimingType = Literal[
    "explicit_date",
    "relative",
    "duration",
    "recurring",
    "uncertain",
    "ordering",
]
PatientTimelineDatePrecision = Literal["day", "month", "year"]
PatientTimelineDateCertainty = Literal["explicit", "inferred", "uncertain"]

###############################################################################
class PatientTimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1200)
    event_type: PatientTimelineEventType = "other"
    timing_type: PatientTimelineTimingType = "uncertain"
    event_date: str | None = Field(default=None, max_length=40)
    event_date_end: str | None = Field(default=None, max_length=40)
    date_precision: PatientTimelineDatePrecision | None = None
    date_certainty: PatientTimelineDateCertainty = "uncertain"
    uncertainty_reason: str | None = Field(default=None, max_length=500)
    relative_time: str | None = Field(default=None, max_length=80)
    extracted_timing_text: str | None = Field(default=None, max_length=240)
    source_evidence: str | None = Field(default=None, max_length=1200)
    linked_patient_event_ids: list[str] = Field(default_factory=list)
    source: str | None = Field(default=None, max_length=80)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_rationale: str | None = Field(default=None, max_length=500)
    sort_order: int = Field(default=0, ge=0)

    # -------------------------------------------------------------------------
    @field_validator(
        "event_id",
        "title",
        "event_date",
        "event_date_end",
        "relative_time",
        "extracted_timing_text",
        "source",
        mode="before",
    )
    @classmethod
    def strip_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split()).strip()
        return normalized or None

    # -------------------------------------------------------------------------
    @field_validator(
        "description", "source_evidence", "confidence_rationale", "uncertainty_reason", mode="before"
    )
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split()).strip()
        return normalized or None

    # -------------------------------------------------------------------------
    @field_validator("linked_patient_event_ids", mode="before")
    @classmethod
    def normalize_linked_event_ids(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = " ".join(str(item).split()).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text[:120])
        return normalized

###############################################################################
class PatientTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timeline_id: int | None = Field(default=None, ge=1)
    session_id: int = Field(..., ge=1)
    generated_at: datetime
    generation_status: PatientTimelineGenerationStatus = "llm_generated"
    generation_note: str | None = Field(default=None, max_length=500)
    generation_error_code: PatientTimelineGenerationErrorCode | None = None
    source_model: str | None = Field(default=None, max_length=200)
    source_kind: PatientTimelineSourceKind | None = None
    model_provider: str | None = Field(default=None, max_length=40)
    source_payload_hash: str | None = Field(default=None, min_length=64, max_length=64)
    events: list[PatientTimelineEvent] = Field(default_factory=list)

    # -------------------------------------------------------------------------
    @field_validator("generation_note", mode="before")
    @classmethod
    def normalize_generation_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split()).strip()
        return normalized or None

###############################################################################
class PatientTimelineExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[PatientTimelineEvent] = Field(default_factory=list)

###############################################################################
class SessionTimelinePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timeline_id: int | None = Field(default=None, ge=1)
    session_id: int = Field(..., ge=1)
    generated_at: datetime
    generation_status: PatientTimelineGenerationStatus = "llm_generated"
    generation_note: str | None = Field(default=None, max_length=500)
    generation_error_code: PatientTimelineGenerationErrorCode | None = None
    source_model: str | None = Field(default=None, max_length=200)
    source_kind: PatientTimelineSourceKind | None = None
    model_provider: str | None = Field(default=None, max_length=40)
    source_payload_hash: str | None = Field(default=None, min_length=64, max_length=64)
    event_count: int = Field(default=0, ge=0)
    start_date: str | None = Field(default=None, max_length=40)
    end_date: str | None = Field(default=None, max_length=40)
    title: str | None = Field(default=None, max_length=200)
    source_evidence_event_count: int = Field(default=0, ge=0)
    missing_evidence_event_count: int = Field(default=0, ge=0)
    uncertain_event_count: int = Field(default=0, ge=0)
    undated_event_count: int = Field(default=0, ge=0)

###############################################################################
class SessionTimelineListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[SessionTimelinePreview] = Field(default_factory=list)

###############################################################################
class SessionTimelineRegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    force_regenerate: bool = False
