# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ClinicalDraftForm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_name: str = Field(default="", max_length=200)
    visit_date: date | None = None
    clinical_input: str = Field(default="", max_length=100_000)


class ClinicalDraftWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)
    expected_draft_key: UUID | None = None
    form: ClinicalDraftForm


class ClinicalDraftSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_key: UUID | None
    revision: int
    updated_at: datetime | None
    form: ClinicalDraftForm | None
