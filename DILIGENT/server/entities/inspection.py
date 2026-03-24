from __future__ import annotations

from datetime import date as DateValue
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SessionStatus = Literal["successful", "failed"]
DateFilterMode = Literal["before", "after", "exact"]


###############################################################################
class SessionCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: int
    patient_name: str | None = None
    session_timestamp: datetime | None = None
    status: SessionStatus
    total_duration: float | None = None


###############################################################################
class SessionCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[SessionCatalogItem] = Field(default_factory=list)
    total: int
    offset: int
    limit: int


###############################################################################
class SessionReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: int
    report: str


###############################################################################
class RxNavCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    last_update: str | None = None


###############################################################################
class RxNavCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[RxNavCatalogItem] = Field(default_factory=list)
    total: int
    offset: int
    limit: int


###############################################################################
class DrugAliasEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias: str
    alias_kind: str


###############################################################################
class DrugAliasGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    aliases: list[DrugAliasEntry] = Field(default_factory=list)


###############################################################################
class DrugAliasesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    groups: list[DrugAliasGroup] = Field(default_factory=list)


###############################################################################
class LiverToxCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    last_update: str | None = None


###############################################################################
class LiverToxCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[LiverToxCatalogItem] = Field(default_factory=list)
    total: int
    offset: int
    limit: int


###############################################################################
class LiverToxExcerptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    excerpt: str
    last_update: str | None = None


###############################################################################
class DeleteEntityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deleted: bool


###############################################################################
class SessionListFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search: str | None = None
    status: SessionStatus | None = None
    date_mode: DateFilterMode | None = None
    date: DateValue | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)


###############################################################################
class CatalogListFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)
