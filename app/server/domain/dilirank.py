from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

###############################################################################
class DiliRankCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int | None = None
    drug_name: str | None = None
    ltkb_id: str
    compound_name: str
    severity_class: int | None = None
    label_section: str | None = None
    dili_concern: str
    comment: str | None = None
    source_url: str | None = None
    source_last_modified: str | None = None

###############################################################################
class DiliRankCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[DiliRankCatalogItem] = Field(default_factory=list)
    total: int
    offset: int
    limit: int

###############################################################################
class DiliRankDrugResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    records: list[DiliRankCatalogItem] = Field(default_factory=list)

###############################################################################
class DiliRankUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    redownload: bool = False

###############################################################################
class DiliRankUpdateConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["dilirank"] = "dilirank"
    defaults: dict[str, object] = Field(default_factory=lambda: {"redownload": False})
    allowed_fields: list[str] = Field(default_factory=lambda: ["redownload"])
    summary: dict[str, object] = Field(default_factory=dict)
    read_only: bool = False
