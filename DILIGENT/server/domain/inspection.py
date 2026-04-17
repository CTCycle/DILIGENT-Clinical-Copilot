from __future__ import annotations

from datetime import date as DateValue
from datetime import datetime
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SessionStatus = Literal["successful", "failed"]
DateFilterMode = Literal["before", "after", "exact"]
InspectionUpdateTarget = Literal["rxnav", "livertox", "dili_priors", "drug_labels", "rag"]
InspectionUpdateJobType = Literal[
    "rxnav_update",
    "livertox_update",
    "dili_priors_update",
    "drug_labels_update",
    "rag_update",
]
InspectionJobPhase = Literal[
    "configuration_accepted",
    "update_started",
    "source_data_loading",
    "processing_extraction",
    "persistence_indexing",
    "finalization",
    "completed",
    "cancelled",
    "failed",
]

CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MAX_SEARCH_LENGTH = 256


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
class DiliPriorCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    dilirank_class: str | None = None
    dilist_class: str | None = None
    linked_source_count: int = 0


###############################################################################
class DiliPriorCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[DiliPriorCatalogItem] = Field(default_factory=list)
    total: int
    offset: int
    limit: int


###############################################################################
class DiliPriorDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    annotations: list["DiliPriorAnnotation"] = Field(default_factory=list)


###############################################################################
class DiliPriorAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_dataset: str | None = None
    source_record_id: str | None = None
    source_name: str | None = None
    source_name_norm: str | None = None
    classification: str | None = None
    severity_class: str | None = None
    concern_class: str | None = None
    label_section: str | None = None
    routes: str | None = None
    comment: str | None = None
    source_url: str | None = None
    source_last_modified: str | None = None

    # -------------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


###############################################################################
class DrugLabelCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    source: str
    effective_date: str | None = None
    retained_section_count: int = 0


###############################################################################
class DrugLabelCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[DrugLabelCatalogItem] = Field(default_factory=list)
    total: int
    offset: int
    limit: int


###############################################################################
class DrugLabelSectionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    source: str
    set_id: str
    spl_version: int
    effective_date: str | None = None
    sections: list["DrugLabelSection"] = Field(default_factory=list)


###############################################################################
class DrugLabelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_key: str | None = None
    section_title: str | None = None
    text: str | None = None
    contains_hepatic_keywords: bool
    display_order: int

    # -------------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


###############################################################################
class DeleteEntityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deleted: bool


###############################################################################
class SessionListFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search: str | None = Field(default=None, max_length=MAX_SEARCH_LENGTH)
    status: SessionStatus | None = None
    date_mode: DateFilterMode | None = None
    date: DateValue | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)

    # -------------------------------------------------------------------------
    @field_validator("search", mode="before")
    @classmethod
    def normalize_search(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = CONTROL_CHARACTERS_RE.sub(" ", str(value)).strip()
        return normalized or None


###############################################################################
class CatalogListFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search: str | None = Field(default=None, max_length=MAX_SEARCH_LENGTH)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)

    # -------------------------------------------------------------------------
    @field_validator("search", mode="before")
    @classmethod
    def normalize_search(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = CONTROL_CHARACTERS_RE.sub(" ", str(value)).strip()
        return normalized or None


###############################################################################
class InspectionUpdateConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: InspectionUpdateTarget
    defaults: dict[str, Any] = Field(default_factory=dict)
    allowed_fields: list[str] = Field(default_factory=list)


###############################################################################
class InspectionRxNavOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rxnav_request_timeout: float | None = Field(default=None, ge=1.0, le=120.0)
    rxnav_max_concurrency: int | None = Field(default=None, ge=1, le=64)


###############################################################################
class InspectionLiverToxOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    livertox_monograph_max_workers: int | None = Field(default=None, ge=1, le=32)
    livertox_archive: str | None = Field(default=None, max_length=255)
    redownload: bool | None = None

    # -------------------------------------------------------------------------
    @field_validator("livertox_archive")
    @classmethod
    def validate_livertox_archive(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if "/" in normalized or "\\" in normalized:
            raise ValueError("livertox_archive must be a file name only")
        return normalized


###############################################################################
class InspectionDiliPriorsOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    redownload: bool | None = None


###############################################################################
class InspectionDrugLabelsOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dailymed_request_timeout: float | None = Field(default=None, ge=1.0, le=120.0)
    dailymed_max_concurrency: int | None = Field(default=None, ge=1, le=64)
    redownload: bool | None = None


###############################################################################
class InspectionRagOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents_path: str | None = Field(default=None, max_length=1024)
    chunk_size: int | None = Field(default=None, ge=64, le=8192)
    chunk_overlap: int | None = Field(default=None, ge=0, le=2048)
    embedding_batch_size: int | None = Field(default=None, ge=1, le=4096)
    vector_stream_batch_size: int | None = Field(default=None, ge=1, le=16384)
    embedding_max_workers: int | None = Field(default=None, ge=1, le=64)
    embedding_backend: str | None = Field(default=None, max_length=32)
    ollama_embedding_model: str | None = Field(default=None, max_length=200)
    hf_embedding_model: str | None = Field(default=None, max_length=200)
    cloud_provider: str | None = Field(default=None, max_length=32)
    cloud_embedding_model: str | None = Field(default=None, max_length=200)
    use_cloud_embeddings: bool | None = None
    reset_vector_collection: bool | None = None

    # -------------------------------------------------------------------------
    @field_validator("embedding_backend")
    @classmethod
    def validate_embedding_backend(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in {"ollama", "huggingface", "cloud"}:
            raise ValueError("Unsupported embedding_backend")
        return normalized

    # -------------------------------------------------------------------------
    @field_validator("cloud_provider")
    @classmethod
    def validate_cloud_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in {"openai", "gemini"}:
            raise ValueError("Unsupported cloud_provider")
        return normalized


###############################################################################
class RagDocumentListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    file_name: str
    extension: str
    file_size: int
    last_modified: str
    supported_for_ingestion: bool


###############################################################################
class RagDocumentListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[RagDocumentListItem] = Field(default_factory=list)
    total: int


class LanceVectorStoreSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_documents_path: str
    vector_db_path: str
    collection_name: str
    collection_exists: bool
    embedding_count: int
    distinct_document_count: int
    embedding_dimension: int | None = None
    index_ready: bool
    configured_metric: str | None = None
    configured_index_type: str | None = None


###############################################################################
class RagUpdateJobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: int = 0
    chunks: int = 0
    backend: str = "local"
