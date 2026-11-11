# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from domain.llm.providers import CloudProviderDescriptor, CloudProviderId

CatalogProviderId = Literal[
    "ollama",
    "openai",
    "gemini",
    "deepseek",
    "anthropic",
    "opencode_zen",
    "opencode_go",
]
CatalogStatus = Literal[
    "available", "cached", "not_loaded", "unavailable", "authentication_required"
]


###############################################################################
class ReasoningLevel(StrEnum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


###############################################################################
@dataclass(frozen=True)
class ModelConfigSnapshot:
    """Shared, non-execution configuration; agents own models and roles."""

    rag_settings: dict[str, object] | None = None
    updated_at: datetime | None = None


###############################################################################
class LocalModelCard(BaseModel):
    name: str
    family: str
    description: str
    available_in_ollama: bool
    recommended_for_local_extraction: bool = False
    recommended_rank: int | None = None


###############################################################################
class LocalCatalogMetadata(BaseModel):
    status: CatalogStatus
    updated_at: datetime | None = None
    message: str | None = None


###############################################################################
class ModelCatalogOperationResponse(BaseModel):
    catalog_provider: CatalogProviderId
    outcome: Literal["cached", "refreshed", "failed"]
    error: str | None = None
    state: "ModelConfigStateResponse"


###############################################################################
class SharedRagSettingsUpdateRequest(BaseModel):
    """Shared knowledge-index settings; retrieval preferences live on agents."""

    model_config = ConfigDict(extra="forbid")
    allow_local_filesystem_access: bool | None = None
    vector_collection_name: str | None = Field(default=None, min_length=1, max_length=255)
    chunk_size: int | None = Field(default=None, ge=1, le=100000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=100000)
    embedding_batch_size: int | None = Field(default=None, ge=1, le=4096)
    vector_index_metric: str | None = Field(default=None, min_length=1, max_length=32)
    vector_index_type: str | None = Field(default=None, min_length=1, max_length=64)
    vector_stream_batch_size: int | None = Field(default=None, ge=1, le=100000)
    embedding_offline_mode: bool | None = None


###############################################################################
class SharedRagSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_local_filesystem_access: bool
    vector_collection_name: str
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    vector_index_metric: str
    vector_index_type: str
    vector_stream_batch_size: int
    embedding_offline_mode: bool


###############################################################################
class ModelConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rag_settings: SharedRagSettingsUpdateRequest | None = None


###############################################################################
class EmbeddingRuntimeStatus(BaseModel):
    model_display_name: str
    model_revision: str
    device: str
    cache_status: str
    loaded: bool


###############################################################################
class EmbeddingIndexStatus(BaseModel):
    status: str
    fingerprint: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    built_at: datetime | None = None


###############################################################################
class ModelConfigStateResponse(BaseModel):
    local_models: list[LocalModelCard]
    cloud_providers: list[CloudProviderDescriptor]
    local_catalog: LocalCatalogMetadata
    rag_settings: SharedRagSettingsResponse
    embedding_runtime: EmbeddingRuntimeStatus
    embedding_index: EmbeddingIndexStatus
    updated_at: datetime | None = None


###############################################################################
class ModelConfigPersistResponse(BaseModel):
    """Configuration values returned after a persistence-only update."""

    rag_settings: SharedRagSettingsResponse
    updated_at: datetime | None = None


###############################################################################
class EmbeddingStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_runtime: EmbeddingRuntimeStatus
    embedding_index: EmbeddingIndexStatus


###############################################################################
class ConnectivityCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: CloudProviderId
    model: str = Field(min_length=1)


###############################################################################
class ConnectivityCheckResponse(BaseModel):
    provider: CloudProviderId
    model: str
    ok: bool
    response_preview: str | None = None
    error: str | None = None
