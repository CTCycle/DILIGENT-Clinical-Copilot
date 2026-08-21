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

class ReasoningLevel(StrEnum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

###############################################################################
@dataclass(frozen=True)
class ModelConfigSnapshot:
    clinical_model: str | None
    text_extraction_model: str | None
    use_cloud_models: bool
    cloud_provider: str | None
    cloud_model: str | None
    revision_model: str | None = None
    timeline_model: str | None = None
    reasoning_level: ReasoningLevel = ReasoningLevel.OFF
    ollama_seed: int | None = 42
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
class RagSettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_batch_size: int | None = None
    use_hybrid_search: bool | None = None
    use_reranking: bool | None = None
    retrieval_candidate_count: int | None = None
    retrieval_selected_count: int | None = None
    reranker_model: str | None = None
    hybrid_vector_weight: float | None = None
    hybrid_text_weight: float | None = None
    vector_stream_batch_size: int | None = None
    embedding_offline_mode: bool | None = None

###############################################################################
class RagSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    use_hybrid_search: bool
    use_reranking: bool
    retrieval_candidate_count: int
    retrieval_selected_count: int
    reranker_model: str
    hybrid_vector_weight: float
    hybrid_text_weight: float
    vector_stream_batch_size: int
    embedding_offline_mode: bool

###############################################################################
class ModelConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_cloud_services: bool | None = None
    llm_provider: CloudProviderId | None = None
    cloud_model: str | None = None
    text_extraction_model: str | None = None
    clinical_model: str | None = None
    revision_model: str | None = None
    timeline_model: str | None = None
    reasoning_level: ReasoningLevel | None = None
    ollama_seed: int | None = Field(default=None, ge=0)
    rag_settings: RagSettingsUpdateRequest | None = None

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
    use_cloud_services: bool
    llm_provider: CloudProviderId
    cloud_model: str | None
    text_extraction_model: str | None
    clinical_model: str | None
    revision_model: str | None
    timeline_model: str | None
    reasoning_level: ReasoningLevel
    ollama_seed: int | None
    rag_settings: RagSettingsResponse
    embedding_runtime: EmbeddingRuntimeStatus
    embedding_index: EmbeddingIndexStatus
    updated_at: datetime | None = None

###############################################################################
class ModelConfigPersistResponse(BaseModel):
    """Configuration values returned after a persistence-only update."""

    use_cloud_services: bool
    llm_provider: CloudProviderId
    cloud_model: str | None
    text_extraction_model: str | None
    clinical_model: str | None
    revision_model: str | None
    timeline_model: str | None
    reasoning_level: ReasoningLevel
    ollama_seed: int | None
    rag_settings: RagSettingsResponse
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
