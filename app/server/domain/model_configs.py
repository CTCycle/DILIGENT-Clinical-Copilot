from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from domain.llm.providers import CloudProviderDescriptor, CloudProviderId

###############################################################################
@dataclass(frozen=True)
class ModelConfigSnapshot:
    clinical_model: str | None
    text_extraction_model: str | None
    use_cloud_models: bool
    cloud_provider: str | None
    cloud_model: str | None
    ollama_reasoning: bool = False
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
class ModelConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_cloud_services: bool | None = None
    llm_provider: CloudProviderId | None = None
    cloud_model: str | None = None
    text_extraction_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("text_extraction_model", "text_extraction_model"),
    )
    clinical_model: str | None = None
    ollama_reasoning: bool | None = None
    ollama_seed: int | None = Field(default=None, ge=0)
    rag_settings: dict[str, object] | None = None

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
    use_cloud_services: bool
    llm_provider: CloudProviderId
    cloud_model: str | None
    text_extraction_model: str | None
    clinical_model: str | None
    ollama_reasoning: bool
    ollama_seed: int | None
    rag_settings: dict[str, object]
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
    ollama_reasoning: bool
    ollama_seed: int | None
    rag_settings: dict[str, object]
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
