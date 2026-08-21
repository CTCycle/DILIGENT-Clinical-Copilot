from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CloudProviderId = Literal[
    "openai", "gemini", "deepseek", "anthropic", "opencode_zen", "opencode_go"
]
CredentialProviderId = Literal[
    "openai", "gemini", "deepseek", "anthropic", "opencode", "brave"
]
ModelDiscoveryStrategy = Literal["api", "static"]
TransportStrategy = Literal[
    "openai_responses",
    "openai_chat_completions",
    "anthropic_messages",
    "gemini_generate_content",
    "model_metadata_routed",
]

###############################################################################
class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    chat: bool
    structured_output: bool
    reasoning: bool
    model_listing: bool
    embeddings: bool
    vision: bool

###############################################################################
class CloudProviderDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider_id: CloudProviderId
    display_name: str = Field(min_length=1)
    credential_scope: CredentialProviderId
    discovery_strategy: ModelDiscoveryStrategy
    models_endpoint: str | None = None
    default_model: str | None = None
    models: tuple[str, ...] = ()
    capabilities: ProviderCapabilities
    transport_strategy: TransportStrategy

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_catalog(self) -> "CloudProviderDefinition":
        if self.discovery_strategy == "static" and not self.models:
            raise ValueError("static providers require models")
        if (
            self.default_model is not None
            and self.models
            and self.default_model not in self.models
        ):
            raise ValueError("default_model must exist in the provider model catalog")
        return self

###############################################################################
class CloudModelDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    display_name: str
    endpoint_family: str | None = None
    capabilities: ProviderCapabilities | None = None
    input_token_limit: int | None = Field(default=None, ge=1)
    output_token_limit: int | None = Field(default=None, ge=1)
    supports_thinking: bool | None = None
    supports_temperature: bool | None = None

###############################################################################
class CloudProviderDescriptor(BaseModel):
    id: CloudProviderId
    display_name: str
    credential_scope: CredentialProviderId
    capabilities: ProviderCapabilities
    catalog_status: Literal[
        "available", "cached", "not_loaded", "unavailable", "authentication_required"
    ]
    catalog_updated_at: datetime | None = None
    catalog_message: str | None = None
    models: list[CloudModelDescriptor]
