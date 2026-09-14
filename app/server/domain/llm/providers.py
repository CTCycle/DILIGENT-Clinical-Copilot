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
ReasoningParameter = Literal[
    "none", "boolean", "level", "effort", "budget_tokens", "adaptive"
]
ReasoningToggleParameter = Literal["none", "thinking", "reasoning", "provider_default"]
ToolCallMode = Literal["native", "structured", "unsupported"]
CapabilityEvidence = Literal[
    "catalog", "documented", "probe", "provider", "fallback"
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
class ModelCapabilityMetadata(BaseModel):
    """Canonical capability metadata for one provider model.

    Provider-level capabilities describe what a provider can do in general.  This
    model describes the selected model and is the only metadata source used when
    constructing an inference request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_model_id: str | None = None
    aliases: tuple[str, ...] = ()
    endpoint_family: str | None = None
    context_window_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    supports_chat: bool | None = None
    supports_streaming: bool | None = None
    supports_tools: bool | None = None
    tool_call_mode: ToolCallMode = "unsupported"
    supports_tool_choice: bool | None = None
    supports_parallel_tool_calls: bool | None = None
    supports_structured_output: bool | None = None
    supports_json_mode: bool | None = None
    supports_native_json_schema: bool | None = None
    supports_reasoning: bool | None = None
    reasoning_levels: tuple[str, ...] = ()
    reasoning_parameter: ReasoningParameter = "none"
    reasoning_toggle_parameter: ReasoningToggleParameter = "provider_default"
    reasoning_counts_toward_output_limit: bool | None = None
    reasoning_unsupported_parameters: tuple[str, ...] = ()
    requires_reasoning_content_for_tool_calls: bool | None = None
    requires_assistant_content_for_tool_calls: bool | None = None
    supports_temperature: bool | None = None
    supports_top_p: bool | None = None
    supports_usage_metadata: bool | None = None
    supports_finish_reason: bool | None = None
    evidence: CapabilityEvidence = "fallback"

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
    model_capabilities: ModelCapabilityMetadata = Field(
        default_factory=ModelCapabilityMetadata
    )
    # These fields remain as a serialization bridge for existing cached catalog
    # rows and clients.  Runtime code reads model_capabilities instead.
    input_token_limit: int | None = Field(default=None, ge=1)
    output_token_limit: int | None = Field(default=None, ge=1)
    supports_thinking: bool | None = None
    supports_temperature: bool | None = None
    supports_json_mode: bool | None = None
    supports_native_json_schema: bool | None = None

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def normalize_model_capabilities(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw = data.get("model_capabilities")
        if isinstance(raw, dict):
            metadata = dict(raw)
        elif isinstance(raw, ModelCapabilityMetadata):
            metadata = raw.model_dump(mode="python")
        else:
            metadata = {}

        legacy_to_canonical = {
            "input_token_limit": "context_window_tokens",
            "output_token_limit": "max_output_tokens",
            "supports_thinking": "supports_reasoning",
            "supports_temperature": "supports_temperature",
            "supports_json_mode": "supports_json_mode",
            "supports_native_json_schema": "supports_native_json_schema",
        }
        for legacy_name, canonical_name in legacy_to_canonical.items():
            if canonical_name not in metadata and data.get(legacy_name) is not None:
                metadata[canonical_name] = data[legacy_name]
        if data.get("endpoint_family") and "endpoint_family" not in metadata:
            metadata["endpoint_family"] = data["endpoint_family"]
        provider_capabilities = data.get("capabilities")
        if isinstance(provider_capabilities, dict):
            if (
                "supports_structured_output" not in metadata
                and provider_capabilities.get("structured_output") is not None
            ):
                metadata["supports_structured_output"] = provider_capabilities[
                    "structured_output"
                ]
            if (
                "supports_reasoning" not in metadata
                and provider_capabilities.get("reasoning") is not None
            ):
                metadata["supports_reasoning"] = provider_capabilities["reasoning"]
        data["model_capabilities"] = metadata

        # Keep old fields populated when new catalog entries only provide the
        # canonical object, so the current API can be upgraded atomically.
        reverse = {
            "context_window_tokens": "input_token_limit",
            "max_output_tokens": "output_token_limit",
            "supports_reasoning": "supports_thinking",
            "supports_temperature": "supports_temperature",
            "supports_json_mode": "supports_json_mode",
            "supports_native_json_schema": "supports_native_json_schema",
        }
        for canonical_name, legacy_name in reverse.items():
            if data.get(legacy_name) is None and metadata.get(canonical_name) is not None:
                data[legacy_name] = metadata[canonical_name]
        return data

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
