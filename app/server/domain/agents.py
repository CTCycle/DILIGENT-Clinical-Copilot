# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Typed contracts for reusable, versioned DILIGENT agents.

An agent is one persistent entity that bundles three independently enabled
capabilities: DILI assessment, report revision and patient timeline. Each
capability owns its configuration; disabling one keeps its values but makes
the agent incompatible with that workflow. Agent records never hold provider
credentials, and nothing here can disable clinical validation: instructions
are bounded guidance layered beneath the authoritative prompt templates.

Lifecycle (flags on the agent, independent from configuration revisions):
``editable`` = active and unlocked and not archived;
``selectable(capability)`` = active and not archived and capability enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domain.clinical.entities import DEFAULT_EVIDENCE_SOURCES, EvidenceSource
from domain.model_configs import CatalogProviderId, ReasoningLevel

Capability = Literal["assessment", "revision", "timeline"]
CAPABILITIES: tuple[Capability, ...] = ("assessment", "revision", "timeline")
AGENT_CONFIG_SCHEMA_VERSION = 3

ExtractionStage = Literal["therapy_drugs", "anamnesis_drugs", "diseases", "laboratory"]
RevisionStage = Literal["planning", "tool_selection", "editing", "qa"]
ReportTemplateFormat = Literal["md", "json", "xml", "txt"]

INSTRUCTION_MAX_LENGTH = 4000
REPORT_TEMPLATE_MAX_LENGTH = 20000


###############################################################################
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


###############################################################################
class AgentModelRef(_StrictModel):
    """A provider/model pair for one execution role."""

    provider: CatalogProviderId
    model: str = Field(min_length=1, max_length=255)
    # None inherits the capability-level reasoning preference.
    reasoning_level: ReasoningLevel | None = None


###############################################################################
class AgentRetrievalPolicy(_StrictModel):
    """Per-capability retrieval preferences.

    Physical index settings (chunking, embedding batch size, vector index
    type/metric, collection name) are shared infrastructure, not agent
    configuration: changing an agent must never rebuild a shared index.
    """

    evidence_sources: list[EvidenceSource] = Field(
        default_factory=lambda: list(DEFAULT_EVIDENCE_SOURCES)
    )
    use_hybrid_search: bool = True
    use_reranking: bool = True
    reranker_model: str = Field(
        default="lightweight-balanced-v1", min_length=1, max_length=255
    )
    retrieval_candidate_count: int = Field(default=40, ge=1, le=10000)
    retrieval_selected_count: int = Field(default=6, ge=1, le=10000)
    hybrid_vector_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    hybrid_text_weight: float = Field(default=0.35, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_counts(self) -> AgentRetrievalPolicy:
        if len(set(self.evidence_sources)) != len(self.evidence_sources):
            raise ValueError("Evidence sources must be unique.")
        if self.retrieval_candidate_count < self.retrieval_selected_count:
            raise ValueError(
                "retrieval_candidate_count must be >= retrieval_selected_count."
            )
        return self

    def rag_settings(self) -> dict[str, object]:
        """Return the RAG settings keys this policy owns."""

        return self.model_dump(exclude={"evidence_sources"})


AGENT_RAG_SETTING_KEYS: frozenset[str] = frozenset(
    AgentRetrievalPolicy.model_fields.keys() - {"evidence_sources"}
)


###############################################################################
class AssessmentInstructions(_StrictModel):
    custom_instructions: str = Field(default="", max_length=INSTRUCTION_MAX_LENGTH)


###############################################################################
class ReportTemplate(_StrictModel):
    """A user-supplied text template governing the report's presentation."""

    file_name: str = Field(min_length=1, max_length=255)
    format: ReportTemplateFormat
    content: str = Field(min_length=1, max_length=REPORT_TEMPLATE_MAX_LENGTH)

    @model_validator(mode="after")
    def _validate_content(self) -> ReportTemplate:
        if not self.content.strip():
            raise ValueError("Report template content must not be blank.")
        return self


###############################################################################
class AssessmentModels(_StrictModel):
    extraction: AgentModelRef
    clinical: AgentModelRef
    # Optional per-stage overrides; absent stages use ``extraction``.
    extraction_overrides: dict[ExtractionStage, AgentModelRef] = Field(
        default_factory=dict
    )


###############################################################################
class AssessmentConfig(_StrictModel):
    enabled: bool = True
    instructions: AssessmentInstructions = Field(default_factory=AssessmentInstructions)
    # None keeps the default report formatting instructions.
    report_template: ReportTemplate | None = None
    models: AssessmentModels
    reasoning_level: ReasoningLevel = ReasoningLevel.OFF
    ollama_seed: int | None = Field(default=42, ge=0)
    retrieval: AgentRetrievalPolicy = Field(default_factory=AgentRetrievalPolicy)


###############################################################################
class RevisionInstructions(_StrictModel):
    review_objectives: str = Field(default="", max_length=INSTRUCTION_MAX_LENGTH)
    editing_style: str = Field(default="", max_length=INSTRUCTION_MAX_LENGTH)


###############################################################################
class RevisionModels(_StrictModel):
    planning: AgentModelRef
    # Optional per-stage overrides; absent stages use ``planning``.
    stage_overrides: dict[RevisionStage, AgentModelRef] = Field(default_factory=dict)


###############################################################################
class RevisionExecutionLimits(_StrictModel):
    max_tasks: int = Field(default=8, ge=1, le=8)
    max_tool_iterations: int = Field(default=24, ge=1, le=24)
    # None allows every registered revision tool.
    allowed_tools: list[str] | None = Field(default=None, max_length=12)

    @model_validator(mode="after")
    def _validate_tools(self) -> RevisionExecutionLimits:
        if self.allowed_tools is not None and len(set(self.allowed_tools)) != len(
            self.allowed_tools
        ):
            raise ValueError("Allowed tools must be unique.")
        return self


###############################################################################
class RevisionConfig(_StrictModel):
    enabled: bool = True
    instructions: RevisionInstructions = Field(default_factory=RevisionInstructions)
    models: RevisionModels
    reasoning_level: ReasoningLevel = ReasoningLevel.OFF
    ollama_seed: int | None = Field(default=42, ge=0)
    limits: RevisionExecutionLimits = Field(default_factory=RevisionExecutionLimits)


###############################################################################
class TimelineInstructions(_StrictModel):
    timeline_focus: str = Field(default="", max_length=INSTRUCTION_MAX_LENGTH)


###############################################################################
class TimelineConfig(_StrictModel):
    enabled: bool = True
    instructions: TimelineInstructions = Field(default_factory=TimelineInstructions)
    model: AgentModelRef
    reasoning_level: ReasoningLevel = ReasoningLevel.OFF
    ollama_seed: int | None = Field(default=42, ge=0)


CapabilityConfig = AssessmentConfig | RevisionConfig | TimelineConfig


###############################################################################
class AgentConfig(_StrictModel):
    """The complete persisted configuration of one agent revision."""

    schema_version: Literal[3] = AGENT_CONFIG_SCHEMA_VERSION
    assessment: AssessmentConfig
    revision: RevisionConfig
    timeline: TimelineConfig

    def capability(self, name: Capability) -> CapabilityConfig:
        return getattr(self, name)

    def enabled_capabilities(self) -> list[Capability]:
        return [name for name in CAPABILITIES if self.capability(name).enabled]


###############################################################################
def parse_agent_config(payload: object) -> AgentConfig:
    return AgentConfig.model_validate(payload)


###############################################################################
def canonical_agent_config_json(config: AgentConfig) -> str:
    return json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


###############################################################################
def agent_config_hash(config: AgentConfig) -> str:
    return hashlib.sha256(
        canonical_agent_config_json(config).encode("utf-8")
    ).hexdigest()


###############################################################################
def agent_is_editable(*, is_active: bool, is_locked: bool, is_archived: bool) -> bool:
    return is_active and not is_locked and not is_archived


###############################################################################
def agent_is_selectable(
    config: AgentConfig,
    capability: Capability,
    *,
    is_active: bool,
    is_archived: bool,
) -> bool:
    return is_active and not is_archived and config.capability(capability).enabled


###############################################################################
class AgentRevisionResponse(_StrictModel):
    revision_id: int
    revision_number: int
    schema_version: int
    configuration_hash: str
    configuration: AgentConfig
    created_at: datetime | None = None


###############################################################################
class AgentResponse(_StrictModel):
    id: str
    name: str
    description: str
    current_revision: int
    is_default: bool
    is_active: bool
    is_locked: bool
    is_archived: bool
    editable: bool
    capabilities: list[Capability]
    reference_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    revision: AgentRevisionResponse


###############################################################################
class AgentListResponse(_StrictModel):
    agents: list[AgentResponse]


###############################################################################
class AgentRevisionListResponse(_StrictModel):
    revisions: list[AgentRevisionResponse]


###############################################################################
class AgentCreateRequest(_StrictModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    configuration: AgentConfig


###############################################################################
class AgentUpdateRequest(_StrictModel):
    expected_revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    configuration: AgentConfig


###############################################################################
class AgentDuplicateRequest(_StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)


###############################################################################
class AgentLifecycleRequest(_StrictModel):
    """Partial lifecycle change; omitted flags are left untouched."""

    is_active: bool | None = None
    is_locked: bool | None = None
    is_archived: bool | None = None

    @model_validator(mode="after")
    def _require_change(self) -> AgentLifecycleRequest:
        if (
            self.is_active is None
            and self.is_locked is None
            and self.is_archived is None
        ):
            raise ValueError("At least one lifecycle flag is required.")
        return self


###############################################################################
class AgentValidationIssue(_StrictModel):
    code: str
    field: str | None = None
    message: str




###############################################################################
@dataclass(frozen=True)
class AgentExecutionContext:
    """Immutable agent capability configuration resolved once per launch."""

    agent_id: str
    agent_name: str
    agent_revision_id: int
    revision_number: int
    capability: Capability
    config_hash: str
    configuration: CapabilityConfig

    def provenance(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_revision_id": self.agent_revision_id,
            "agent_revision": self.revision_number,
            "agent_capability": self.capability,
            "agent_config_hash": self.config_hash,
            "agent_configuration": self.configuration.model_dump(mode="json"),
        }
