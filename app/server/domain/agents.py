# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Typed contracts for reusable, versioned DILIGENT agents.

An agent is a versioned configuration of an existing execution workflow
(DILI assessment or report revision). Agent records never hold provider
credentials, and nothing here can disable clinical validation: instructions
are bounded guidance layered beneath the authoritative prompt templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from domain.clinical.entities import DEFAULT_EVIDENCE_SOURCES, EvidenceSource
from domain.model_configs import CatalogProviderId, ReasoningLevel

AgentKind = Literal["dili", "revision"]
AGENT_KINDS: tuple[AgentKind, ...] = ("dili", "revision")
AGENT_CONFIG_SCHEMA_VERSION = 1

ExtractionStage = Literal["therapy_drugs", "anamnesis_drugs", "diseases", "laboratory"]
RevisionStage = Literal["planning", "tool_selection", "editing", "qa"]

INSTRUCTION_MAX_LENGTH = 4000


###############################################################################
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


###############################################################################
class AgentModelRef(_StrictModel):
    """A provider/model pair for one execution role."""

    provider: CatalogProviderId
    model: str = Field(min_length=1, max_length=255)
    # None inherits the agent-level reasoning preference.
    reasoning_level: ReasoningLevel | None = None


###############################################################################
class DiliAgentInstructions(_StrictModel):
    assessment_focus: str = Field(default="", max_length=INSTRUCTION_MAX_LENGTH)
    report_style: str = Field(default="", max_length=INSTRUCTION_MAX_LENGTH)
    extraction_guidance: str = Field(default="", max_length=INSTRUCTION_MAX_LENGTH)


###############################################################################
class DiliAgentModels(_StrictModel):
    extraction: AgentModelRef
    clinical: AgentModelRef
    timeline: AgentModelRef
    # Optional per-stage overrides; absent stages use ``extraction``.
    extraction_overrides: dict[ExtractionStage, AgentModelRef] = Field(
        default_factory=dict
    )
    # Optional final-report model; absent uses ``clinical``.
    report_synthesis: AgentModelRef | None = None


###############################################################################
class AgentRetrievalPolicy(_StrictModel):
    """Per-agent retrieval preferences.

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
class DiliAgentConfig(_StrictModel):
    kind: Literal["dili"] = "dili"
    schema_version: Literal[1] = AGENT_CONFIG_SCHEMA_VERSION
    instructions: DiliAgentInstructions = Field(default_factory=DiliAgentInstructions)
    models: DiliAgentModels
    reasoning_level: ReasoningLevel = ReasoningLevel.OFF
    ollama_seed: int | None = Field(default=42, ge=0)
    retrieval: AgentRetrievalPolicy = Field(default_factory=AgentRetrievalPolicy)


###############################################################################
class RevisionAgentInstructions(_StrictModel):
    review_objectives: str = Field(default="", max_length=INSTRUCTION_MAX_LENGTH)
    editing_style: str = Field(default="", max_length=INSTRUCTION_MAX_LENGTH)


###############################################################################
class RevisionAgentModels(_StrictModel):
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
class RevisionAgentConfig(_StrictModel):
    kind: Literal["revision"] = "revision"
    schema_version: Literal[1] = AGENT_CONFIG_SCHEMA_VERSION
    instructions: RevisionAgentInstructions = Field(
        default_factory=RevisionAgentInstructions
    )
    models: RevisionAgentModels
    reasoning_level: ReasoningLevel = ReasoningLevel.OFF
    ollama_seed: int | None = Field(default=42, ge=0)
    limits: RevisionExecutionLimits = Field(default_factory=RevisionExecutionLimits)


AgentConfig = Annotated[
    DiliAgentConfig | RevisionAgentConfig, Field(discriminator="kind")
]
AGENT_CONFIG_ADAPTER: TypeAdapter[DiliAgentConfig | RevisionAgentConfig] = TypeAdapter(
    AgentConfig
)


###############################################################################
def parse_agent_config(payload: object) -> DiliAgentConfig | RevisionAgentConfig:
    return AGENT_CONFIG_ADAPTER.validate_python(payload)


###############################################################################
def canonical_agent_config_json(
    config: DiliAgentConfig | RevisionAgentConfig,
) -> str:
    return json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


###############################################################################
def agent_config_hash(config: DiliAgentConfig | RevisionAgentConfig) -> str:
    return hashlib.sha256(
        canonical_agent_config_json(config).encode("utf-8")
    ).hexdigest()


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
    kind: AgentKind
    name: str
    description: str
    current_revision: int
    is_default: bool
    is_archived: bool
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
class AgentValidationIssue(_StrictModel):
    code: str
    field: str | None = None
    message: str


###############################################################################
class AgentValidationResponse(_StrictModel):
    valid: bool
    issues: list[AgentValidationIssue] = Field(default_factory=list)


###############################################################################
@dataclass(frozen=True)
class AgentExecutionContext:
    """Immutable agent configuration resolved once per execution launch."""

    agent_id: str
    agent_name: str
    agent_revision_id: int
    revision_number: int
    kind: AgentKind
    config_hash: str
    configuration: DiliAgentConfig | RevisionAgentConfig

    def provenance(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_revision_id": self.agent_revision_id,
            "agent_revision": self.revision_number,
            "agent_kind": self.kind,
            "agent_config_hash": self.config_hash,
            "agent_configuration": self.configuration.model_dump(mode="json"),
        }
