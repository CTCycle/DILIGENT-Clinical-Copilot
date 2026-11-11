// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { CatalogProvider, EvidenceSourceId, ReasoningLevel } from "./types";

export type AgentKind = "dili" | "revision";
export type ExtractionStage = "therapy_drugs" | "anamnesis_drugs" | "diseases" | "laboratory";
export type RevisionStage = "planning" | "tool_selection" | "editing" | "qa";

export type AgentModelRef = {
  provider: CatalogProvider;
  model: string;
  reasoning_level?: ReasoningLevel | null;
};

export type DiliAgentInstructions = {
  assessment_focus: string;
  report_style: string;
  extraction_guidance: string;
};

export type DiliAgentModels = {
  extraction: AgentModelRef;
  clinical: AgentModelRef;
  timeline: AgentModelRef;
  extraction_overrides: Partial<Record<ExtractionStage, AgentModelRef>>;
  report_synthesis?: AgentModelRef | null;
};

export type AgentRetrievalPolicy = {
  evidence_sources: EvidenceSourceId[];
  use_hybrid_search: boolean;
  use_reranking: boolean;
  reranker_model: string;
  retrieval_candidate_count: number;
  retrieval_selected_count: number;
  hybrid_vector_weight: number;
  hybrid_text_weight: number;
};

export type DiliAgentConfig = {
  kind: "dili";
  schema_version: 1;
  instructions: DiliAgentInstructions;
  models: DiliAgentModels;
  reasoning_level: ReasoningLevel;
  ollama_seed: number | null;
  retrieval: AgentRetrievalPolicy;
};

export type RevisionAgentInstructions = {
  review_objectives: string;
  editing_style: string;
};

export type RevisionAgentModels = {
  planning: AgentModelRef;
  stage_overrides: Partial<Record<RevisionStage, AgentModelRef>>;
};

export type RevisionExecutionLimits = {
  max_tasks: number;
  max_tool_iterations: number;
  allowed_tools: string[] | null;
};

export type RevisionAgentConfig = {
  kind: "revision";
  schema_version: 1;
  instructions: RevisionAgentInstructions;
  models: RevisionAgentModels;
  reasoning_level: ReasoningLevel;
  ollama_seed: number | null;
  limits: RevisionExecutionLimits;
};

export type AgentConfig = DiliAgentConfig | RevisionAgentConfig;

export type AgentRevision = {
  revision_id: number;
  revision_number: number;
  schema_version: number;
  configuration_hash: string;
  configuration: AgentConfig;
  created_at: string | null;
};

export type Agent = {
  id: string;
  kind: AgentKind;
  name: string;
  description: string;
  current_revision: number;
  is_default: boolean;
  is_archived: boolean;
  created_at: string | null;
  updated_at: string | null;
  revision: AgentRevision;
};

export type AgentListResponse = { agents: Agent[] };
export type AgentRevisionListResponse = { revisions: AgentRevision[] };

export type AgentCreateRequest = {
  name: string;
  description: string;
  configuration: AgentConfig;
};

export type AgentUpdateRequest = {
  expected_revision: number;
  name: string;
  description: string;
  configuration: AgentConfig;
};

export type AgentDuplicateRequest = { name?: string | null };

export type AgentValidationIssue = {
  code: string;
  field: string | null;
  message: string;
};

export type AgentValidationResponse = {
  valid: boolean;
  issues: AgentValidationIssue[];
};

export type AgentSummary = {
  clinicalModel: string | null;
  extractionModel: string | null;
  timelineModel: string | null;
  evidenceSources: EvidenceSourceId[];
};

function formatModelRef(ref: AgentModelRef): string {
  return `${ref.provider} / ${ref.model}`;
}

/** Display strings for wizard and summary cards. Revision agents expose only their planning model. */
export function summarizeAgent(agent: Pick<Agent, "revision">): AgentSummary {
  const config = agent.revision.configuration;
  if (config.kind === "dili") {
    return {
      clinicalModel: formatModelRef(config.models.clinical),
      extractionModel: formatModelRef(config.models.extraction),
      timelineModel: formatModelRef(config.models.timeline),
      evidenceSources: [...config.retrieval.evidence_sources],
    };
  }
  return {
    clinicalModel: formatModelRef(config.models.planning),
    extractionModel: null,
    timelineModel: null,
    evidenceSources: [],
  };
}
