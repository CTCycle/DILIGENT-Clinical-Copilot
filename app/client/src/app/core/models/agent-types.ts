// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { CatalogProvider, EvidenceSourceId, ReasoningLevel } from "./types";

export type Capability = "assessment" | "revision" | "timeline";
export const CAPABILITIES: readonly Capability[] = ["assessment", "revision", "timeline"];
export const CAPABILITY_LABELS: Record<Capability, string> = {
  assessment: "DILI Assessment",
  revision: "DILI Report Review",
  timeline: "Timeline",
};

export type ExtractionStage = "therapy_drugs" | "anamnesis_drugs" | "diseases" | "laboratory";
export type RevisionStage = "planning" | "tool_selection" | "editing" | "qa";

export type AgentModelRef = {
  provider: CatalogProvider;
  model: string;
  reasoning_level?: ReasoningLevel | null;
};

export type AssessmentInstructions = {
  custom_instructions: string;
};

export type ReportTemplateFormat = "md" | "json" | "xml" | "txt";

export type ReportTemplate = {
  file_name: string;
  format: ReportTemplateFormat;
  content: string;
};

export type AssessmentModels = {
  extraction: AgentModelRef;
  clinical: AgentModelRef;
  extraction_overrides: Partial<Record<ExtractionStage, AgentModelRef>>;
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

export type AssessmentConfig = {
  enabled: boolean;
  instructions: AssessmentInstructions;
  report_template: ReportTemplate | null;
  models: AssessmentModels;
  reasoning_level: ReasoningLevel;
  ollama_seed: number | null;
  retrieval: AgentRetrievalPolicy;
};

export type RevisionInstructions = {
  review_objectives: string;
  editing_style: string;
};

export type RevisionModels = {
  planning: AgentModelRef;
  stage_overrides: Partial<Record<RevisionStage, AgentModelRef>>;
};

export type RevisionExecutionLimits = {
  max_tasks: number;
  max_tool_iterations: number;
  allowed_tools: string[] | null;
};

export type RevisionConfig = {
  enabled: boolean;
  instructions: RevisionInstructions;
  models: RevisionModels;
  reasoning_level: ReasoningLevel;
  ollama_seed: number | null;
  limits: RevisionExecutionLimits;
};

export type TimelineInstructions = { timeline_focus: string };

export type TimelineConfig = {
  enabled: boolean;
  instructions: TimelineInstructions;
  model: AgentModelRef;
  reasoning_level: ReasoningLevel;
  ollama_seed: number | null;
};

export type AgentConfig = {
  schema_version: 3;
  assessment: AssessmentConfig;
  revision: RevisionConfig;
  timeline: TimelineConfig;
};

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
  name: string;
  description: string;
  current_revision: number;
  is_default: boolean;
  is_active: boolean;
  is_locked: boolean;
  is_archived: boolean;
  editable: boolean;
  capabilities: Capability[];
  reference_count: number;
  created_at: string | null;
  updated_at: string | null;
  revision: AgentRevision;
};

export type AgentLifecycleRequest = {
  is_active?: boolean;
  is_locked?: boolean;
  is_archived?: boolean;
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

/** True when the agent can run new operations of the given capability. */
export function isSelectableFor(
  agent: Pick<Agent, "is_active" | "is_archived" | "capabilities">,
  capability: Capability,
): boolean {
  return agent.is_active && !agent.is_archived && agent.capabilities.includes(capability);
}

export type AgentSummary = {
  clinicalModel: string;
  extractionModel: string;
  revisionModel: string;
  timelineModel: string;
  evidenceSources: EvidenceSourceId[];
};

function formatModelRef(ref: AgentModelRef): string {
  return `${ref.provider} / ${ref.model}`;
}

/** Display strings for selector summary cards. */
export function summarizeAgent(agent: Pick<Agent, "revision">): AgentSummary {
  const config = agent.revision.configuration;
  return {
    clinicalModel: formatModelRef(config.assessment.models.clinical),
    extractionModel: formatModelRef(config.assessment.models.extraction),
    revisionModel: formatModelRef(config.revision.models.planning),
    timelineModel: formatModelRef(config.timeline.model),
    evidenceSources: [...config.assessment.retrieval.evidence_sources],
  };
}
