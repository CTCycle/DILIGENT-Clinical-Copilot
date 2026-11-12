// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Agent, AgentConfig, AgentModelRef } from '../../core/models/agent-types';

export const INSTRUCTION_MAX_LENGTH = 4000;
export const REPORT_TEMPLATE_MAX_LENGTH = 20000;
export const NAME_MAX_LENGTH = 120;
export const DESCRIPTION_MAX_LENGTH = 2000;

/** Mirrors `RevisionToolRegistry.names` (app/server/services/inspection/revision_tools.py). */
export const REVISION_TOOL_NAMES: readonly string[] = [
  'read_session_context',
  'read_result_payload_path',
  'read_manual_edits',
  'read_version_lineage',
  'search_livertox_catalog',
  'get_livertox_excerpt',
  'search_dilirank_catalog',
  'get_dilirank_records',
  'get_drug_knowledge_bundle',
  'search_rag',
];

export type ProviderOption = { value: string; label: string };
export type ModelCatalog = Record<string, readonly string[]>;

/** Sensible defaults for a new agent: a copy of the default agent's configuration, all capabilities enabled. */
export function defaultNewAgentConfig(seed: Agent | null): AgentConfig | null {
  if (!seed) return null;
  const config = structuredClone(seed.revision.configuration);
  config.assessment.enabled = true;
  config.revision.enabled = true;
  config.timeline.enabled = true;
  return config;
}

function checkRef(errors: string[], label: string, ref: AgentModelRef | null | undefined): void {
  if (!ref) return;
  if (!ref.provider.trim() || !ref.model.trim()) {
    errors.push(`${label}: choose a provider and a model.`);
  }
}

function checkSeed(errors: string[], seed: number | null): void {
  if (seed !== null && (!Number.isInteger(seed) || seed < 0)) {
    errors.push('Ollama seed must be a non-negative whole number.');
  }
}

function checkInstructions(errors: string[], instructions: object): void {
  for (const [field, text] of Object.entries(instructions)) {
    if (String(text).length > INSTRUCTION_MAX_LENGTH) {
      errors.push(`${field.replace(/_/g, ' ')} must be at most ${INSTRUCTION_MAX_LENGTH} characters.`);
    }
  }
}

/**
 * Client-side mirror of the server constraints, so obvious mistakes never reach the API.
 * Every capability is checked, enabled or not, because its values are persisted either way.
 */
export function validateAgentDraft(
  name: string,
  description: string,
  config: AgentConfig,
): string[] {
  const errors: string[] = [];
  if (!name.trim()) errors.push('Name is required.');
  if (name.length > NAME_MAX_LENGTH) errors.push(`Name must be at most ${NAME_MAX_LENGTH} characters.`);
  if (description.length > DESCRIPTION_MAX_LENGTH) {
    errors.push(`Description must be at most ${DESCRIPTION_MAX_LENGTH} characters.`);
  }

  const { assessment, revision, timeline } = config;
  checkInstructions(errors, assessment.instructions);
  checkInstructions(errors, revision.instructions);
  checkInstructions(errors, timeline.instructions);
  checkSeed(errors, assessment.ollama_seed);
  checkSeed(errors, revision.ollama_seed);
  checkSeed(errors, timeline.ollama_seed);

  const { models, retrieval } = assessment;
  checkRef(errors, 'Extraction model', models.extraction);
  checkRef(errors, 'Clinical model', models.clinical);
  Object.entries(models.extraction_overrides).forEach(([stage, ref]) =>
    checkRef(errors, `Extraction override (${stage})`, ref),
  );
  if (!retrieval.reranker_model.trim()) errors.push('Reranker model is required.');
  const counts = [retrieval.retrieval_candidate_count, retrieval.retrieval_selected_count];
  if (counts.some((n) => !Number.isInteger(n) || n < 1 || n > 10000)) {
    errors.push('Candidate and selected counts must be whole numbers between 1 and 10000.');
  } else if (counts[0] < counts[1]) {
    errors.push('Candidate count must be at least the selected count.');
  }
  for (const weight of [retrieval.hybrid_vector_weight, retrieval.hybrid_text_weight]) {
    if (!(weight >= 0 && weight <= 1)) errors.push('Hybrid weights must be between 0 and 1.');
  }

  checkRef(errors, 'Planning model', revision.models.planning);
  Object.entries(revision.models.stage_overrides).forEach(([stage, ref]) =>
    checkRef(errors, `Stage override (${stage})`, ref),
  );
  const { max_tasks, max_tool_iterations } = revision.limits;
  if (!Number.isInteger(max_tasks) || max_tasks < 1 || max_tasks > 8) {
    errors.push('Max tasks must be a whole number between 1 and 8.');
  }
  if (!Number.isInteger(max_tool_iterations) || max_tool_iterations < 1 || max_tool_iterations > 24) {
    errors.push('Max tool iterations must be a whole number between 1 and 24.');
  }

  checkRef(errors, 'Timeline model', timeline.model);
  return Array.from(new Set(errors));
}
