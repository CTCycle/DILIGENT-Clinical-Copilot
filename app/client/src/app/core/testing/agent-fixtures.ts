// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Agent, AgentConfig, Capability } from '../models/agent-types';

const OLLAMA = { provider: 'ollama', model: 'qwen3' } as const;

/** A complete three-capability configuration; override any capability slice. */
export function makeAgentConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
  return {
    schema_version: 3,
    assessment: {
      enabled: true,
      instructions: { custom_instructions: '' },
      report_template: null,
      models: {
        extraction: { ...OLLAMA },
        clinical: { provider: 'openai', model: 'gpt-x' },
        extraction_overrides: {},
      },
      reasoning_level: 'off',
      ollama_seed: 42,
      retrieval: {
        evidence_sources: ['livertox', 'dilirank'],
        use_hybrid_search: true,
        use_reranking: true,
        reranker_model: 'r',
        retrieval_candidate_count: 40,
        retrieval_selected_count: 6,
        hybrid_vector_weight: 0.65,
        hybrid_text_weight: 0.35,
      },
    },
    revision: {
      enabled: true,
      instructions: { review_objectives: '', editing_style: '' },
      models: { planning: { ...OLLAMA }, stage_overrides: {} },
      reasoning_level: 'off',
      ollama_seed: 42,
      limits: { max_tasks: 8, max_tool_iterations: 24, allowed_tools: null },
    },
    timeline: {
      enabled: true,
      instructions: { timeline_focus: '' },
      model: { ...OLLAMA },
      reasoning_level: 'off',
      ollama_seed: 42,
    },
    ...overrides,
  };
}

/** An active, unlocked agent with every capability enabled. */
export function makeAgent(
  id: string,
  overrides: Partial<Agent> = {},
  config: AgentConfig = makeAgentConfig(),
): Agent {
  const capabilities: Capability[] = (['assessment', 'revision', 'timeline'] as const).filter(
    (name) => config[name].enabled,
  );
  return {
    id,
    name: id,
    description: '',
    current_revision: 1,
    is_default: false,
    is_active: true,
    is_locked: false,
    is_archived: false,
    editable: true,
    capabilities,
    reference_count: 0,
    created_at: null,
    updated_at: null,
    revision: {
      revision_id: 1,
      revision_number: 1,
      schema_version: 2,
      configuration_hash: 'h',
      configuration: config,
      created_at: null,
    },
    ...overrides,
  };
}

/** Disable the given capabilities on a configuration, keeping their values. */
export function withDisabled(config: AgentConfig, ...capabilities: Capability[]): AgentConfig {
  const copy = structuredClone(config);
  for (const name of capabilities) copy[name].enabled = false;
  return copy;
}
