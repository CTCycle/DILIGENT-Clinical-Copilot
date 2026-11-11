// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Agent, AgentConfig, AgentKind, summarizeAgent } from '../models/agent-types';
import { AgentsStateService } from './agents-state.service';

const ref = { provider: 'ollama', model: 'qwen3' };

function makeConfig(kind: AgentKind): AgentConfig {
  if (kind === 'revision') {
    return {
      kind,
      schema_version: 1,
      instructions: { review_objectives: '', editing_style: '' },
      models: { planning: ref, stage_overrides: {} },
      reasoning_level: 'off',
      ollama_seed: 42,
      limits: { max_tasks: 8, max_tool_iterations: 24, allowed_tools: null },
    };
  }
  return {
    kind,
    schema_version: 1,
    instructions: { assessment_focus: '', report_style: '', extraction_guidance: '' },
    models: {
      extraction: ref,
      clinical: { provider: 'openai', model: 'gpt-x' },
      timeline: ref,
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
  };
}

function makeAgent(id: string, kind: AgentKind, overrides: Partial<Agent> = {}): Agent {
  return {
    id,
    kind,
    name: id,
    description: '',
    current_revision: 1,
    is_default: false,
    is_archived: false,
    created_at: null,
    updated_at: null,
    revision: {
      revision_id: 1,
      revision_number: 1,
      schema_version: 1,
      configuration_hash: 'h',
      configuration: makeConfig(kind),
      created_at: null,
    },
    ...overrides,
  };
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('AgentsStateService', () => {
  let service: AgentsStateService;
  const fetchSpy = vi.fn();
  const dili = makeAgent('d1', 'dili', { is_default: true });
  const rev = makeAgent('r1', 'revision', { is_default: true });
  const archived = makeAgent('d2', 'dili', { is_archived: true });

  const listCalls = () => fetchSpy.mock.calls.filter(([, init]) => init?.method === 'GET');

  beforeEach(() => {
    fetchSpy.mockReset().mockImplementation(async () => jsonResponse({ agents: [dili, rev, archived] }));
    vi.stubGlobal('fetch', fetchSpy);
    service = TestBed.inject(AgentsStateService);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('dedupes concurrent loads and skips refetch once ready', async () => {
    await Promise.all([service.load(), service.load()]);
    await service.load();
    expect(listCalls()).toHaveLength(1);
    expect(service.status()).toBe('ready');
  });

  it('exposes default selectors and excludes archived agents by default', async () => {
    await service.load();
    expect(service.agents().map((a) => a.id)).toEqual(['d1', 'r1']);
    expect(service.defaultDiliAgent()?.id).toBe('d1');
    expect(service.defaultRevisionAgent()?.id).toBe('r1');
    expect(service.byId('d2')).toBeNull();
  });

  it('includes archived agents when loaded with includeArchived', async () => {
    await service.load();
    await service.load({ includeArchived: true });
    expect(String(listCalls().at(-1)?.[0])).toContain('include_archived=true');
    expect(service.byId('d2')?.is_archived).toBe(true);
  });

  it('replaces the cached agent with the server response on save', async () => {
    await service.load();
    const saved = makeAgent('d1', 'dili', { is_default: true, current_revision: 2 });
    fetchSpy.mockImplementationOnce(async () => jsonResponse(saved));
    await service.save('d1', 1, 'd1', '', saved.revision.configuration);
    const [url, init] = fetchSpy.mock.calls.at(-1)!;
    expect(String(url)).toMatch(/\/agents\/d1$/);
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body).expected_revision).toBe(1);
    expect(service.byId('d1')?.current_revision).toBe(2);
    expect(service.agents().map((a) => a.id)).toEqual(['d1', 'r1']);
  });

  it('moves the default flag when another agent of the same kind becomes default', async () => {
    fetchSpy.mockImplementationOnce(async () =>
      jsonResponse({ agents: [dili, makeAgent('d3', 'dili'), rev] }),
    );
    await service.load();
    fetchSpy.mockImplementationOnce(async () =>
      jsonResponse(makeAgent('d3', 'dili', { is_default: true })),
    );
    await service.setDefault('d3');
    expect(service.defaultDiliAgent()?.id).toBe('d3');
    expect(service.byId('d1')?.is_default).toBe(false);
    expect(service.defaultRevisionAgent()?.id).toBe('r1');
  });

  it('records an error status when loading fails', async () => {
    fetchSpy.mockRejectedValue(new Error('boom'));
    await expect(service.load()).rejects.toThrow();
    expect(service.status()).toBe('error');
    expect(service.error()).toBeTruthy();
  });

  it('summarizes model and evidence display strings', () => {
    expect(summarizeAgent(dili)).toEqual({
      clinicalModel: 'openai / gpt-x',
      extractionModel: 'ollama / qwen3',
      timelineModel: 'ollama / qwen3',
      evidenceSources: ['livertox', 'dilirank'],
    });
  });
});
