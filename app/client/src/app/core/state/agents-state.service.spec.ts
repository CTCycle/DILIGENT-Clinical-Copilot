// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { summarizeAgent } from '../models/agent-types';
import { makeAgent, makeAgentConfig, withDisabled } from '../testing/agent-fixtures';
import { AgentsStateService } from './agents-state.service';

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('AgentsStateService', () => {
  let service: AgentsStateService;
  const fetchSpy = vi.fn();
  const full = makeAgent('a1', { is_default: true });
  const assessmentOnly = makeAgent(
    'a2',
    {},
    withDisabled(makeAgentConfig(), 'revision', 'timeline'),
  );
  const inactive = makeAgent('a3', { is_active: false });
  const archived = makeAgent('a4', { is_archived: true });

  const listCalls = () => fetchSpy.mock.calls.filter(([, init]) => init?.method === 'GET');

  beforeEach(() => {
    fetchSpy
      .mockReset()
      .mockImplementation(async () =>
        jsonResponse({ agents: [full, assessmentOnly, inactive, archived] }),
      );
    vi.stubGlobal('fetch', fetchSpy);
    service = TestBed.inject(AgentsStateService);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('dedupes concurrent loads, skips refetch once ready, and always loads archived agents', async () => {
    await Promise.all([service.load(), service.load()]);
    await service.load();
    expect(listCalls()).toHaveLength(1);
    expect(String(listCalls()[0][0])).toContain('include_archived=true');
    expect(service.status()).toBe('ready');
  });

  it('hides archived agents from the list until requested, but still resolves them by id', async () => {
    await service.load();
    expect(service.agents().map((a) => a.id)).toEqual(['a1', 'a2', 'a3']);
    expect(service.byId('a4')?.is_archived).toBe(true);
    service.showArchived.set(true);
    expect(service.agents().map((a) => a.id)).toEqual(['a1', 'a2', 'a3', 'a4']);
  });

  it('offers only active, non-archived agents whose capability is enabled', async () => {
    await service.load();
    expect(service.selectableFor('assessment').map((a) => a.id)).toEqual(['a1', 'a2']);
    expect(service.selectableFor('revision').map((a) => a.id)).toEqual(['a1']);
    expect(service.selectableFor('timeline').map((a) => a.id)).toEqual(['a1']);
  });

  it('prefers the default agent and falls back to the first compatible agent', async () => {
    await service.load();
    expect(service.preferredFor('assessment')?.id).toBe('a1');
    service.showArchived.set(false);
    fetchSpy.mockImplementationOnce(async () =>
      jsonResponse({ agents: [makeAgent('x', { is_default: true, is_active: false }), assessmentOnly] }),
    );
    await service.load({ force: true });
    expect(service.preferredFor('assessment')?.id).toBe('a2');
    expect(service.preferredFor('timeline')).toBeNull();
  });

  it('replaces the cached agent with the server response on save', async () => {
    await service.load();
    const saved = makeAgent('a1', { is_default: true, current_revision: 2 });
    fetchSpy.mockImplementationOnce(async () => jsonResponse(saved));
    await service.save('a1', 1, 'a1', '', saved.revision.configuration);
    const [url, init] = fetchSpy.mock.calls.at(-1)!;
    expect(String(url)).toMatch(/\/agents\/a1$/);
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body).expected_revision).toBe(1);
    expect(service.byId('a1')?.current_revision).toBe(2);
  });

  it('moves the single default flag when another agent becomes default', async () => {
    await service.load();
    fetchSpy.mockImplementationOnce(async () =>
      jsonResponse(makeAgent('a2', { is_default: true })),
    );
    await service.setDefault('a2');
    expect(service.defaultAgent()?.id).toBe('a2');
    expect(service.byId('a1')?.is_default).toBe(false);
  });

  it('applies lifecycle changes from the server response and removes deleted agents', async () => {
    await service.load();
    fetchSpy.mockImplementationOnce(async () =>
      jsonResponse(makeAgent('a2', { is_locked: true, editable: false })),
    );
    await service.setLifecycle('a2', { is_locked: true });
    const [url, init] = fetchSpy.mock.calls.at(-1)!;
    expect(String(url)).toMatch(/\/agents\/a2\/lifecycle$/);
    expect(JSON.parse(init.body)).toEqual({ is_locked: true });
    expect(service.byId('a2')?.editable).toBe(false);

    fetchSpy.mockImplementationOnce(async () => new Response(null, { status: 204 }));
    await service.remove('a2');
    expect(fetchSpy.mock.calls.at(-1)![1].method).toBe('DELETE');
    expect(service.byId('a2')).toBeNull();
  });

  it('records an error status when loading fails', async () => {
    fetchSpy.mockRejectedValue(new Error('boom'));
    await expect(service.load()).rejects.toThrow();
    expect(service.status()).toBe('error');
    expect(service.error()).toBeTruthy();
  });

  it('summarizes model and evidence display strings for every capability', () => {
    expect(summarizeAgent(full)).toEqual({
      clinicalModel: 'openai / gpt-x',
      extractionModel: 'ollama / qwen3',
      revisionModel: 'ollama / qwen3',
      timelineModel: 'ollama / qwen3',
      evidenceSources: ['livertox', 'dilirank'],
    });
  });
});
