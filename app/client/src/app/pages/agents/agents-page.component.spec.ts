// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Agent, AgentKind } from '../../core/models/agent-types';
import { AgentsPageComponent } from './agents-page.component';

const ref = { provider: 'ollama', model: 'qwen3' };

function makeAgent(id: string, kind: AgentKind, overrides: Partial<Agent> = {}): Agent {
  const configuration =
    kind === 'dili'
      ? {
          kind,
          schema_version: 1 as const,
          instructions: { assessment_focus: '', report_style: '', extraction_guidance: '' },
          models: { extraction: ref, clinical: ref, timeline: ref, extraction_overrides: {} },
          reasoning_level: 'off' as const,
          ollama_seed: 42,
          retrieval: {
            evidence_sources: ['livertox' as const],
            use_hybrid_search: true,
            use_reranking: true,
            reranker_model: 'lightweight-balanced-v1',
            retrieval_candidate_count: 40,
            retrieval_selected_count: 6,
            hybrid_vector_weight: 0.65,
            hybrid_text_weight: 0.35,
          },
        }
      : {
          kind,
          schema_version: 1 as const,
          instructions: { review_objectives: '', editing_style: '' },
          models: { planning: ref, stage_overrides: {} },
          reasoning_level: 'off' as const,
          ollama_seed: 42,
          limits: { max_tasks: 8, max_tool_iterations: 24, allowed_tools: null },
        };
  return {
    id,
    kind,
    name: `Agent ${id}`,
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
      configuration_hash: 'abcdef123456',
      configuration,
      created_at: null,
    },
    ...overrides,
  };
}

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('AgentsPageComponent', () => {
  let fixture: ComponentFixture<AgentsPageComponent>;
  const defaultDili = makeAgent('d1', 'dili', { is_default: true });
  const otherDili = makeAgent('d2', 'dili');
  const revision = makeAgent('r1', 'revision', { is_default: true });
  const fetchSpy = vi.fn();
  let putResponse: () => Response;

  const el = (): HTMLElement => fixture.nativeElement;
  const flush = async (): Promise<void> => {
    for (let i = 0; i < 3; i++) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      fixture.detectChanges();
    }
  };
  const click = async (selector: string, text?: string): Promise<void> => {
    const nodes = Array.from(el().querySelectorAll<HTMLElement>(selector));
    const target = text ? nodes.find((n) => n.textContent?.trim() === text) : nodes[0];
    expect(target, `${selector} ${text ?? ''}`).toBeTruthy();
    target!.click();
    await flush();
  };

  beforeEach(async () => {
    putResponse = () => json({ ...otherDili, current_revision: 2 });
    fetchSpy.mockReset().mockImplementation(async (url: string, init?: RequestInit) => {
      if (init?.method === 'PUT') return putResponse();
      if (String(url).includes('/revisions')) return json({ revisions: [otherDili.revision] });
      if (String(url).endsWith('/agents') || String(url).includes('/agents?')) {
        return json({ agents: [defaultDili, otherDili, revision] });
      }
      return json({}, 404);
    });
    vi.stubGlobal('fetch', fetchSpy);
    await TestBed.configureTestingModule({ imports: [AgentsPageComponent] }).compileComponents();
    fixture = TestBed.createComponent(AgentsPageComponent);
    fixture.detectChanges();
    await flush();
  });

  afterEach(() => vi.unstubAllGlobals());

  it('lists agents with kind and default badges', () => {
    const text = el().querySelector('#agent-studio-panel-agents')!.textContent!;
    expect(text).toContain('Agent d1');
    expect(text).toContain('Agent r1');
    expect(text).toContain('Default');
    expect(text).toContain('Revision');
  });

  it('disables archive for the default agent only', () => {
    const archive = (name: string) =>
      el().querySelector<HTMLButtonElement>(`button[aria-label="Archive ${name}"]`)!;
    expect(archive('Agent d1').disabled).toBe(true);
    expect(archive('Agent d2').disabled).toBe(false);
  });

  it('asks for in-page confirmation before archiving', async () => {
    await click('button[aria-label="Archive Agent d2"]');
    expect(el().textContent).toContain('Archive "Agent d2"?');
    expect(fetchSpy.mock.calls.some(([u]) => String(u).includes('/archive'))).toBe(false);
  });

  it('tracks unsaved edits and saves with expected_revision', async () => {
    await click('button.agent-name', 'Agent d2');
    const saveButton = () =>
      Array.from(el().querySelectorAll<HTMLButtonElement>('button')).find(
        (b) => b.textContent?.trim() === 'Save agent',
      )!;
    expect(saveButton().disabled).toBe(true);

    const nameInput = el().querySelector<HTMLInputElement>('#identity-heading + label input')!;
    nameInput.value = 'Renamed';
    nameInput.dispatchEvent(new Event('input'));
    await flush();
    expect(el().textContent).toContain('Unsaved changes');
    expect(saveButton().disabled).toBe(false);

    saveButton().click();
    await flush();
    const put = fetchSpy.mock.calls.find(([, init]) => init?.method === 'PUT')!;
    const body = JSON.parse(put[1].body);
    expect(body.expected_revision).toBe(1);
    expect(body.name).toBe('Renamed');
    expect(el().textContent).toContain('Saved as revision 2');
    expect(el().textContent).not.toContain('Unsaved changes');
  });

  it('surfaces a reload prompt when the server reports a conflict', async () => {
    putResponse = () => json({ detail: 'stale' }, 409);
    await click('button.agent-name', 'Agent d2');
    const nameInput = el().querySelector<HTMLInputElement>('#identity-heading + label input')!;
    nameInput.value = 'Renamed';
    nameInput.dispatchEvent(new Event('input'));
    await flush();
    await click('button.is-primary', 'Save agent');
    expect(el().textContent).toContain('changed elsewhere');
    expect(
      Array.from(el().querySelectorAll('button')).some((b) => b.textContent?.trim() === 'Reload'),
    ).toBe(true);
  });
});
