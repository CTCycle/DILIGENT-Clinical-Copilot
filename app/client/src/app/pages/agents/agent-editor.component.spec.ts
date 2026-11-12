// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Agent } from '../../core/models/agent-types';
import { makeAgent } from '../../core/testing/agent-fixtures';
import { AgentEditorComponent } from './agent-editor.component';

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('AgentEditorComponent', () => {
  let fixture: ComponentFixture<AgentEditorComponent>;
  let agents: Agent[];
  let putResponse: () => Response;
  let modelConfig: unknown;
  const fetchSpy = vi.fn();

  const el = (): HTMLElement => fixture.nativeElement;
  const flush = async (): Promise<void> => {
    for (let i = 0; i < 4; i++) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      fixture.detectChanges();
    }
  };
  const open = async (id: string): Promise<void> => {
    await TestBed.configureTestingModule({
      imports: [AgentEditorComponent],
      providers: [
        provideRouter([]),
        { provide: ActivatedRoute, useValue: { paramMap: of(convertToParamMap({ id })) } },
      ],
    }).compileComponents();
    fixture = TestBed.createComponent(AgentEditorComponent);
    fixture.detectChanges();
    await flush();
  };
  const button = (text: string): HTMLButtonElement =>
    Array.from(el().querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === text,
    )!;
  const tab = (label: string): HTMLButtonElement =>
    Array.from(el().querySelectorAll<HTMLButtonElement>('[role="tab"]')).find((t) =>
      t.textContent?.replace('Off', '').trim() === label,
    )!;
  const goTo = async (label: string): Promise<void> => {
    tab(label).click();
    await flush();
  };
  const nameInput = (): HTMLInputElement => el().querySelector<HTMLInputElement>('.overview input[type="text"]')!;
  const capabilitySwitch = (): HTMLInputElement => el().querySelector<HTMLInputElement>('.capability-head input[role="switch"]')!;
  const lifecycleSwitch = (label: string): HTMLInputElement =>
    el().querySelector<HTMLInputElement>(`.lifecycle input[aria-label="${label}"]`)!;
  const type = async (input: HTMLInputElement, value: string): Promise<void> => {
    input.value = value;
    input.dispatchEvent(new Event('input'));
    await flush();
  };

  beforeEach(() => {
    modelConfig = { local_models: [], cloud_providers: [] };
    agents = [makeAgent('a1', { is_default: true, name: 'Main', current_revision: 3 }), makeAgent('a2', { name: 'Spare' })];
    putResponse = () => json({ ...agents[1], current_revision: 2, name: 'Renamed' });
    fetchSpy.mockReset().mockImplementation(async (url: string, init?: RequestInit) => {
      const target = String(url);
      if (init?.method === 'PUT' && target.endsWith('/lifecycle')) {
        const body = JSON.parse(String(init.body));
        return json({ ...agents[1], ...body, editable: !(body.is_locked ?? false) });
      }
      if (init?.method === 'PUT') return putResponse();
      if (init?.method === 'POST' && target.endsWith('/agents')) {
        return json(makeAgent('new1', { name: JSON.parse(String(init.body)).name }), 201);
      }
      if (target.includes('/revisions')) return json({ revisions: [agents[1].revision] });
      if (target.includes('/agents')) return json({ agents });
      return json(modelConfig);
    });
    vi.stubGlobal('fetch', fetchSpy);
  });

  afterEach(() => vi.unstubAllGlobals());

  it('edits an existing agent through four tabs with no validation UI', async () => {
    await open('a2');
    expect(Array.from(el().querySelectorAll('[role="tab"]')).map((t) => t.textContent?.trim())).toEqual([
      'Overview',
      'DILI Assessment',
      'DILI Report Review',
      'Timeline',
    ]);
    expect(nameInput().value).toBe('Spare');
    expect(el().textContent).toContain('Revision 1');
    expect(el().textContent).not.toContain('Agent validation');
    expect(el().textContent).not.toContain('Check readiness');
    expect(el().querySelector('app-assessment-sections')).toBeNull();
    await goTo('DILI Assessment');
    expect(el().querySelector('app-assessment-sections')).not.toBeNull();
    await goTo('DILI Report Review');
    expect(el().querySelector('app-revision-sections')).not.toBeNull();
    await goTo('Timeline');
    expect(el().querySelector('app-timeline-sections')).not.toBeNull();
  });

  it('saves edits with expected_revision and shows the new revision', async () => {
    await open('a2');
    expect(button('Save').disabled).toBe(true);
    await type(nameInput(), 'Renamed');
    expect(el().textContent).toContain('Unsaved changes');
    button('Save').click();
    await flush();
    const put = fetchSpy.mock.calls.find(([, init]) => init?.method === 'PUT')!;
    const body = JSON.parse(put[1].body);
    expect(body.expected_revision).toBe(1);
    expect(body.name).toBe('Renamed');
    expect(body.configuration.schema_version).toBe(3);
    expect(el().textContent).toContain('Saved as revision 2');
  });

  it('shows a reload prompt when the server reports a stale revision', async () => {
    putResponse = () => json({ detail: 'stale' }, 409);
    await open('a2');
    await type(nameInput(), 'Renamed');
    button('Save').click();
    await flush();
    expect(el().textContent).toContain('changed elsewhere');
    expect(button('Reload')).toBeTruthy();
  });

  it('keeps edits across tabs and saves every section together', async () => {
    await open('a2');
    await type(nameInput(), 'Renamed');
    await goTo('Timeline');
    const focus = el().querySelector<HTMLTextAreaElement>('app-timeline-sections textarea')!;
    focus.value = 'Dates first';
    focus.dispatchEvent(new Event('input'));
    await flush();

    capabilitySwitch().checked = false;
    capabilitySwitch().dispatchEvent(new Event('change'));
    await flush();
    expect(el().querySelector('.capability-state')!.textContent).toContain('Disabled');
    expect(tab('Timeline').textContent).toContain('Off');

    await goTo('DILI Assessment');
    expect(el().querySelector('.capability-state')!.textContent).toContain('Enabled');
    await goTo('Overview');
    expect(nameInput().value).toBe('Renamed');
    await goTo('Timeline');
    expect(el().querySelector<HTMLTextAreaElement>('app-timeline-sections textarea')!.value).toBe('Dates first');

    button('Save').click();
    await flush();
    const body = JSON.parse(fetchSpy.mock.calls.find(([, init]) => init?.method === 'PUT')![1].body);
    expect(body.name).toBe('Renamed');
    expect(body.configuration.timeline.enabled).toBe(false);
    expect(body.configuration.timeline.instructions.timeline_focus).toBe('Dates first');
    expect(body.configuration.assessment.enabled).toBe(true);
    expect(body.configuration.revision.enabled).toBe(true);
  });

  it('is read-only while locked and lets the lock be lifted from the Overview tab', async () => {
    agents[1] = makeAgent('a2', { name: 'Spare', is_locked: true, editable: false });
    await open('a2');
    expect(el().textContent).toContain('This agent is locked');
    expect(nameInput().disabled).toBe(true);
    expect(el().querySelector('.editor-context')!.textContent).toContain('Read-only · locked');
    await goTo('Timeline');
    expect(capabilitySwitch().disabled).toBe(true);
    expect(el().querySelector<HTMLFieldSetElement>('app-timeline-sections fieldset')!.disabled).toBe(true);
    await goTo('Overview');
    expect(button('Save').disabled).toBe(true);
    // The unlock control itself is never blocked by the lock.
    expect(lifecycleSwitch('Locked').disabled).toBe(false);

    const unlock = lifecycleSwitch('Locked');
    unlock.checked = false;
    unlock.dispatchEvent(new Event('change'));
    await flush();
    const put = fetchSpy.mock.calls.find(([url]) => String(url).endsWith('/lifecycle'))!;
    expect(JSON.parse(put[1].body)).toEqual({ is_locked: false });
    expect(el().textContent).not.toContain('This agent is locked');
  });

  it('keeps reactivate and restore reachable for inactive and archived agents, but protects the default', async () => {
    agents[1] = makeAgent('a2', { is_active: false, is_archived: true, editable: false });
    await open('a2');
    expect(el().textContent).toContain('This agent is archived');
    expect(lifecycleSwitch('Active').disabled).toBe(false);
    expect(lifecycleSwitch('Archived').disabled).toBe(false);
  });

  it('prevents the default agent from being deactivated or archived', async () => {
    agents[1] = makeAgent('a2', { is_default: true });
    await open('a2');
    expect(lifecycleSwitch('Active').disabled).toBe(true);
    expect(lifecycleSwitch('Archived').disabled).toBe(true);
  });

  it('creates a new active, unlocked agent from the default configuration', async () => {
    await open('new');
    expect(el().querySelector('h1')!.textContent).toContain('New agent');
    expect(el().querySelector('.lifecycle')).toBeNull();
    expect(button('Create agent').disabled).toBe(false);
    await type(nameInput(), 'Fresh');
    button('Create agent').click();
    await flush();
    const post = fetchSpy.mock.calls.find(([, init]) => init?.method === 'POST')!;
    const body = JSON.parse(post[1].body);
    expect(body.name).toBe('Fresh');
    expect(body.configuration.assessment.enabled).toBe(true);
    expect(body.configuration.revision.enabled).toBe(true);
    expect(body.configuration.timeline.enabled).toBe(true);
  });

  it('blocks create with validation errors from the shared form rules', async () => {
    await open('new');
    await type(nameInput(), '');
    expect(el().textContent).toContain('Name is required.');
    expect(button('Create agent').disabled).toBe(true);
  });

  it('offers a single custom-instructions field and a report template section', async () => {
    await open('a2');
    await goTo('DILI Assessment');
    const section = el().querySelector('app-assessment-sections')!;
    expect(section.querySelectorAll('textarea').length).toBe(1);
    expect(section.textContent).toContain('Custom instructions');
    expect(section.textContent).not.toContain('Extraction guidance');
    expect(section.textContent).toContain('Using the default report format.');
  });

  it('uploads, previews and removes a report template and saves it with the agent', async () => {
    await open('a2');
    await goTo('DILI Assessment');
    const picker = el().querySelector<HTMLInputElement>('app-report-template-field input[type="file"]')!;
    const file = new File(['# Report\n## Findings'], 'layout.md', { type: 'text/markdown' });
    Object.defineProperty(picker, 'files', { value: [file], configurable: true });
    picker.dispatchEvent(new Event('change'));
    await flush();
    const field = el().querySelector('app-report-template-field')!;
    expect(field.textContent).toContain('layout.md');
    expect(button('Preview')).toBeTruthy();
    button('Preview').click();
    await flush();
    expect(field.querySelector('pre')!.textContent).toContain('## Findings');
    button('Remove').click();
    await flush();
    expect(field.textContent).toContain('Using the default report format.');
    picker.dispatchEvent(new Event('change'));
    await flush();
    button('Save').click();
    await flush();
    const put = fetchSpy.mock.calls.find(([, init]) => init?.method === 'PUT')!;
    expect(JSON.parse(put[1].body).configuration.assessment.report_template).toEqual({
      file_name: 'layout.md',
      format: 'md',
      content: '# Report\n## Findings',
    });
  });

  it('rejects unsupported template files', async () => {
    await open('a2');
    await goTo('DILI Assessment');
    const picker = el().querySelector<HTMLInputElement>('app-report-template-field input[type="file"]')!;
    Object.defineProperty(picker, 'files', { value: [new File(['x'], 'layout.pdf')], configurable: true });
    picker.dispatchEvent(new Event('change'));
    await flush();
    expect(el().querySelector('app-report-template-field')!.textContent).toContain('Unsupported file type');
  });

  it('populates model dropdowns from the selected provider and installed Ollama models', async () => {
    modelConfig = {
      local_models: [
        { name: 'qwen3', available_in_ollama: true },
        { name: 'not-installed', available_in_ollama: false },
      ],
      cloud_providers: [{ id: 'openai', display_name: 'OpenAI', models: [{ id: 'gpt-x' }, { id: 'gpt-y' }] }],
    };
    await open('a2');
    await goTo('DILI Assessment');
    const row = el().querySelector('app-agent-model-field .model-row')!;
    const [provider, model] = Array.from(row.querySelectorAll<HTMLSelectElement>('select'));
    const options = (select: HTMLSelectElement) => Array.from(select.options).map((o) => o.textContent?.trim());
    expect(options(model)).toEqual(['qwen3']);
    provider.value = 'openai';
    provider.dispatchEvent(new Event('change'));
    await flush();
    const updated = el().querySelector('app-agent-model-field .model-row')!.querySelectorAll<HTMLSelectElement>('select')[1];
    expect(options(updated)).toEqual(['gpt-x', 'gpt-y']);
    expect(updated.value).toBe('gpt-x');
  });
});
