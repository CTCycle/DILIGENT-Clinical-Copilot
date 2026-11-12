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
  const section = (title: string): HTMLElement =>
    Array.from(el().querySelectorAll<HTMLElement>('app-capability-section')).find((node) =>
      node.querySelector('h3')?.textContent?.trim() === title,
    )!;
  const lifecycleSwitch = (label: string): HTMLInputElement =>
    el().querySelector<HTMLInputElement>(`.lifecycle input[aria-label="${label}"]`)!;
  const type = async (input: HTMLInputElement, value: string): Promise<void> => {
    input.value = value;
    input.dispatchEvent(new Event('input'));
    await flush();
  };

  beforeEach(() => {
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
      return json({ local_models: [], cloud_providers: [] });
    });
    vi.stubGlobal('fetch', fetchSpy);
  });

  afterEach(() => vi.unstubAllGlobals());

  it('edits an existing agent: profile, version and three capability sections', async () => {
    await open('a2');
    expect(el().querySelector<HTMLInputElement>('#profile-heading + label input')!.value).toBe('Spare');
    expect(el().textContent).toContain('Revision 1');
    expect(['DILI Assessment', 'DILI Revision', 'Timeline'].every((t) => section(t))).toBe(true);
    expect(el().textContent).toContain('Agent validation');
    expect(el().textContent).toContain('Coming soon');
  });

  it('saves edits with expected_revision and shows the new revision', async () => {
    await open('a2');
    expect(button('Save agent').disabled).toBe(true);
    await type(el().querySelector<HTMLInputElement>('#profile-heading + label input')!, 'Renamed');
    expect(el().textContent).toContain('Unsaved changes');
    button('Save agent').click();
    await flush();
    const put = fetchSpy.mock.calls.find(([, init]) => init?.method === 'PUT')!;
    const body = JSON.parse(put[1].body);
    expect(body.expected_revision).toBe(1);
    expect(body.name).toBe('Renamed');
    expect(body.configuration.schema_version).toBe(2);
    expect(el().textContent).toContain('Saved as revision 2');
  });

  it('shows a reload prompt when the server reports a stale revision', async () => {
    putResponse = () => json({ detail: 'stale' }, 409);
    await open('a2');
    await type(el().querySelector<HTMLInputElement>('#profile-heading + label input')!, 'Renamed');
    button('Save agent').click();
    await flush();
    expect(el().textContent).toContain('changed elsewhere');
    expect(button('Reload')).toBeTruthy();
  });

  it('disabling or collapsing a capability keeps its values and does not touch the others', async () => {
    await open('a2');
    const timeline = section('Timeline');
    const focus = timeline.querySelector<HTMLTextAreaElement>('textarea')!;
    focus.value = 'Dates first';
    focus.dispatchEvent(new Event('input'));
    await flush();

    const toggle = timeline.querySelector<HTMLInputElement>('input[role="switch"]')!;
    toggle.checked = false;
    toggle.dispatchEvent(new Event('change'));
    await flush();
    expect(timeline.textContent).toContain('Disabled');
    expect(section('DILI Assessment').textContent).toContain('Enabled');

    // Collapse keeps the body mounted and the edit intact.
    timeline.querySelector<HTMLButtonElement>('.capability-toggle')!.click();
    await flush();
    expect(timeline.querySelector('.capability-body')!.hasAttribute('hidden')).toBe(true);
    expect(timeline.querySelector<HTMLTextAreaElement>('textarea')!.value).toBe('Dates first');
    expect(timeline.querySelector('.capability-toggle')!.getAttribute('aria-expanded')).toBe('false');

    button('Save agent').click();
    await flush();
    const config = JSON.parse(fetchSpy.mock.calls.find(([, init]) => init?.method === 'PUT')![1].body).configuration;
    expect(config.timeline.enabled).toBe(false);
    expect(config.timeline.instructions.timeline_focus).toBe('Dates first');
    expect(config.assessment.enabled).toBe(true);
    expect(config.revision.enabled).toBe(true);
  });

  it('is read-only while locked and lets the lock be lifted from the profile card', async () => {
    agents[1] = makeAgent('a2', { name: 'Spare', is_locked: true, editable: false });
    await open('a2');
    expect(el().textContent).toContain('This agent is locked');
    expect(el().querySelector<HTMLInputElement>('#profile-heading + label input')!.disabled).toBe(true);
    expect(section('Timeline').querySelector<HTMLInputElement>('input[role="switch"]')!.disabled).toBe(true);
    expect(section('Timeline').querySelector('fieldset')!.disabled).toBe(true);
    expect(button('Save agent').disabled).toBe(true);
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
    await type(el().querySelector<HTMLInputElement>('#profile-heading + label input')!, 'Fresh');
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
    await type(el().querySelector<HTMLInputElement>('#profile-heading + label input')!, '');
    expect(el().textContent).toContain('Name is required.');
    expect(button('Create agent').disabled).toBe(true);
  });
});
