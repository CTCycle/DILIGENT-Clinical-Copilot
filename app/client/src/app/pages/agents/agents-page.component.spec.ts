// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { makeAgent, makeAgentConfig, withDisabled } from '../../core/testing/agent-fixtures';
import { AgentsPageComponent } from './agents-page.component';

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('AgentsPageComponent', () => {
  let fixture: ComponentFixture<AgentsPageComponent>;
  const base = makeAgent('d1', { is_default: true, name: 'Default agent', description: 'Main agent' });
  const partial = makeAgent(
    'p1',
    { name: 'Assessment only' },
    withDisabled(makeAgentConfig(), 'revision', 'timeline'),
  );
  const used = makeAgent('u1', { name: 'Used', reference_count: 3 });
  const locked = makeAgent('l1', { name: 'Locked', is_locked: true, editable: false });
  const archived = makeAgent('z1', { name: 'Archived one', is_archived: true, editable: false });
  const fetchSpy = vi.fn();
  let lifecycleResponse: () => Response;

  const el = (): HTMLElement => fixture.nativeElement;
  const flush = async (): Promise<void> => {
    for (let i = 0; i < 3; i++) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      fixture.detectChanges();
    }
  };
  const card = (name: string): HTMLElement =>
    Array.from(el().querySelectorAll<HTMLElement>('[data-testid="agent-card"]')).find((node) =>
      node.querySelector('.agent-name')?.textContent?.trim() === name,
    )!;
  const button = (name: string, label: string): HTMLButtonElement =>
    card(name).querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!;

  beforeEach(async () => {
    lifecycleResponse = () => json({ ...partial, is_locked: true, editable: false });
    fetchSpy.mockReset().mockImplementation(async (url: string, init?: RequestInit) => {
      if (init?.method === 'PUT' && String(url).endsWith('/lifecycle')) return lifecycleResponse();
      if (init?.method === 'DELETE') return new Response(null, { status: 204 });
      return json({ agents: [base, partial, used, locked, archived] });
    });
    vi.stubGlobal('fetch', fetchSpy);
    await TestBed.configureTestingModule({
      imports: [AgentsPageComponent],
      providers: [provideRouter([])],
    }).compileComponents();
    fixture = TestBed.createComponent(AgentsPageComponent);
    fixture.detectChanges();
    await flush();
  });

  afterEach(() => vi.unstubAllGlobals());

  it('renders compact cards with status, lock and capability indicators, hiding archived by default', () => {
    const names = Array.from(el().querySelectorAll('.agent-name')).map((n) => n.textContent?.trim());
    expect(names).toEqual(['Default agent', 'Assessment only', 'Locked', 'Used']);
    expect(card('Default agent').textContent).toContain('Default');
    expect(card('Default agent').textContent).toContain('Active');
    expect(card('Default agent').textContent).toContain('r1');
    expect(card('Locked').textContent).toContain('Locked');

    const states = (name: string) =>
      Array.from(card(name).querySelectorAll('.capability')).map((n) => n.classList.contains('is-on'));
    expect(states('Default agent')).toEqual([true, true, true]);
    expect(states('Assessment only')).toEqual([true, false, false]);
  });

  it('shows archived agents when requested', async () => {
    const toggle = el().querySelector<HTMLInputElement>('.toolbar input[type="checkbox"]')!;
    toggle.checked = true;
    toggle.dispatchEvent(new Event('change'));
    await flush();
    expect(card('Archived one').textContent).toContain('Archived');
    expect(button('Archived one', 'Restore Archived one').disabled).toBe(false);
  });

  it('explains every disabled action through the tooltip', () => {
    const tip = (name: string, label: string) =>
      button(name, label).closest('.icon-button-host')!.getAttribute('data-tooltip');
    expect(button('Default agent', 'Archive Default agent').disabled).toBe(true);
    expect(tip('Default agent', 'Archive Default agent')).toContain('another agent the default');
    expect(tip('Default agent', 'Delete Default agent')).toBe('The default agent cannot be deleted');
    expect(tip('Default agent', 'Set Default agent as default')).toBe('Already the default agent');
    expect(tip('Used', 'Delete Used')).toContain('Used by 3 saved runs');
    expect(tip('Locked', 'Delete Locked')).toBe('Unlock this agent before deleting it');
    // Enabled actions show their name.
    expect(button('Assessment only', 'Delete Assessment only').disabled).toBe(false);
    expect(tip('Assessment only', 'Delete Assessment only')).toBe('Delete Assessment only');
  });

  it('marks locked agents as read-only on the edit action', () => {
    expect(button('Locked', 'View Locked (read-only)')).toBeTruthy();
    expect(button('Assessment only', 'Edit Assessment only')).toBeTruthy();
  });

  it('locks an agent through the lifecycle endpoint without creating a revision', async () => {
    button('Assessment only', 'Lock Assessment only').click();
    await flush();
    const put = fetchSpy.mock.calls.find(([url, init]) => init?.method === 'PUT' && String(url).endsWith('/lifecycle'))!;
    expect(JSON.parse(put[1].body)).toEqual({ is_locked: true });
    expect(el().textContent).toContain('Locked "Assessment only".');
  });

  it('requires confirmation before permanently deleting an agent', async () => {
    button('Assessment only', 'Delete Assessment only').click();
    await flush();
    expect(fetchSpy.mock.calls.some(([, init]) => init?.method === 'DELETE')).toBe(false);
    const confirm = Array.from(document.querySelectorAll<HTMLButtonElement>('app-confirm-dialog button'))
      .find((b) => b.textContent?.trim() === 'Delete agent')!;
    confirm.click();
    await flush();
    expect(fetchSpy.mock.calls.some(([, init]) => init?.method === 'DELETE')).toBe(true);
    expect(el().textContent).toContain('Deleted "Assessment only".');
    expect(el().querySelector('[data-testid="agent-card"] .agent-name')?.textContent).toContain('Default agent');
    expect(Array.from(el().querySelectorAll('.agent-name')).some((n) => n.textContent?.trim() === 'Assessment only')).toBe(false);
  });

  it('filters agents by name', async () => {
    const search = el().querySelector<HTMLInputElement>('input[type="search"]')!;
    search.value = 'assess';
    search.dispatchEvent(new Event('input'));
    await flush();
    expect(Array.from(el().querySelectorAll('.agent-name')).map((n) => n.textContent?.trim())).toEqual([
      'Assessment only',
    ]);
  });
});
