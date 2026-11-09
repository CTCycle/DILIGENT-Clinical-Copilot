// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, Router, convertToParamMap, provideRouter } from '@angular/router';
import { BehaviorSubject } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { FetchCall, jsonResponse, sessionState } from '../../core/testing/clinical-session-fixtures';
import { DiliAgentPageComponent } from './dili-agent-page.component';

describe('DiliAgentPageComponent', () => {
  let fixture: ComponentFixture<DiliAgentPageComponent>;
  let component: DiliAgentPageComponent;
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let draftInput: Record<string, unknown>;
  const query = new BehaviorSubject(convertToParamMap({}));

  const root = () => fixture.nativeElement as HTMLElement;
  const text = () => root().textContent ?? '';
  const sessionCalls = () =>
    (fetchSpy.mock.calls as FetchCall[])
      .filter(([url]) => String(url).includes('/clinical/sessions'))
      .map(([url, init]) => `${(init as RequestInit | undefined)?.method ?? 'GET'} ${String(url)}`);

  async function settle(): Promise<void> {
    for (let i = 0; i < 4; i += 1) {
      fixture.detectChanges();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  async function create(params: Record<string, string> = {}): Promise<void> {
    query.next(convertToParamMap(params));
    fixture = TestBed.createComponent(DiliAgentPageComponent);
    component = fixture.componentInstance;
    await settle();
  }

  beforeEach(async () => {
    draftInput = {
      session_id: 5,
      status: 'draft',
      patient_name: 'Stored draft patient',
      visit_date: '2026-10-05',
      clinical_input: 'Stored draft text',
      patient_image_base64: null,
      updated_at: '2026-10-05T09:00:00Z',
    };
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith('/clinical/sessions') && init?.method === 'POST') {
        return jsonResponse(sessionState({ updated_at: '2026-10-05T10:00:00Z' }), 201);
      }
      if (url.endsWith('/clinical/sessions/5/input') && init?.method === 'PUT') {
        return jsonResponse(sessionState({ updated_at: '2026-10-05T10:05:00Z' }));
      }
      if (url.endsWith('/clinical/sessions/5/input')) return jsonResponse(draftInput);
      if (url.endsWith('/clinical/section-template')) return jsonResponse({ headings: {}, template: 'Template' });
      return jsonResponse({}, 500);
    });
    await TestBed.configureTestingModule({
      imports: [DiliAgentPageComponent],
      providers: [
        provideRouter([]),
        {
          provide: ActivatedRoute,
          useValue: {
            queryParamMap: query,
            get snapshot() {
              return { queryParamMap: query.value };
            },
          },
        },
      ],
    }).compileComponents();
  });

  afterEach(() => {
    fixture?.destroy();
    vi.restoreAllMocks();
  });

  it('is a plain creation page: no progress, stop, report link, or recovery banner', async () => {
    await create();

    expect(text()).not.toMatch(/Stop analysis|View report|Saved draft available/);
    expect(root().querySelector<HTMLButtonElement>('#run-analysis-button')?.disabled).toBe(false);
    expect(root().querySelector('#save-draft-button')?.textContent).toContain('Save as draft');
    expect(root().querySelector('app-analysis-wizard')).not.toBeNull();
  });

  it('opens the setup wizard from the run button', async () => {
    await create();

    root().querySelector<HTMLButtonElement>('#run-analysis-button')!.click();

    expect(component.wizardOpen()).toBe(true);
  });

  it('creates the draft on first save, then updates the same session', async () => {
    await create();
    component.handleFormChange('clinicalInput', 'ALT 300');
    await settle();
    expect(text()).toContain('Not saved yet');

    root().querySelector<HTMLButtonElement>('#save-draft-button')!.click();
    await settle();
    expect(component.vm.draftSessionId).toBe(5);
    expect(text()).toMatch(/Draft #5 · Saved/);

    component.handleFormChange('clinicalInput', 'ALT 400');
    await settle();
    expect(text()).toContain('Unsaved changes');
    root().querySelector<HTMLButtonElement>('#save-draft-button')!.click();
    await settle();

    expect(sessionCalls()).toEqual([
      'POST /api/clinical/sessions',
      'PUT /api/clinical/sessions/5/input',
    ]);
  });

  it('keeps the input and offers a reload when the draft has changed elsewhere', async () => {
    await create();
    component.stateService.updateDiliAgent({ draftSessionId: 5, draftUpdatedAt: '2026-10-01T00:00:00Z' });
    fetchSpy.mockImplementationOnce(async () => jsonResponse({ detail: 'stale' }, 409));
    component.handleFormChange('clinicalInput', 'Local edit');

    await component.saveDraft();
    await settle();

    expect(text()).toContain('Reload it before saving');
    expect(component.vm.form.clinicalInput).toBe('Local edit');
    expect(text()).toContain('Reload draft');
  });

  it('restores a draft from ?session= and shows its saved state', async () => {
    await create({ session: '5' });

    expect(component.vm.draftSessionId).toBe(5);
    expect(component.vm.form).toMatchObject({
      patientName: 'Stored draft patient',
      visitDate: '2026-10-05',
      clinicalInput: 'Stored draft text',
    });
    expect(text()).toMatch(/Draft #5 · Saved/);
  });

  it('sends a session that already launched to its report instead of the editor', async () => {
    draftInput['status'] = 'running';
    const navigate = vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);

    await create({ session: '5' });

    expect(component.vm.draftSessionId).toBeNull();
    expect(navigate).toHaveBeenCalledWith(['/report', 5], { replaceUrl: true });
  });

  it('clears the form and forgets the draft without deleting it', async () => {
    await create({ session: '5' });

    component.clearAll();
    await settle();

    expect(component.vm.form.clinicalInput).toBe('');
    expect(component.vm.draftSessionId).toBeNull();
    expect(sessionCalls().some((call: string) => call.startsWith('DELETE'))).toBe(false);
  });

  it('keeps unsaved input unless clearing is confirmed', async () => {
    await create({ session: '5' });
    component.handleFormChange('clinicalInput', 'Unsaved edit');
    const confirm = vi.spyOn(globalThis, 'confirm').mockReturnValue(false);

    component.clearAll();
    expect(component.vm.form.clinicalInput).toBe('Unsaved edit');

    confirm.mockReturnValue(true);
    component.clearAll();
    expect(component.vm.form.clinicalInput).toBe('');
  });
});
