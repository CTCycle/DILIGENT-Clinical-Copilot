// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ClinicalSessionInput } from '../models/types';
import { AppStateService } from '../state/app-state.service';
import { FetchCall, jsonResponse, sessionState } from '../testing/clinical-session-fixtures';
import { ClinicalSessionDraftService, DraftNotEditableError } from './clinical-session-draft.service';
import { ClinicalSessionPoller } from './clinical-session-poller';

describe('ClinicalSessionDraftService', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let appState: AppStateService;
  let drafts: ClinicalSessionDraftService;

  const calls = () => (fetchSpy.mock.calls as FetchCall[]).map(([url, init]) => `${(init as RequestInit)?.method} ${String(url)}`);

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, 'fetch');
    appState = TestBed.inject(AppStateService);
    drafts = TestBed.inject(ClinicalSessionDraftService);
    appState.updateDiliAgent({
      form: { patientName: 'Ada', visitDate: '2026-10-01', patientImageDataUrl: null, clinicalInput: 'ALT 300' },
      isDirty: true,
    });
  });

  afterEach(() => vi.restoreAllMocks());

  it('creates the draft on first save and updates the same session afterwards', async () => {
    fetchSpy
      .mockResolvedValueOnce(jsonResponse(sessionState({ updated_at: '2026-10-01T08:01:00Z' }), 201))
      .mockResolvedValueOnce(jsonResponse(sessionState({ updated_at: '2026-10-01T08:02:00Z' })));

    expect(await drafts.save()).toBe(5);
    expect(appState.state().diliAgent).toMatchObject({
      draftSessionId: 5,
      draftUpdatedAt: '2026-10-01T08:01:00Z',
      saveStatus: 'saved',
      isDirty: false,
    });

    appState.updateDiliAgent({ form: { ...appState.state().diliAgent.form, clinicalInput: 'ALT 400' }, isDirty: true });
    expect(await drafts.save()).toBe(5);

    expect(calls()).toEqual(['POST /api/clinical/sessions', 'PUT /api/clinical/sessions/5/input']);
    const update = JSON.parse(String((fetchSpy.mock.calls[1][1] as RequestInit).body));
    expect(update).toMatchObject({ clinical_input: 'ALT 400', expected_updated_at: '2026-10-01T08:01:00Z' });
  });

  it('reports a conflict instead of overwriting when the draft is no longer editable', async () => {
    appState.updateDiliAgent({ draftSessionId: 5, draftUpdatedAt: '2026-10-01T08:00:00Z' });
    fetchSpy.mockResolvedValueOnce(jsonResponse({ detail: 'Session is no longer a draft.' }, 409));

    expect(await drafts.save()).toBeNull();
    expect(appState.state().diliAgent.saveStatus).toBe('conflict');
    expect(appState.state().diliAgent.isDirty).toBe(true);
  });

  it('keeps the form dirty when it changed while the save was in flight', async () => {
    let release!: (response: Response) => void;
    fetchSpy.mockReturnValueOnce(new Promise<Response>((resolve) => { release = resolve; }));

    const pending = drafts.save();
    await vi.waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    appState.updateDiliAgent({ form: { ...appState.state().diliAgent.form, clinicalInput: 'ALT 999' } });
    release(jsonResponse(sessionState(), 201));
    await pending;

    expect(appState.state().diliAgent.isDirty).toBe(true);
  });

  it('loads a stored draft into the form and refuses sessions that already launched', async () => {
    const input: ClinicalSessionInput = {
      session_id: 5,
      status: 'draft',
      patient_name: 'Grace',
      visit_date: null,
      clinical_input: 'Stored text',
      patient_image_base64: null,
      updated_at: '2026-10-02T09:00:00Z',
    };
    fetchSpy
      .mockResolvedValueOnce(jsonResponse(input))
      .mockResolvedValueOnce(jsonResponse({ ...input, status: 'running' }));

    await drafts.load(5);
    expect(appState.state().diliAgent).toMatchObject({
      draftSessionId: 5,
      draftUpdatedAt: '2026-10-02T09:00:00Z',
      isDirty: false,
      form: { patientName: 'Grace', visitDate: '', clinicalInput: 'Stored text' },
    });

    await expect(drafts.load(5)).rejects.toBeInstanceOf(DraftNotEditableError);
  });

  it('resets the form and draft linkage', () => {
    appState.updateDiliAgent({ draftSessionId: 5, draftUpdatedAt: 'x', saveStatus: 'saved' });

    drafts.reset();

    expect(appState.state().diliAgent).toMatchObject({
      draftSessionId: null,
      saveStatus: 'idle',
      isDirty: false,
      form: { patientName: '', clinicalInput: '' },
    });
  });
});

describe('ClinicalSessionPoller', () => {
  afterEach(() => vi.restoreAllMocks());

  it('ignores older snapshots and stops once the session is no longer active', async () => {
    const snapshots = [
      sessionState({ status: 'running', version: 2 }),
      sessionState({ status: 'queued', version: 1 }),
      sessionState({ status: 'completed', version: 3, has_report: true }),
      sessionState({ status: 'completed', version: 4 }),
    ];
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(
      async () => jsonResponse(snapshots[Math.min(fetchSpy.mock.calls.length - 1, snapshots.length - 1)]),
    );
    const seen: number[] = [];

    await new Promise<void>((resolve) => {
      TestBed.inject(ClinicalSessionPoller).watch(5, (state) => {
        seen.push(state.version);
        if (state.status === 'completed') resolve();
      });
    });
    await new Promise((resolve) => setTimeout(resolve, 600));

    expect(seen).toEqual([2, 3]);
    expect(fetchSpy).toHaveBeenCalledTimes(3);
    expect(String(fetchSpy.mock.calls[0][0])).toBe('/api/clinical/sessions/5/status');
  });

  it('reports a missing session and stops when cancelled', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ detail: 'not found' }, 404));
    const onError = vi.fn();

    TestBed.inject(ClinicalSessionPoller).watch(9, () => undefined, onError);
    await vi.waitFor(() => expect(onError).toHaveBeenCalledTimes(1));

    const stop = TestBed.inject(ClinicalSessionPoller).watch(5, () => undefined);
    stop();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(fetchSpy.mock.calls.filter(([url]) => String(url).includes('/sessions/9/'))).toHaveLength(1);
  });
});
