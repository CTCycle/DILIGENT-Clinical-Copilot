// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ClinicalDraftSnapshot, ClinicalDraftWrite, ClinicalFormState } from '../models/types';
import { ClinicalDraftService } from './clinical-draft.service';

const emptySnapshot: ClinicalDraftSnapshot = {
  draft_key: null,
  revision: 0,
  updated_at: null,
  form: null,
};

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const form: ClinicalFormState = {
  patientName: '  Synthetic patient  ',
  visitDate: '2026-10-23',
  patientImageDataUrl: 'data:image/png;base64,not-persisted',
  clinicalInput: '  First line\r\n\r\n second line\t ',
};

function snapshot(revision: number, formValue: ClinicalFormState) {
  return {
    draft_key: 'aa5cae46-66a9-4dbf-a9b5-17d3b58ff104',
    revision,
    updated_at: '2026-10-24T08:30:00Z',
    form: {
      patient_name: formValue.patientName,
      visit_date: formValue.visitDate || null,
      clinical_input: formValue.clinicalInput,
    },
  };
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

describe('ClinicalDraftService', () => {
  let service: ClinicalDraftService;
  let serverSnapshot: ClinicalDraftSnapshot;
  let putRequests: ClinicalDraftWrite[];
  let deleteRequests: string[];

  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    vi.useFakeTimers();
    serverSnapshot = emptySnapshot;
    putRequests = [];
    deleteRequests = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (method === 'GET') {
        return response(serverSnapshot);
      }
      if (method === 'PUT') {
        const payload = JSON.parse(String(init?.body)) as ClinicalDraftWrite;
        putRequests.push(payload);
        if (
          payload.expected_revision !== serverSnapshot.revision ||
          payload.expected_draft_key !== serverSnapshot.draft_key
        ) {
          return response({ detail: 'Saved draft changed.' }, 409);
        }
        const next = {
          draft_key: serverSnapshot.draft_key ?? 'aa5cae46-66a9-4dbf-a9b5-17d3b58ff104',
          revision: serverSnapshot.revision + 1,
          updated_at: '2026-10-24T08:30:00Z',
          form: payload.form,
        };
        serverSnapshot = next;
        return response(next);
      }
      if (method === 'DELETE') {
        deleteRequests.push(url);
        const query = new URL(url, 'http://localhost').searchParams;
        const expectedRevision = Number(query.get('expected_revision'));
        const expectedDraftKey = query.get('expected_draft_key');
        if (
          expectedRevision !== serverSnapshot.revision ||
          expectedDraftKey !== serverSnapshot.draft_key
        ) {
          return response({ detail: 'Saved draft changed.' }, 409);
        }
        const tombstone = {
          draft_key: serverSnapshot.draft_key,
          revision: serverSnapshot.revision + 1,
          updated_at: null,
          form: null,
        };
        serverSnapshot = tombstone;
        return response(tombstone);
      }
      return response({ detail: 'Unexpected request.' }, 500);
    });
    TestBed.configureTestingModule({ providers: [ClinicalDraftService] });
    service = TestBed.inject(ClinicalDraftService);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('offers a saved draft without automatically restoring it', async () => {
    serverSnapshot = snapshot(4, form);

    await service.loadOnOpen();

    expect(service.pendingRecovery()).toBe(true);
    expect(service.autosaveEnabled()).toBe(false);
    expect(service.getRestorableForm()?.clinical_input).toBe(form.clinicalInput);
    expect(putRequests).toHaveLength(0);
  });

  it('opts in on first save and serializes only the approved fields', async () => {
    const storageWrite = vi.spyOn(Storage.prototype, 'setItem');
    await service.loadOnOpen();

    expect(await service.saveAndEnable(form)).toBe(true);

    expect(service.autosaveEnabled()).toBe(true);
    expect(putRequests[0]).toEqual({
      expected_revision: 0,
      expected_draft_key: null,
      form: {
        patient_name: form.patientName,
        visit_date: form.visitDate,
        clinical_input: form.clinicalInput,
      },
    });
    expect(JSON.stringify(putRequests[0])).not.toContain(
      'patientImageDataUrl',
    );
    expect(service.hasUnsavedChanges()).toBe(false);
    expect(storageWrite).not.toHaveBeenCalled();
  });

  it('debounces autosave for 800 ms and saves the latest exact input', async () => {
    await service.loadOnOpen();
    await service.saveAndEnable(form);
    putRequests = [];

    service.scheduleSave({ ...form, clinicalInput: 'first edit' });
    await vi.advanceTimersByTimeAsync(500);
    service.scheduleSave({ ...form, clinicalInput: '  latest\nvalue  ' });
    await vi.advanceTimersByTimeAsync(799);
    expect(putRequests).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1);
    await flushPromises();

    expect(putRequests).toHaveLength(1);
    expect(putRequests[0]?.form.clinical_input).toBe('  latest\nvalue  ');
    expect(putRequests[0]?.expected_revision).toBe(1);
    expect(service.saveStatus()).toBe('saved');
  });

  it('serializes queued saves with the revision returned by the earlier write', async () => {
    await service.loadOnOpen();
    await service.saveAndEnable(form);
    putRequests = [];
    let releaseFirst: ((value: ClinicalDraftSnapshot) => void) | undefined;
    vi.mocked(globalThis.fetch).mockImplementationOnce(
      (_input, init) => new Promise((resolve) => {
        putRequests.push(JSON.parse(String(init?.body)) as ClinicalDraftWrite);
        releaseFirst = (value) => resolve(response(value));
      }),
    );

    service.scheduleSave({ ...form, clinicalInput: 'first queued value' });
    await vi.advanceTimersByTimeAsync(800);
    await flushPromises();
    service.scheduleSave({ ...form, clinicalInput: 'latest queued value' });
    await vi.advanceTimersByTimeAsync(800);
    serverSnapshot = snapshot(2, { ...form, clinicalInput: 'first queued value' });
    releaseFirst?.(serverSnapshot);
    await flushPromises();
    await vi.advanceTimersByTimeAsync(0);
    await flushPromises();

    expect(putRequests).toHaveLength(2);
    expect(putRequests[1]?.expected_revision).toBe(2);
    expect(putRequests[1]?.form.clinical_input).toBe('latest queued value');
  });

  it('keeps unsaved state on failure and pauses autosave on a conflict', async () => {
    await service.loadOnOpen();
    await service.saveAndEnable(form);
    vi.mocked(globalThis.fetch).mockImplementationOnce(
      async () => response({ detail: 'Saved draft changed.' }, 409),
    );

    service.scheduleSave({ ...form, clinicalInput: 'local unsaved value' });
    await vi.advanceTimersByTimeAsync(800);
    await flushPromises();

    expect(service.autosaveEnabled()).toBe(false);
    expect(service.saveStatus()).toBe('conflict');
    expect(service.hasUnsavedChanges()).toBe(true);
    expect(service.errorMessage()).toContain('database switched');
  });

  it('waits for pending writes and cancels the debounce before discard', async () => {
    await service.loadOnOpen();
    await service.saveAndEnable(form);
    service.scheduleSave({ ...form, clinicalInput: 'pending edit' });
    const discardPromise = service.discard();
    await flushPromises();
    await vi.advanceTimersByTimeAsync(1000);
    expect(putRequests).toHaveLength(1);
    expect(deleteRequests).toHaveLength(1);

    expect(await discardPromise).toBe(true);
    expect(service.hasPersistedDraft()).toBe(false);
    expect(service.getRestorableForm()).toBeNull();
  });
});
