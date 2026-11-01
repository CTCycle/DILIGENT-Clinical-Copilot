// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable, inject } from '@angular/core';

import { DEFAULT_FORM_STATE } from '../constants';
import { AppStateService, DEFAULT_DILI_AGENT_STATE } from '../state/app-state.service';
import { buildSessionInputPayload } from '../utils';
import {
  createClinicalSession,
  fetchClinicalSessionInput,
  updateClinicalSessionInput,
} from './clinical-session-lifecycle-api';
import { ApiHttpError } from './http-api';

export class DraftNotEditableError extends Error {
  constructor() {
    super('This session is no longer a draft and cannot be edited.');
    this.name = 'DraftNotEditableError';
  }
}

/** Persists the DILI form as a draft session: the first save creates it, later saves update it. */
@Injectable({ providedIn: 'root' })
export class ClinicalSessionDraftService {
  private readonly stateService = inject(AppStateService);
  private saveQueue: Promise<unknown> = Promise.resolve();

  private get vm() {
    return this.stateService.state().diliAgent;
  }

  /** Saves the current form; resolves to the draft session id, or null when saving failed. */
  save(): Promise<number | null> {
    const run = this.saveQueue.then(() => this.performSave(), () => this.performSave());
    this.saveQueue = run;
    return run;
  }

  /** Loads a stored draft into the form. Throws `DraftNotEditableError` for non-drafts. */
  async load(sessionId: number): Promise<void> {
    const input = await fetchClinicalSessionInput(sessionId);
    if (input.status !== 'draft') {
      throw new DraftNotEditableError();
    }
    this.stateService.updateDiliAgent({
      form: {
        patientName: input.patient_name ?? '',
        visitDate: input.visit_date ?? '',
        patientImageDataUrl: input.patient_image_base64
          ? `data:image/png;base64,${input.patient_image_base64}`
          : null,
        clinicalInput: input.clinical_input,
      },
      draftSessionId: input.session_id,
      draftUpdatedAt: input.updated_at,
      saveStatus: 'saved',
      lastSavedAt: input.updated_at,
      isDirty: false,
    });
  }

  /** Clears the form and forgets the draft (the stored session is untouched). */
  reset(): void {
    this.stateService.updateDiliAgent({
      ...DEFAULT_DILI_AGENT_STATE,
      form: { ...DEFAULT_FORM_STATE },
      isPulling: this.vm.isPulling,
    });
  }

  private async performSave(): Promise<number | null> {
    const { form, draftSessionId, draftUpdatedAt } = this.vm;
    const payload = buildSessionInputPayload(form);
    this.stateService.updateDiliAgent({ saveStatus: 'saving' });
    try {
      const state = draftSessionId !== null && draftUpdatedAt !== null
        ? await updateClinicalSessionInput(draftSessionId, payload, draftUpdatedAt)
        : await createClinicalSession(payload);
      // Edits made while the request was in flight keep the form dirty.
      const unchanged = this.vm.form === form;
      this.stateService.updateDiliAgent({
        draftSessionId: state.session_id,
        draftUpdatedAt: state.updated_at,
        saveStatus: 'saved',
        lastSavedAt: state.updated_at,
        isDirty: !unchanged,
      });
      return state.session_id;
    } catch (error) {
      const conflict = error instanceof ApiHttpError && error.statusCode === 409;
      this.stateService.updateDiliAgent({ saveStatus: conflict ? 'conflict' : 'error' });
      return null;
    }
  }
}
