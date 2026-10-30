// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable, signal } from '@angular/core';

import {
  ClinicalDraftForm,
  ClinicalDraftSnapshot,
  ClinicalDraftWrite,
  ClinicalFormState,
} from '../models/types';
import {
  discardClinicalDraft,
  fetchClinicalDraft,
  saveClinicalDraft,
} from './clinical-draft-api';
import { ApiHttpError } from './http-api';

export type ClinicalDraftSaveStatus = 'idle' | 'saving' | 'saved' | 'error' | 'conflict';

function formForDraft(form: ClinicalFormState): ClinicalDraftForm {
  return {
    patient_name: form.patientName,
    visit_date: form.visitDate || null,
    clinical_input: form.clinicalInput,
  };
}

function copyDraftForm(form: ClinicalDraftForm): ClinicalDraftForm {
  return { ...form };
}

function serializeDraftForm(form: ClinicalDraftForm): string {
  return JSON.stringify(form);
}

@Injectable({ providedIn: 'root' })
export class ClinicalDraftService {
  readonly loading = signal(false);
  readonly pendingRecovery = signal(false);
  readonly autosaveEnabled = signal(false);
  readonly saveStatus = signal<ClinicalDraftSaveStatus>('idle');
  readonly lastSavedAt = signal<string | null>(null);
  readonly hasPersistedDraft = signal(false);
  readonly hasUnsavedChanges = signal(false);
  readonly errorMessage = signal<string | null>(null);

  private serverRevision = 0;
  private serverDraftKey: string | null = null;
  private restorableForm: ClinicalDraftForm | null = null;
  private latestScheduledForm: ClinicalDraftForm | null = null;
  private debounceTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private writeQueue: Promise<void> = Promise.resolve();
  private generation = 0;
  private loaded = false;
  private loadTask: Promise<boolean> | null = null;
  private discarding = false;

  loadOnOpen(): Promise<boolean> {
    if (this.loaded) {
      return Promise.resolve(true);
    }
    if (this.loadTask) {
      return this.loadTask;
    }

    const generation = ++this.generation;
    this.loading.set(true);
    this.errorMessage.set(null);
    const task = fetchClinicalDraft()
      .then((snapshot) => {
        if (generation !== this.generation) {
          return false;
        }
        this.acceptSnapshot(snapshot);
        this.pendingRecovery.set(snapshot.form !== null);
        this.saveStatus.set(snapshot.form ? 'saved' : 'idle');
        this.loaded = true;
        return true;
      })
      .catch(() => {
        if (generation === this.generation) {
          this.saveStatus.set('error');
          this.errorMessage.set(
            'Draft storage is unavailable. Your current input remains on this screen.',
          );
        }
        return false;
      })
      .finally(() => {
        if (generation === this.generation) {
          this.loading.set(false);
        }
        this.loadTask = null;
      });
    this.loadTask = task;
    return task;
  }

  getRestorableForm(): ClinicalDraftForm | null {
    return this.restorableForm ? copyDraftForm(this.restorableForm) : null;
  }

  resume(): void {
    const form = this.getRestorableForm();
    if (!form) {
      return;
    }
    this.pendingRecovery.set(false);
    this.autosaveEnabled.set(true);
    this.hasPersistedDraft.set(true);
    this.hasUnsavedChanges.set(false);
    this.latestScheduledForm = form;
    this.saveStatus.set('saved');
    this.errorMessage.set(null);
  }

  async saveAndEnable(form: ClinicalFormState): Promise<boolean> {
    if (!(await this.loadOnOpen()) || this.pendingRecovery() || this.discarding) {
      return false;
    }
    const draftForm = formForDraft(form);
    this.latestScheduledForm = draftForm;
    this.hasUnsavedChanges.set(true);
    this.errorMessage.set(null);
    this.saveStatus.set('saving');
    const succeeded = await this.enqueueWrite(() => this.performSave(draftForm));
    if (succeeded) {
      this.pendingRecovery.set(false);
      this.autosaveEnabled.set(true);
      this.hasPersistedDraft.set(true);
    }
    return succeeded;
  }

  scheduleSave(form: ClinicalFormState): void {
    if (!this.autosaveEnabled() || this.discarding) {
      return;
    }
    const draftForm = formForDraft(form);
    if (
      this.latestScheduledForm &&
      serializeDraftForm(this.latestScheduledForm) === serializeDraftForm(draftForm) &&
      !this.hasUnsavedChanges()
    ) {
      return;
    }
    this.latestScheduledForm = draftForm;
    this.hasUnsavedChanges.set(true);
    this.errorMessage.set(null);
    this.cancelScheduledSave();
    this.debounceTimer = globalThis.setTimeout(() => {
      this.debounceTimer = null;
      if (this.autosaveEnabled() && !this.discarding) {
        void this.enqueueWrite(() => this.performSave(draftForm));
      }
    }, 800);
  }

  retrySave(form: ClinicalFormState): Promise<boolean> {
    if (!this.autosaveEnabled() || this.discarding) {
      return Promise.resolve(false);
    }
    this.cancelScheduledSave();
    const draftForm = formForDraft(form);
    this.latestScheduledForm = draftForm;
    this.hasUnsavedChanges.set(true);
    this.errorMessage.set(null);
    return this.enqueueWrite(() => this.performSave(draftForm));
  }

  async reloadDraft(): Promise<boolean> {
    this.cancelScheduledSave();
    this.autosaveEnabled.set(false);
    this.pendingRecovery.set(false);
    await this.writeQueue;
    const generation = ++this.generation;
    this.loading.set(true);
    this.errorMessage.set(null);
    try {
      const snapshot = await fetchClinicalDraft();
      if (generation !== this.generation) {
        return false;
      }
      this.acceptSnapshot(snapshot);
      this.pendingRecovery.set(snapshot.form !== null);
      this.hasUnsavedChanges.set(false);
      this.saveStatus.set(snapshot.form ? 'saved' : 'idle');
      this.loaded = true;
      return true;
    } catch {
      if (generation === this.generation) {
        this.saveStatus.set('error');
        this.errorMessage.set('The saved draft could not be reloaded. Your current input is unchanged.');
      }
      return false;
    } finally {
      if (generation === this.generation) {
        this.loading.set(false);
      }
    }
  }

  async discard(): Promise<boolean> {
    if (!(await this.loadOnOpen()) || this.discarding) {
      return false;
    }
    this.discarding = true;
    this.cancelScheduledSave();
    this.autosaveEnabled.set(false);
    await this.writeQueue;
    const generation = ++this.generation;
    this.saveStatus.set('saving');
    this.errorMessage.set(null);
    try {
      const snapshot = await this.enqueueWrite(() =>
        discardClinicalDraft(this.serverRevision, this.serverDraftKey),
      );
      if (generation !== this.generation) {
        return false;
      }
      this.acceptSnapshot(snapshot);
      this.restorableForm = null;
      this.latestScheduledForm = null;
      this.pendingRecovery.set(false);
      this.hasPersistedDraft.set(false);
      this.hasUnsavedChanges.set(false);
      this.saveStatus.set('idle');
      return true;
    } catch (error) {
      if (generation === this.generation) {
        this.setFailure(error);
      }
      return false;
    } finally {
      this.discarding = false;
    }
  }

  private enqueueWrite<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.writeQueue.then(operation, operation);
    this.writeQueue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  private async performSave(form: ClinicalDraftForm): Promise<boolean> {
    this.saveStatus.set('saving');
    try {
      const payload: ClinicalDraftWrite = {
        expected_revision: this.serverRevision,
        expected_draft_key: this.serverDraftKey,
        form: copyDraftForm(form),
      };
      const snapshot = await saveClinicalDraft(payload);
      this.acceptSnapshot(snapshot);
      this.hasPersistedDraft.set(true);
      const latest = this.latestScheduledForm;
      const sameAsLatest = latest !== null &&
        serializeDraftForm(latest) === serializeDraftForm(form);
      if (sameAsLatest) {
        this.hasUnsavedChanges.set(false);
        this.saveStatus.set('saved');
        this.errorMessage.set(null);
      } else {
        this.saveStatus.set('idle');
      }
      return true;
    } catch (error) {
      this.setFailure(error);
      return false;
    }
  }

  private setFailure(error: unknown): void {
    if (error instanceof ApiHttpError && error.statusCode === 409) {
      this.autosaveEnabled.set(false);
      this.saveStatus.set('conflict');
      this.errorMessage.set(
        'Draft changed or database switched. Reload before saving.',
      );
      return;
    }
    this.saveStatus.set('error');
    this.errorMessage.set('Draft could not be saved. Your changes remain on this screen.');
  }

  private acceptSnapshot(snapshot: ClinicalDraftSnapshot): void {
    this.serverRevision = snapshot.revision;
    this.serverDraftKey = snapshot.draft_key;
    this.restorableForm = snapshot.form ? copyDraftForm(snapshot.form) : null;
    this.lastSavedAt.set(snapshot.updated_at);
    this.hasPersistedDraft.set(snapshot.form !== null);
  }

  private cancelScheduledSave(): void {
    if (this.debounceTimer !== null) {
      globalThis.clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
  }
}
