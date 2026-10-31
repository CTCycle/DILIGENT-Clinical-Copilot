// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { CommonModule } from '@angular/common';
import { Component, ElementRef, ViewChild, computed, inject, signal, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { AnalysisProgressComponent } from '../../components/analysis-progress/analysis-progress.component';
import { ModalShellComponent } from '../../components/modal-shell/modal-shell.component';
import { DEFAULT_FORM_STATE } from '../../core/constants';
import { DILI_ASSESSMENT_TOUR, GUIDANCE_DEFINITIONS } from '../../core/guidance/guidance-content';
import { FeatureTipComponent } from '../../core/guidance/feature-tip.component';
import { GuidanceTourService } from '../../core/guidance/guidance-tour.service';
import { ClinicalDraftForm } from '../../core/models/types';
import { fetchClinicalSectionTemplate } from '../../core/services/clinical-api';
import { DiliJobTrackerService } from '../../core/services/dili-job-tracker.service';
import { LucideSave } from '@lucide/angular';
import { ClinicalDraftService } from '../../core/services/clinical-draft.service';
import { NotificationService } from '../../core/services/notification.service';
import { AppStateService } from '../../core/state/app-state.service';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { formatUnknownError, normalizeVisitDateInput } from '../../core/utils';
import { AnalysisWizardComponent } from './components/analysis-wizard/analysis-wizard.component';
import { SourceSelection } from './components/analysis-wizard/section-preview';

const todayIso = new Date().toISOString().slice(0, 10);
const CLINICAL_INPUT_TEMPLATE_FALLBACK = `Chief concern:
History of present illness:
Current medications (dose/start date):
Recent labs (ALT/AST/ALP/Total bilirubin with dates):
Relevant imaging or procedures:
Comorbidities / liver risk factors:
Timeline of symptoms and treatment changes:
Working clinical question:`;

type DraftConfirmationAction = 'resume' | 'discard' | 'clear';

@Component({
  selector: 'app-dili-agent-page',
  imports: [
    CommonModule,
    FormsModule,
    AnalysisProgressComponent,
    AnalysisWizardComponent,
    ModalShellComponent,
    FeatureTipComponent,
    LucideSave,
  ],
  templateUrl: './dili-agent-page.component.html',
  styleUrl: './dili-agent-page.component.scss',
})
export class DiliAgentPageComponent {
  @ViewChild('patientImageInput') private patientImageInput?: ElementRef<HTMLInputElement>;

  readonly stateService = inject(AppStateService);
  readonly modelConfigState = inject(ModelConfigStateService);
  readonly guidanceDefinitions = GUIDANCE_DEFINITIONS;
  private readonly router = inject(Router);
  private readonly guidanceTour = inject(GuidanceTourService);
  private readonly diliJobTracker = inject(DiliJobTrackerService);
  readonly clinicalDraft = inject(ClinicalDraftService);
  private readonly notifications = inject(NotificationService);
  private readonly wizard = viewChild(AnalysisWizardComponent);

  readonly isCancelling = signal(false);
  readonly wizardOpen = signal(false);
  readonly todayIso = todayIso;
  readonly clinicalInputTemplate = signal('');
  readonly draftConfirmation = signal<DraftConfirmationAction | null>(null);
  readonly hasReport = computed(() => {
    const vm = this.vm;
    return vm.jobStatus === 'completed' && Boolean(vm.message);
  });
  readonly canViewReport = computed(() => this.hasReport() || this.isActiveJob || this.vm.sessionId !== null);
  readonly showProgress = computed(() => this.vm.jobStatus !== null || this.vm.isStarting || this.vm.isRunning);
  readonly progressError = computed(() => (this.vm.jobStatus === 'failed' ? this.vm.message : null));

  constructor() {
    void this.clinicalDraft.loadOnOpen();
    void this.loadClinicalSectionTemplate();
    void this.modelConfigState.load().catch(() => undefined);
  }

  get vm() {
    return this.stateService.state().diliAgent;
  }

  startAssessmentTour(): void {
    this.guidanceTour.start(DILI_ASSESSMENT_TOUR);
  }

  openConfigurations(): void {
    void this.router.navigateByUrl('/settings/models');
  }

  viewReport(): void {
    void this.router.navigateByUrl('/report');
  }

  handleFormChange<K extends keyof typeof this.vm.form>(key: K, value: (typeof this.vm.form)[K]): void {
    const currentMessage = this.vm.message ?? '';
    const shouldClearStaleOutput =
      !this.vm.isRunning &&
      !this.vm.isStarting &&
      Boolean(currentMessage || this.vm.jobId || this.vm.jobStatus);
    const shouldClearValidationMessage =
      !this.vm.isRunning &&
      !this.vm.isStarting &&
      currentMessage.startsWith('[ERROR]') &&
      currentMessage !== '[ERROR] Clinical analysis failed.';

    const updatedForm = { ...this.vm.form, [key]: value };
    this.stateService.updateDiliAgent({
      form: updatedForm,
      ...(shouldClearStaleOutput || shouldClearValidationMessage
        ? {
            message: '',
            jobId: null,
            jobProgress: 0,
            jobStatus: null,
            jobStage: null,
            jobStageMessage: null,
            jobStartedAtMs: null,
            jobLastProgressAtMs: null,
            pollIntervalMs: null,
          }
        : {}),
    });
    if (this.clinicalDraft.autosaveEnabled()) {
      this.clinicalDraft.scheduleSave(updatedForm);
    }
  }

  private async loadClinicalSectionTemplate(): Promise<void> {
    try {
      const response = await fetchClinicalSectionTemplate();
      if (response.template?.trim()) {
        this.clinicalInputTemplate.set(response.template.trim());
        return;
      }
    } catch {}
    this.clinicalInputTemplate.set(CLINICAL_INPUT_TEMPLATE_FALLBACK);
  }

  handleVisitDateChange(value: string): void {
    this.handleFormChange('visitDate', normalizeVisitDateInput(value));
  }

  openPatientImagePicker(): void {
    this.patientImageInput?.nativeElement.click();
  }

  handlePatientImageSelection(event: Event): void {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) {
      return;
    }
    const file = target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = typeof reader.result === 'string' ? reader.result : null;
      this.handleFormChange('patientImageDataUrl', dataUrl);
    };
    reader.readAsDataURL(file);
  }

  closeWizard(): void {
    this.wizardOpen.set(false);
  }

  fixInput(targetId: string | null): void {
    this.closeWizard();
    globalThis.setTimeout(() => document.getElementById(targetId ?? 'clinical-input')?.focus());
  }

  editSource(selection: SourceSelection | null): void {
    this.closeWizard();
    globalThis.setTimeout(() => {
      const input = document.getElementById('clinical-input') as HTMLTextAreaElement | null;
      input?.focus();
      if (input && selection && selection.end <= input.value.length) {
        input.setSelectionRange(selection.start, selection.end);
      }
    });
  }

  async stopSession(): Promise<void> {
    if (!this.vm.jobId) {
      return;
    }
    this.isCancelling.set(true);
    try {
      await this.diliJobTracker.cancelSession();
    } catch (error) {
      this.stateService.updateDiliAgent({
        message: formatUnknownError(error, 'Failed to request cancellation.'),
      });
    } finally {
      this.isCancelling.set(false);
    }
  }

  async saveDraft(): Promise<void> {
    const saved = await this.clinicalDraft.saveAndEnable(this.vm.form);
    if (saved) {
      this.clinicalDraft.scheduleSave(this.vm.form);
      this.notifications.notify('Draft saved', 'success', 2500);
    }
  }

  resumeDraft(): void {
    if (this.isActiveJob) {
      return;
    }
    const draft = this.clinicalDraft.getRestorableForm();
    if (!draft) {
      return;
    }
    const current = this.vm.form;
    const hasCurrentInput = Boolean(
      current.patientName || current.visitDate || current.clinicalInput || current.patientImageDataUrl,
    );
    const differs = current.patientName !== draft.patient_name ||
      current.visitDate !== (draft.visit_date ?? '') ||
      current.clinicalInput !== draft.clinical_input ||
      current.patientImageDataUrl !== null;
    if (hasCurrentInput && differs) {
      this.draftConfirmation.set('resume');
      return;
    }
    this.applyResumedDraft(draft);
  }

  discardDraft(): void {
    if (this.clinicalDraft.hasPersistedDraft()) {
      this.draftConfirmation.set('discard');
    }
  }

  clearAll(): void {
    if (this.isActiveJob) {
      return;
    }
    if (this.clinicalDraft.hasPersistedDraft()) {
      this.draftConfirmation.set('clear');
      return;
    }
    this.performClearAll();
  }

  async confirmDraftAction(): Promise<void> {
    const action = this.draftConfirmation();
    this.draftConfirmation.set(null);
    if (action === 'resume') {
      const draft = this.clinicalDraft.getRestorableForm();
      if (draft) {
        this.applyResumedDraft(draft);
      }
      return;
    }
    if (action !== 'discard' && action !== 'clear') {
      return;
    }
    if (!(await this.clinicalDraft.discard())) {
      return;
    }
    this.stateService.updateDiliAgent({ form: { ...DEFAULT_FORM_STATE } });
    if (action === 'clear') {
      this.performClearAll();
    }
  }

  closeDraftConfirmation(): void {
    this.draftConfirmation.set(null);
  }

  async reloadDraft(): Promise<void> {
    await this.clinicalDraft.reloadDraft();
  }

  get isActiveJob(): boolean {
    return this.vm.isRunning || this.vm.isStarting ||
      this.vm.jobStatus === 'pending' || this.vm.jobStatus === 'running';
  }

  get draftStatusText(): string {
    switch (this.clinicalDraft.saveStatus()) {
      case 'saving':
        return 'Saving';
      case 'conflict':
        return 'Draft changed or database switched. Reload before saving.';
      case 'error':
        return 'Unsaved changes. Draft could not be saved.';
      case 'saved': {
        if (this.clinicalDraft.hasUnsavedChanges()) {
          return 'Unsaved changes';
        }
        const savedAt = this.clinicalDraft.lastSavedAt();
        return savedAt
          ? `Saved at ${new Date(savedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
          : 'Saved';
      }
      default:
        return this.clinicalDraft.hasUnsavedChanges() ? 'Unsaved changes' : '';
    }
  }

  get draftConfirmationTitle(): string {
    return this.draftConfirmation() === 'resume'
      ? 'Replace current input?'
      : this.draftConfirmation() === 'clear'
        ? 'Discard saved draft and clear all?'
        : 'Discard saved draft?';
  }

  get draftConfirmationDescription(): string {
    if (this.draftConfirmation() === 'resume') {
      return 'Resuming replaces the current form with the saved draft. The selected patient image is not part of the draft and must be chosen again.';
    }
    return 'This permanently removes the saved draft from the configured database and clears the current form. This does not change an assessment already in progress.';
  }

  private applyResumedDraft(draft: ClinicalDraftForm): void {
    if (this.isActiveJob) {
      return;
    }
    this.diliJobTracker.clearJobState();
    this.stateService.updateDiliAgent({
      form: {
        patientName: draft.patient_name,
        visitDate: draft.visit_date ?? '',
        patientImageDataUrl: null,
        clinicalInput: draft.clinical_input,
      },
    });
    this.clinicalDraft.resume();
  }

  private performClearAll(): void {
    this.diliJobTracker.clearJobState();
    this.wizard()?.reset();
    this.stateService.updateDiliAgent({
      form: { ...DEFAULT_FORM_STATE },
    });
  }

  runOrStop(): void {
    if (this.isCancelling() || this.vm.isStarting) {
      return;
    }
    if (this.vm.isRunning) {
      void this.stopSession();
      return;
    }
    this.wizardOpen.set(true);
  }

  get runActionDisabled(): boolean {
    return this.isCancelling() || this.vm.isStarting;
  }

  get runActionLabel(): string {
    if (!this.vm.isRunning) {
      return 'Run DILI analysis';
    }
    if (this.isCancelling()) {
      return 'Stopping...';
    }
    if (!this.vm.jobId) {
      return 'Starting...';
    }
    return 'Stop analysis';
  }

  get patientNameLabel(): string {
    return this.vm.form.patientName.trim() || 'Unnamed patient';
  }

  get patientImageDataUrl(): string | null {
    return this.vm.form.patientImageDataUrl;
  }
}
