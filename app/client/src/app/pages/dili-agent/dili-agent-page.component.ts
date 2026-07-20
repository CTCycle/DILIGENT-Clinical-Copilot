import { CommonModule } from '@angular/common';
import { Component, ElementRef, HostListener, OnDestroy, ViewChild, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ModalShellComponent } from '../../components/modal-shell/modal-shell.component';
import { DEFAULT_FORM_STATE, REPORT_EXPORT_FILENAME } from '../../core/constants';
import {
  ClinicalInputPreflightIssue,
  ClinicalInputPreflightResponse,
  ClinicalRequestPayload,
  JobStatus,
} from '../../core/models/types';
import { validateClinicalInput, fetchClinicalSectionTemplate } from '../../core/services/clinical-api';
import { DiliJobTrackerService } from '../../core/services/dili-job-tracker.service';
import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';
import { AppStateService } from '../../core/state/app-state.service';
import {
  buildClinicalPayload,
  createDownloadUrl,
  formatUnknownError,
  normalizeVisitDateInput,
} from '../../core/utils';

const todayIso = new Date().toISOString().slice(0, 10);
const STALL_THRESHOLD_MS = 600_000;
const PREFLIGHT_FIELD_TARGETS: Record<string, string> = {
  anamnesis: 'clinical-input',
  clinical_input: 'clinical-input',
  drugs: 'clinical-input',
  laboratory_analysis: 'clinical-input',
  selected_model_providers: 'run-analysis-button',
  use_rag: 'rag-enabled',
  visit_date: 'visit-date',
};
const CLINICAL_INPUT_TEMPLATE_FALLBACK = `Chief concern:
History of present illness:
Current medications (dose/start date):
Recent labs (ALT/AST/ALP/Total bilirubin with dates):
Relevant imaging or procedures:
Comorbidities / liver risk factors:
Timeline of symptoms and treatment changes:
Working clinical question:`;

function isTerminalJobStatus(status: JobStatus | null): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled';
}

@Component({
  selector: 'app-dili-agent-page',
  imports: [CommonModule, FormsModule, ModalShellComponent],
  templateUrl: './dili-agent-page.component.html',
  styleUrl: './dili-agent-page.component.scss',
})
export class DiliAgentPageComponent implements OnDestroy {
  @ViewChild('patientImageInput') private patientImageInput?: ElementRef<HTMLInputElement>;
  @ViewChild('runAnalysisButton') private runAnalysisButton?: ElementRef<HTMLButtonElement>;

  readonly stateService = inject(AppStateService);
  private readonly markdownRenderer = inject(MarkdownRendererService);
  private readonly diliJobTracker = inject(DiliJobTrackerService);

  readonly isCancelling = signal(false);
  readonly isRunActionLocked = signal(false);
  readonly isPreflightChecking = signal(false);
  readonly preflightDialog = signal<ClinicalInputPreflightResponse | null>(null);
  readonly preflightContinuationInFlight = signal(false);
  readonly preflightIssues = computed<ClinicalInputPreflightIssue[]>(() => {
    const result = this.preflightDialog();
    return result
      ? [...result.blocking_issues, ...result.non_blocking_issues]
      : [];
  });
  readonly preflightHasBlockingIssues = computed(
    () => Boolean(this.preflightDialog()?.blocking_issues.length),
  );
  readonly preflightBlockingCount = computed(
    () => this.preflightDialog()?.blocking_issues.length ?? 0,
  );
  readonly preflightWarningCount = computed(
    () => this.preflightDialog()?.non_blocking_issues.length ?? 0,
  );
  readonly todayIso = todayIso;
  readonly finalReportMarkdown = computed(() => this.stateService.state().diliAgent.message || this.reportBody);
  readonly renderedReport = computed(() => this.markdownRenderer.render(this.finalReportMarkdown()));
  readonly clinicalInputTemplate = signal('');

  private runActionLockTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private runControlDebounceTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private runControlDebounced = false;
  private pendingPreflightPayload: ClinicalRequestPayload | null = null;
  private pendingPreflightFocusTargetId: string | null = null;
  private preflightAttempt = 0;

  constructor() {
    void this.loadClinicalSectionTemplate();
    effect(() => {
      const state = this.vm;
      if (
        state.jobStatus === 'completed' &&
        state.message &&
        !state.exportUrl &&
        !state.isRunning &&
        !state.isStarting
      ) {
        const restoredExportUrl = createDownloadUrl(state.message, REPORT_EXPORT_FILENAME);
        this.stateService.updateDiliAgent({ exportUrl: restoredExportUrl });
      }
    });
  }

  get vm() {
    return this.stateService.state().diliAgent;
  }

  ngOnDestroy(): void {
    this.clearRunActionLock();
    if (this.runControlDebounceTimer !== null) {
      globalThis.clearTimeout(this.runControlDebounceTimer);
      this.runControlDebounceTimer = null;
    }
  }

  handleFormChange<K extends keyof typeof this.vm.form>(key: K, value: (typeof this.vm.form)[K]): void {
    this.invalidatePreflightState();
    const currentMessage = this.vm.message ?? '';
    const shouldClearStaleOutput =
      !this.vm.isRunning &&
      !this.vm.isStarting &&
      Boolean(currentMessage || this.vm.exportUrl || this.vm.jobId || this.vm.jobStatus);
    const shouldClearValidationMessage =
      !this.vm.isRunning &&
      !this.vm.isStarting &&
      currentMessage.startsWith('[ERROR]') &&
      currentMessage !== '[ERROR] Clinical analysis failed.';

    if (shouldClearStaleOutput && this.vm.exportUrl) {
      this.diliJobTracker.revokeCurrentExportUrl();
    }
    this.stateService.updateDiliAgent({
      form: {
        ...this.vm.form,
        [key]: value,
      },
      ...(shouldClearStaleOutput || shouldClearValidationMessage
        ? {
            message: '',
            exportUrl: null,
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

  private invalidatePreflightState(): void {
    this.preflightAttempt += 1;
    this.isPreflightChecking.set(false);
    this.preflightDialog.set(null);
    this.pendingPreflightPayload = null;
    this.preflightContinuationInFlight.set(false);
  }

  returnToInputFromPreflight(): void {
    this.pendingPreflightFocusTargetId = this.resolveFirstAffectedControlId();
    this.invalidatePreflightState();
    this.clearRunActionLock();
  }

  focusPreflightTarget(): void {
    const targetId = this.pendingPreflightFocusTargetId;
    this.pendingPreflightFocusTargetId = null;
    const target = targetId
      ? document.getElementById(targetId)
      : this.runAnalysisButton?.nativeElement;
    target?.focus();
  }

  async continueAfterPreflight(): Promise<void> {
    if (
      this.preflightHasBlockingIssues()
      || this.preflightContinuationInFlight()
      || !this.pendingPreflightPayload
    ) {
      return;
    }
    this.preflightContinuationInFlight.set(true);
    const result = this.preflightDialog();
    const requiresRagFallback = Boolean(
      result?.non_blocking_issues.some((issue) => issue.field === 'use_rag'),
    );
    const payload = requiresRagFallback
      ? { ...this.pendingPreflightPayload, use_rag: false }
      : this.pendingPreflightPayload;
    this.preflightDialog.set(null);
    this.pendingPreflightPayload = null;
    this.clearRunActionLock();
    try {
      await this.startValidatedSession(payload);
    } finally {
      this.preflightContinuationInFlight.set(false);
    }
  }

  private resolveFirstAffectedControlId(): string | null {
    for (const issue of this.preflightIssues()) {
      const target = issue.field ? PREFLIGHT_FIELD_TARGETS[issue.field] : null;
      if (target) {
        return target;
      }
    }
    return null;
  }

  private clearRunActionLock(): void {
    if (this.runActionLockTimer !== null) {
      globalThis.clearTimeout(this.runActionLockTimer);
      this.runActionLockTimer = null;
    }
    this.isRunActionLocked.set(false);
  }

  private lockRunAction(windowMs: number = 1750): void {
    this.clearRunActionLock();
    this.isRunActionLocked.set(true);
    this.runActionLockTimer = globalThis.setTimeout(() => {
      this.runActionLockTimer = null;
      this.isRunActionLocked.set(false);
    }, Math.max(500, windowMs));
  }

  private async executeRunSession(
    payloadOverride?: ClinicalRequestPayload,
  ): Promise<void> {
    if (
      this.vm.isStarting
      || this.vm.isRunning
      || this.isRunActionLocked()
      || this.isPreflightChecking()
    ) {
      return;
    }

    this.lockRunAction();
    this.isCancelling.set(false);
    this.invalidatePreflightState();
    const payload = payloadOverride ?? buildClinicalPayload(this.vm.form, this.vm.settings);
    const attempt = ++this.preflightAttempt;
    this.isPreflightChecking.set(true);

    try {
      const preflight = await validateClinicalInput(payload);
      if (attempt !== this.preflightAttempt) {
        return;
      }
      const issues = [
        ...preflight.blocking_issues,
        ...preflight.non_blocking_issues,
      ];
      if (issues.length) {
        this.pendingPreflightPayload = payload;
        this.preflightDialog.set(preflight);
        return;
      }
      await this.startValidatedSession(payload);
    } catch (error) {
      if (attempt !== this.preflightAttempt) {
        return;
      }
      console.error('Clinical pre-flight request failed.', error);
      const description =
        'The application could not complete the safety checks. Verify backend connectivity and runtime configuration, then retry.';
      this.pendingPreflightPayload = null;
      this.preflightDialog.set({
        ready: false,
        blocking_issues: [
          {
            severity: 'blocking',
            code: 'preflight_request_failed',
            message: description,
            field: null,
            title: 'Pre-flight checks are unavailable',
            description,
            affected_section: 'Analysis readiness',
            consequence: 'The analysis cannot start until the safety checks complete successfully.',
            continuation_allowed: false,
          },
        ],
        non_blocking_issues: [],
        runtime_settings: {},
        extraction_quality: {},
        deterministic_diagnostics: {},
        rag_readiness: null,
      });
    } finally {
      if (attempt === this.preflightAttempt) {
        this.isPreflightChecking.set(false);
      }
    }
  }

  private async startValidatedSession(payload: ClinicalRequestPayload): Promise<void> {
    await this.diliJobTracker.startSession(payload, null);
  }

  async runSession(): Promise<void> {
    await this.executeRunSession();
  }

  async stopSession(): Promise<void> {
    if (this.vm.isStarting && !this.vm.jobId) {
      return;
    }
    if (!this.vm.jobId) {
      return;
    }
    this.lockRunAction(5000);
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

  clearAll(): void {
    this.invalidatePreflightState();
    this.diliJobTracker.clearJobState();
    this.stateService.updateDiliAgent({
      form: { ...DEFAULT_FORM_STATE },
    });
  }

  async copyReport(): Promise<void> {
    const rendered = this.renderedReport();
    if (!rendered.text) {
      return;
    }
    const clipboardItemCtor = (globalThis as { ClipboardItem?: typeof ClipboardItem }).ClipboardItem;
    if (navigator.clipboard && clipboardItemCtor) {
      const item = new clipboardItemCtor({
        'text/html': new Blob([rendered.html], { type: 'text/html' }),
        'text/plain': new Blob([rendered.text], { type: 'text/plain' }),
      });
      await navigator.clipboard.write([item]);
      return;
    }
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(rendered.text);
    }
  }

  toggleReportExpanded(): void {
    this.stateService.updateDiliAgent({ isExpanded: !this.vm.isExpanded });
  }

  collapseReport(): void {
    this.stateService.updateDiliAgent({ isExpanded: false });
  }

  downloadReport(): void {
    if (!this.vm.exportUrl) return;
    const anchor = document.createElement('a');
    anchor.href = this.vm.exportUrl;
    anchor.download = REPORT_EXPORT_FILENAME;
    anchor.click();
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.preflightDialog()) {
      return;
    }
    if (this.vm.isExpanded) {
      this.collapseReport();
    }
  }

  runOrStop(): void {
    if (this.isCancelling()) {
      return;
    }
    if (this.vm.isRunning) {
      void this.stopSession();
      return;
    }
    if (this.runControlDebounced) {
      return;
    }
    if (this.vm.isStarting || this.isRunActionLocked()) {
      return;
    }
    this.runControlDebounced = true;
    if (this.runControlDebounceTimer !== null) {
      globalThis.clearTimeout(this.runControlDebounceTimer);
    }
    this.runControlDebounceTimer = globalThis.setTimeout(() => {
      this.runControlDebounced = false;
      this.runControlDebounceTimer = null;
    }, 1000);
    void this.runSession();
  }

  get showSpinner(): boolean {
    return this.vm.isRunning && !isTerminalJobStatus(this.vm.jobStatus);
  }

  get spinnerStatusLabel(): string {
    const baseLabel = this.vm.jobStageMessage || 'Starting clinical analysis';
    const progressSuffix = this.vm.jobProgress > 0
      ? `... ${this.vm.jobProgress.toFixed(0)}%`
      : '...';

    const elapsedMs = this.vm.jobStartedAtMs ? Date.now() - this.vm.jobStartedAtMs : 0;
    let elapsedSuffix = '';
    if (elapsedMs >= 1000) {
      const totalSeconds = Math.floor(elapsedMs / 1000);
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      elapsedSuffix = minutes > 0
        ? ` (${minutes}m ${seconds}s)`
        : ` (${seconds}s)`;
    }

    const stalledSinceMs = this.vm.jobLastProgressAtMs ? Date.now() - this.vm.jobLastProgressAtMs : 0;
    const stalled = Boolean(this.vm.jobStartedAtMs) && stalledSinceMs >= STALL_THRESHOLD_MS;
    const stallSuffix = stalled ? ' - this step is taking longer than expected, analysis is still running' : '';
    const localExtractionSuffix = this.isExtractionStage(this.vm.jobStage)
      ? ' Local model extraction can take several minutes.'
      : '';

    return `${baseLabel}${progressSuffix}${elapsedSuffix}${stallSuffix}${localExtractionSuffix}`;
  }

  private isExtractionStage(stage: string | null): boolean {
    return Boolean(stage && stage.includes('extraction'));
  }

  get runActionDisabled(): boolean {
    if (this.vm.isRunning) {
      return this.isCancelling();
    }
    return this.isCancelling()
      || this.vm.isStarting
      || this.isRunActionLocked()
      || this.isPreflightChecking();
  }

  get runActionLabel(): string {
    if (this.isPreflightChecking()) {
      return 'Checking inputs...';
    }
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

  get reportBody(): string {
    return this.vm.message || 'No report generated yet. Run analysis to see results.';
  }

  get patientNameLabel(): string {
    return this.vm.form.patientName.trim() || 'Unnamed patient';
  }

  get patientImageDataUrl(): string | null {
    return this.vm.form.patientImageDataUrl;
  }

  get recordedDateLabel(): string {
    if (!this.vm.form.visitDate) {
      return 'Not set';
    }
    return new Date(`${this.vm.form.visitDate}T00:00:00`).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  }


}
