import { CommonModule } from '@angular/common';
import { Component, ElementRef, HostListener, OnDestroy, ViewChild, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { DEFAULT_FORM_STATE, REPORT_EXPORT_FILENAME } from '../../core/constants';
import { AppStateService } from '../../core/state/app-state.service';
import {
  buildClinicalPayload,
  createDownloadUrl,
  formatErrorMessage,
  formatUnknownError,
  normalizeVisitDateInput,
} from '../../core/utils';
import {
  JobStatus,
  JobStatusResponse,
} from '../../core/models/types';
import {
  fetchClinicalSectionTemplate,
  cancelClinicalJob,
  pollClinicalJobStatus,
  resolvePollIntervalMs,
  startClinicalJob,
  validateClinicalInput,
} from '../../core/services/clinical-api';
import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';

const todayIso = new Date().toISOString().slice(0, 10);
const STAGE_FALLBACK_LABELS: Record<string, string> = {
  session_initialization: 'Step 1/12 - Initializing session context and validating clinical inputs',
  therapy_extraction: 'Step 2/12 - Parsing THERAPY section to extract active treatment lines',
  anamnesis_extraction: 'Step 3/12 - Parsing ANAMNESIS section to identify historical drug exposures',
  anamnesis_disease_extraction: 'Step 4/12 - Parsing ANAMNESIS section to extract comorbidities and risk context',
  anamnesis_lab_extraction: 'Step 5/12 - Parsing LAB ANALYSIS history to reconstruct longitudinal trends',
  hepatotoxicity_pattern: 'Step 6/12 - Computing hepatotoxicity pattern from laboratory trajectory',
  rag_query_building: 'Step 7/12 - Preparing evidence-retrieval query context',
  livertox_lookup: 'Step 8/12 - Cross-checking candidate drugs against LiverTox evidence',
  rucam_estimation: 'Step 9/12 - Estimating per-drug RUCAM scores',
  llm_analysis: 'Step 10/12 - Performing structured LLM causality assessment per candidate drug',
  report_composition: 'Step 11/12 - Drafting integrated clinical assessment and recommendations',
  finalization: 'Step 12/12 - Final consistency checks and session persistence',
};

function isTerminalJobStatus(status: JobStatus | null): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled';
}

@Component({
  selector: 'app-dili-agent-page',
  imports: [CommonModule, FormsModule],
  templateUrl: './dili-agent-page.component.html',
  styleUrl: './dili-agent-page.component.scss',
})
export class DiliAgentPageComponent implements OnDestroy {
  @ViewChild('patientImageInput') private patientImageInput?: ElementRef<HTMLInputElement>;

  readonly stateService = inject(AppStateService);
  private readonly markdownRenderer = inject(MarkdownRendererService);

  readonly isCancelling = signal(false);
  readonly isRunActionLocked = signal(false);
  readonly todayIso = todayIso;
  readonly finalReportMarkdown = computed(() => this.stateService.state().diliAgent.message || this.reportBody);
  readonly renderedReport = computed(() => this.markdownRenderer.render(this.finalReportMarkdown()));
  readonly clinicalInputTemplate = signal('');

  private poller: { stop: () => void } | null = null;
  private runActionLockTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private runControlDebounceTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private runControlDebounced = false;

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
    this.stopPoller();
    this.clearRunActionLock();
    if (this.runControlDebounceTimer !== null) {
      globalThis.clearTimeout(this.runControlDebounceTimer);
      this.runControlDebounceTimer = null;
    }
    this.revokeObjectUrl();
  }

  handleFormChange<K extends keyof typeof this.vm.form>(key: K, value: (typeof this.vm.form)[K]): void {
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
      this.revokeObjectUrl();
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
    } catch {
      this.clinicalInputTemplate.set('');
    }
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

  private revokeObjectUrl(): void {
    if (this.vm.exportUrl) {
      URL.revokeObjectURL(this.vm.exportUrl);
    }
  }

  private resetOutputs(): void {
    this.revokeObjectUrl();
    this.stateService.updateDiliAgent({
      message: '',
      exportUrl: null,
      jobId: null,
      jobProgress: 0,
      jobStatus: null,
      jobStage: null,
      jobStageMessage: null,
    });
  }

  private stopPoller(): void {
    if (this.poller) {
      this.poller.stop();
      this.poller = null;
    }
  }

  private onPollingError(pollError: string): void {
    this.stopPoller();
    this.isCancelling.set(false);
    this.stateService.updateDiliAgent({
      message: formatErrorMessage(pollError),
      exportUrl: null,
      jobStage: null,
      jobStageMessage: null,
      isStarting: false,
      isRunning: false,
    });
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

  private onJobStatusUpdate(status: JobStatusResponse): void {
    const terminalStatus = isTerminalJobStatus(status.status);
    const stage =
      status.result && typeof status.result.progress_stage === 'string'
        ? status.result.progress_stage
        : null;
    const stageMessage =
      status.result && typeof status.result.progress_message === 'string'
        ? status.result.progress_message
        : null;
    const resolvedProgress = typeof status.progress === 'number' ? status.progress : 0;

    this.stateService.updateDiliAgent({
      jobProgress: terminalStatus ? 100 : resolvedProgress,
      jobStatus: status.status,
      jobStage: stage,
      jobStageMessage: stageMessage,
      isStarting: false,
      isRunning: !terminalStatus,
    });

    if (!terminalStatus) {
      return;
    }

    this.stopPoller();
    this.isCancelling.set(false);

    if (status.status === 'completed') {
      const report = typeof status.result?.report === 'string' ? status.result.report : '';
      const newExportUrl = report ? createDownloadUrl(report, REPORT_EXPORT_FILENAME) : null;
      this.stateService.updateDiliAgent({
        message: report || '[INFO] Clinical analysis completed.',
        exportUrl: newExportUrl,
        jobStage: null,
        jobStageMessage: null,
      });
    } else if (status.status === 'failed') {
      const errorMessage = status.error
        ? formatErrorMessage(status.error)
        : '[ERROR] Clinical analysis failed.';
      this.stateService.updateDiliAgent({
        message: errorMessage,
        exportUrl: null,
        jobStage: null,
        jobStageMessage: null,
      });
    } else if (status.status === 'cancelled') {
      this.stateService.updateDiliAgent({
        message: '[INFO] Clinical analysis cancelled.',
        exportUrl: null,
        jobStage: null,
        jobStageMessage: null,
      });
    }
  }

  private startPolling(jobIdToPoll: string, intervalMs: number): void {
    this.stopPoller();
    this.poller = pollClinicalJobStatus(
      jobIdToPoll,
      intervalMs,
      (status) => this.onJobStatusUpdate(status),
      (message) => this.onPollingError(message),
    );
  }

  private async executeRunSession(): Promise<void> {
    if (this.vm.isStarting || this.vm.isRunning || this.isRunActionLocked()) {
      return;
    }
    this.lockRunAction();
    this.isCancelling.set(false);
    this.stateService.updateDiliAgent({ isStarting: true, isRunning: true });
    this.resetOutputs();
    this.stopPoller();

    try {
      const payload = buildClinicalPayload(this.vm.form, this.vm.settings);
      const preflight = await validateClinicalInput(payload);
      if (!preflight.ready) {
        this.stateService.updateDiliAgent({
          isStarting: false,
          isRunning: false,
          message: `[ERROR] ${preflight.blocking_issues.map((issue) => issue.message).join(' ')}`,
        });
        return;
      }
      const preflightWarningSummary = preflight.non_blocking_issues.length
        ? `[WARN] ${preflight.non_blocking_issues.map((issue) => issue.message).join(' ')}`
        : null;
      const startResult = await startClinicalJob(payload);
      this.stateService.updateDiliAgent({
        jobId: startResult.job_id,
        jobProgress: 0,
        jobStatus: startResult.status,
        jobStage: 'session_initialization',
        jobStageMessage: preflightWarningSummary ?? STAGE_FALLBACK_LABELS['session_initialization'],
        isStarting: false,
      });
      const intervalMs = resolvePollIntervalMs(startResult.poll_interval);
      this.startPolling(startResult.job_id, intervalMs);
    } catch (error) {
      this.stateService.updateDiliAgent({
        message: formatUnknownError(error, 'Unexpected error'),
        exportUrl: null,
        jobStage: null,
        jobStageMessage: null,
        isStarting: false,
        isRunning: false,
      });
    }
  }

  async runSession(): Promise<void> {
    const message = this.validationMessage();
    if (message) {
      this.resetOutputs();
      this.stateService.updateDiliAgent({
        isRunning: false,
        message,
      });
      return;
    }

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
      await cancelClinicalJob(this.vm.jobId);
      this.stateService.updateDiliAgent({
        message: '[INFO] Cancellation requested. Waiting for worker shutdown...',
      });
    } catch (error) {
      this.stateService.updateDiliAgent({
        message: formatUnknownError(error, 'Failed to request cancellation.'),
      });
    } finally {
      this.isCancelling.set(false);
    }
  }

  clearAll(): void {
    this.revokeObjectUrl();
    this.stateService.updateDiliAgent({
      form: { ...DEFAULT_FORM_STATE },
      message: '',
      exportUrl: null,
      jobId: null,
      jobProgress: 0,
      jobStatus: null,
      jobStage: null,
      jobStageMessage: null,
      isStarting: false,
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
    const baseLabel = this.vm.jobStageMessage
      || (this.vm.jobStage ? STAGE_FALLBACK_LABELS[this.vm.jobStage] : null)
      || 'Starting clinical analysis';
    const suffix = this.vm.jobProgress > 0
      ? `... ${this.vm.jobProgress.toFixed(0)}%`
      : '...';
    return `${baseLabel}${suffix}`;
  }

  get runActionDisabled(): boolean {
    if (this.vm.isRunning) {
      return this.isCancelling();
    }
    return this.isCancelling()
      || this.vm.isStarting
      || this.isRunActionLocked()
      || !this.canStartSession();
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

  get runDisabledReason(): string | null {
    if (this.vm.isRunning || this.vm.isStarting || this.isCancelling()) {
      return null;
    }
    if (!this.vm.form.visitDate.trim()) {
      return 'Add a visit date to run analysis.';
    }
    if (!this.vm.form.clinicalInput.trim()) {
      return 'Add the clinical input to run analysis.';
    }
    if (!this.clinicalInputMeetsMinimumLength()) {
      return 'Clinical input needs at least 60 words.';
    }
    if (!this.hasSelectedModelProvider()) {
      return 'Select a model provider to run analysis.';
    }
    return null;
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

  private countClinicalWords(value: string): number {
    const matches = value.match(/\b[\wÀ-ÖØ-öø-ÿ']+\b/gu);
    return matches ? matches.length : 0;
  }

  private selectedModelProviders(): string[] {
    const provider = this.vm.settings.provider?.trim();
    return provider ? [provider] : [];
  }

  private clinicalInputMeetsMinimumLength(): boolean {
    return this.countClinicalWords(this.vm.form.clinicalInput.trim()) >= 60;
  }

  private hasSelectedModelProvider(): boolean {
    return this.selectedModelProviders().length > 0;
  }

  canStartSession(): boolean {
    return Boolean(this.vm.form.visitDate.trim())
      && Boolean(this.vm.form.clinicalInput.trim())
      && this.clinicalInputMeetsMinimumLength()
      && this.hasSelectedModelProvider();
  }

  validationMessage(): string | null {
    if (!this.vm.form.visitDate.trim()) {
      return '[ERROR] Provide the visit date.';
    }
    if (!this.vm.form.clinicalInput.trim()) {
      return '[ERROR] Provide the clinical input.';
    }
    if (!this.clinicalInputMeetsMinimumLength()) {
      return '[ERROR] Clinical input must contain at least 60 words.';
    }
    if (!this.hasSelectedModelProvider()) {
      return '[ERROR] Select at least one model provider.';
    }
    return null;
  }
}
