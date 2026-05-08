import { CommonModule } from '@angular/common';
import { Component, ElementRef, HostListener, OnDestroy, ViewChild, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { DEFAULT_FORM_STATE, REPORT_EXPORT_FILENAME } from '../../core/constants';
import { CLINICAL_SECTION_TEMPLATE } from '../../core/clinical-section-template';
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
  cancelClinicalJob,
  pollClinicalJobStatus,
  startClinicalJob,
} from '../../core/services/clinical-api';
import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';

const todayIso = new Date().toISOString().slice(0, 10);
const DEFAULT_POLL_INTERVAL_MS = 1000;

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
  readonly finalReportMarkdown = computed(() => this.stateService.state().diliAgent.message ?? '');
  readonly renderedReport = computed(() => this.markdownRenderer.render(this.finalReportMarkdown()));
  readonly clinicalInputTemplate = CLINICAL_SECTION_TEMPLATE;

  private poller: { stop: () => void } | null = null;
  private runActionLockTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private runControlDebounceTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private runControlDebounced = false;

  constructor() {
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
    const shouldClearValidationMessage =
      !this.vm.isRunning &&
      !this.vm.isStarting &&
      currentMessage.startsWith('[ERROR]') &&
      currentMessage !== '[ERROR] Clinical analysis failed.';

    this.stateService.updateDiliAgent({
      form: {
        ...this.vm.form,
        [key]: value,
      },
      ...(shouldClearValidationMessage ? { message: '' } : {}),
    });
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
      const startResult = await startClinicalJob(payload);
      this.stateService.updateDiliAgent({
        jobId: startResult.job_id,
        jobProgress: 0,
        jobStatus: startResult.status,
        jobStage: 'session_initialization',
        jobStageMessage: 'Initializing clinical session',
        isStarting: false,
      });
      const intervalMs = startResult.poll_interval * 1000;
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
    if (this.runControlDebounced || this.vm.isStarting || this.isCancelling() || this.isRunActionLocked()) {
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
    if (this.vm.isRunning) {
      void this.stopSession();
      return;
    }
    void this.runSession();
  }

  get showSpinner(): boolean {
    return this.vm.isRunning && !isTerminalJobStatus(this.vm.jobStatus);
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

  get defaultPollIntervalMs(): number {
    return DEFAULT_POLL_INTERVAL_MS;
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
