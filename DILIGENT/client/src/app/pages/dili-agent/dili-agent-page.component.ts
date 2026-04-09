import { CommonModule } from '@angular/common';
import { Component, OnDestroy, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ModalShellComponent } from '../../components/modal-shell/modal-shell.component';
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
  cancelClinicalJob,
  pollClinicalJobStatus,
  startClinicalJob,
} from '../../core/services/api';

const todayIso = new Date().toISOString().slice(0, 10);
const DEFAULT_POLL_INTERVAL_MS = 1000;

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
  readonly stateService = inject(AppStateService);

  readonly isCancelling = signal(false);
  readonly isMissingLabsModalOpen = signal(false);
  readonly todayIso = todayIso;

  private poller: { stop: () => void } | null = null;

  get vm() {
    return this.stateService.state().diliAgent;
  }

  ngOnDestroy(): void {
    this.stopPoller();
    this.revokeObjectUrl();
  }

  handleFormChange<K extends keyof typeof this.vm.form>(key: K, value: (typeof this.vm.form)[K]): void {
    this.stateService.updateDiliAgent({
      form: {
        ...this.vm.form,
        [key]: value,
      },
    });
  }

  handleVisitDateChange(value: string): void {
    this.handleFormChange('visitDate', normalizeVisitDateInput(value));
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
      isRunning: false,
    });
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

  private async executeRunSession(allowMissingLabs: boolean | null): Promise<void> {
    this.isCancelling.set(false);
    this.stateService.updateDiliAgent({ isRunning: true });
    this.resetOutputs();
    this.stopPoller();

    try {
      const payload = buildClinicalPayload(this.vm.form, this.vm.settings, allowMissingLabs);
      const startResult = await startClinicalJob(payload);
      this.stateService.updateDiliAgent({
        jobId: startResult.job_id,
        jobProgress: 0,
        jobStatus: startResult.status,
        jobStage: 'session_initialization',
        jobStageMessage: 'Initializing clinical session',
      });
      const intervalMs = startResult.poll_interval * 1000;
      this.startPolling(startResult.job_id, intervalMs);
    } catch (error) {
      this.stateService.updateDiliAgent({
        message: formatUnknownError(error, 'Unexpected error'),
        exportUrl: null,
        jobStage: null,
        jobStageMessage: null,
        isRunning: false,
      });
    }
  }

  async runSession(): Promise<void> {
    const form = this.vm.form;
    const missingMessageByField: Array<[keyof typeof form, string]> = [
      ['anamnesis', '[ERROR] Provide the anamnesis.'],
      ['visitDate', '[ERROR] Provide the visit date.'],
      ['drugs', '[ERROR] Provide current drugs.'],
    ];

    const firstMissing = missingMessageByField.find(([field]) => !String(form[field]).trim());
    if (firstMissing) {
      this.resetOutputs();
      this.stateService.updateDiliAgent({
        isRunning: false,
        message: firstMissing[1],
      });
      return;
    }

    if (!form.laboratoryAnalysis.trim()) {
      this.resetOutputs();
      this.isMissingLabsModalOpen.set(true);
      return;
    }

    await this.executeRunSession(null);
  }

  cancelMissingLabs(): void {
    this.isMissingLabsModalOpen.set(false);
  }

  async confirmMissingLabs(): Promise<void> {
    this.isMissingLabsModalOpen.set(false);
    await this.executeRunSession(true);
  }

  async stopSession(): Promise<void> {
    if (!this.vm.jobId) {
      return;
    }
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
    });
  }

  copyReport(): void {
    const message = this.vm.message;
    if (!message) return;
    void navigator.clipboard.writeText(message);
  }

  toggleExpand(): void {
    this.stateService.updateDiliAgent({ isExpanded: !this.vm.isExpanded });
  }

  downloadReport(): void {
    if (!this.vm.exportUrl) return;
    const anchor = document.createElement('a');
    anchor.href = this.vm.exportUrl;
    anchor.download = REPORT_EXPORT_FILENAME;
    anchor.click();
  }

  runOrStop(): void {
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
}
