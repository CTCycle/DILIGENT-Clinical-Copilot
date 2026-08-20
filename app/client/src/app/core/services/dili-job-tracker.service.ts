import { Injectable, inject } from '@angular/core';

import { REPORT_EXPORT_FILENAME } from '../constants';
import { ClinicalJobResult, JobStatus, JobStatusResponse } from '../models/types';
import { AppStateService } from '../state/app-state.service';
import {
  cancelClinicalJob,
  fetchClinicalJobStatus,
  resolvePollIntervalMs,
  startClinicalJob,
} from './clinical-api';
import { JobPollingService } from './job-polling.service';
import { normalizeThrownError } from './http-api';
import { createDownloadUrl, formatErrorMessage, formatUnknownError } from '../utils';

const POLL_WATCHDOG_MIN_STALE_MS = 15_000;
const STALE_JOB_REATTACH_MESSAGE =
  '[WARN] The previous background analysis is no longer available. Start a new run to continue.';

function isTerminalJobStatus(status: JobStatus | null): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled';
}

function isJobNotFoundError(message: string): boolean {
  return message.toLowerCase().includes('not found');
}

@Injectable({ providedIn: 'root' })
export class DiliJobTrackerService {
  private readonly stateService = inject(AppStateService);
  private readonly jobPolling = inject(JobPollingService);

  private pollToken = 0;
  private latestVersion = -1;
  private lastPollResponseTimestamp = 0;
  private lastProgressSignature = '';
  private pollRecoveryInFlight = false;
  private pollWatchdogTimer: ReturnType<typeof globalThis.setInterval> | null = null;

  constructor() {
    void this.reattachPersistedJob();
  }

  async startSession(
    payload: Parameters<typeof startClinicalJob>[0],
    preflightWarningSummary: string | null,
  ): Promise<void> {
    const vm = this.stateService.state().diliAgent;
    if (vm.isStarting || vm.isRunning) {
      return;
    }

    this.stopPolling();
    this.revokeCurrentExportUrl();
    const now = Date.now();
    this.stateService.updateDiliAgent({
      message: '',
      exportUrl: null,
      jobId: null,
      jobProgress: 0,
      jobStatus: null,
      jobStage: null,
      jobStageMessage: null,
      isStarting: true,
      isRunning: true,
      jobStartedAtMs: now,
      jobLastProgressAtMs: now,
      pollIntervalMs: null,
    });

    try {
      const startResult = await startClinicalJob(payload);
      const intervalMs = resolvePollIntervalMs(startResult.poll_interval);
      this.stateService.updateDiliAgent({
        jobId: startResult.job_id,
        jobProgress: 0,
        jobStatus: startResult.status,
        jobStage: 'session_initialization',
        jobStageMessage: preflightWarningSummary ?? 'Starting clinical analysis',
        isStarting: false,
        isRunning: true,
        jobStartedAtMs: now,
        jobLastProgressAtMs: now,
        pollIntervalMs: intervalMs,
      });
      this.beginPolling(startResult.job_id, intervalMs);
    } catch (error) {
      this.stateService.updateDiliAgent({
        message: formatUnknownError(error, 'Unexpected error'),
        exportUrl: null,
        jobId: null,
        jobProgress: 0,
        jobStatus: null,
        jobStage: null,
        jobStageMessage: null,
        isStarting: false,
        isRunning: false,
        jobStartedAtMs: null,
        jobLastProgressAtMs: null,
        pollIntervalMs: null,
      });
    }
  }

  async cancelSession(): Promise<void> {
    const vm = this.stateService.state().diliAgent;
    if (!vm.jobId) {
      return;
    }

    await cancelClinicalJob(vm.jobId);
    this.stateService.updateDiliAgent({
      message: '[INFO] Cancellation requested. Waiting for worker shutdown...',
    });
  }

  clearJobState(): void {
    this.stopPolling();
    this.revokeCurrentExportUrl();
    this.stateService.updateDiliAgent({
      message: '',
      exportUrl: null,
      jobId: null,
      jobProgress: 0,
      jobStatus: null,
      jobStage: null,
      jobStageMessage: null,
      isStarting: false,
      isRunning: false,
      jobStartedAtMs: null,
      jobLastProgressAtMs: null,
      pollIntervalMs: null,
    });
  }

  revokeCurrentExportUrl(): void {
    const currentExportUrl = this.stateService.state().diliAgent.exportUrl;
    if (currentExportUrl) {
      URL.revokeObjectURL(currentExportUrl);
    }
  }

  private async reattachPersistedJob(): Promise<void> {
    const vm = this.stateService.state().diliAgent;
    if (!vm.jobId || isTerminalJobStatus(vm.jobStatus) || (!vm.isRunning && !vm.isStarting)) {
      return;
    }

    const intervalMs = Math.max(vm.pollIntervalMs ?? 1000, 250);
    this.stateService.updateDiliAgent({
      isStarting: false,
      isRunning: true,
      pollIntervalMs: intervalMs,
      jobStartedAtMs: vm.jobStartedAtMs ?? Date.now(),
      jobLastProgressAtMs: vm.jobLastProgressAtMs ?? Date.now(),
    });

    try {
      const status = await fetchClinicalJobStatus(
        vm.jobId,
        `reattach-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      );
      this.applyJobStatus(status);
      if (!isTerminalJobStatus(status.status) && this.stateService.state().diliAgent.jobId === vm.jobId) {
        this.beginPolling(vm.jobId, intervalMs, true);
      }
    } catch (error) {
      const message = normalizeThrownError(
        error,
        '[ERROR] Polling could not continue. Please retry.',
      );
      if (isJobNotFoundError(message)) {
        this.stopPolling();
        this.stateService.updateDiliAgent({
          message: STALE_JOB_REATTACH_MESSAGE,
          exportUrl: null,
          jobId: null,
          jobProgress: 0,
          jobStatus: null,
          jobStage: null,
          jobStageMessage: null,
          isStarting: false,
          isRunning: false,
          jobStartedAtMs: null,
          jobLastProgressAtMs: null,
          pollIntervalMs: null,
        });
        return;
      }
      this.handlePollingError(message);
    }
  }

  private beginPolling(jobId: string, intervalMs: number, delayFirstPoll = false): void {
    this.stopPolling();
    this.latestVersion = -1;
    this.lastPollResponseTimestamp = Date.now();
    const pollToken = ++this.pollToken;
    let shouldDelayFirstPoll = delayFirstPoll;
    void this.jobPolling.run({
      intervalMs,
      isCancelled: () => pollToken !== this.pollToken,
      pollStep: async () => {
        if (shouldDelayFirstPoll) {
          shouldDelayFirstPoll = false;
          await new Promise((resolve) => globalThis.setTimeout(resolve, Math.max(intervalMs, 250)));
          if (pollToken !== this.pollToken) {
            return false;
          }
        }
        try {
          const status = await fetchClinicalJobStatus(
            jobId,
            `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          );
          if (pollToken !== this.pollToken) {
            return false;
          }
          this.lastPollResponseTimestamp = Date.now();
          const incomingVersion =
            typeof status.version === 'number' && Number.isFinite(status.version)
              ? status.version
              : -1;
          if (incomingVersion >= 0 && incomingVersion < this.latestVersion) {
            return true;
          }
          if (incomingVersion >= 0) {
            this.latestVersion = incomingVersion;
          }
          this.applyJobStatus(status);
          return !isTerminalJobStatus(status.status);
        } catch (error) {
          if (pollToken !== this.pollToken) {
            return false;
          }
          this.handlePollingError(
            normalizeThrownError(
              error,
              '[ERROR] Polling could not continue. Please retry.',
            ),
          );
          return false;
        }
      },
    });
    this.schedulePollWatchdog(intervalMs);
  }

  private stopPolling(): void {
    this.pollToken += 1;
    this.pollRecoveryInFlight = false;
    this.latestVersion = -1;
    if (this.pollWatchdogTimer !== null) {
      globalThis.clearInterval(this.pollWatchdogTimer);
      this.pollWatchdogTimer = null;
    }
  }

  private applyJobStatus(status: JobStatusResponse<ClinicalJobResult>): void {
    const terminalStatus = isTerminalJobStatus(status.status);
    const stage =
      status.result && typeof status.result.progress_stage === 'string'
        ? status.result.progress_stage
        : null;
    const stageMessage =
      status.result && typeof status.result.progress_message === 'string'
        ? status.result.progress_message
        : null;
    const currentAgentState = this.stateService.state().diliAgent;
    const resolvedStage = terminalStatus ? status.status : stage ?? currentAgentState.jobStage;
    const resolvedStageMessage = terminalStatus
      ? this.resolveTerminalStageMessage(status)
      : stageMessage ?? currentAgentState.jobStageMessage;
    const resolvedProgress = typeof status.progress === 'number' ? status.progress : 0;
    const now = Date.now();
    const progressSignature = [
      status.status,
      resolvedProgress.toFixed(2),
      resolvedStage ?? '',
      resolvedStageMessage ?? '',
    ].join('|');
    const lastProgressAtMs =
      this.lastProgressSignature !== progressSignature
        ? now
        : currentAgentState.jobLastProgressAtMs;
    this.lastProgressSignature = progressSignature;

    this.stateService.updateDiliAgent({
      jobProgress: terminalStatus ? 100 : resolvedProgress,
      jobStatus: status.status,
      jobStage: resolvedStage,
      jobStageMessage: resolvedStageMessage,
      isStarting: false,
      isRunning: !terminalStatus,
      jobStartedAtMs: currentAgentState.jobStartedAtMs ?? now,
      jobLastProgressAtMs: lastProgressAtMs ?? now,
      ...(status.stop_requested && !terminalStatus
        ? { message: '[INFO] Cancellation requested. Waiting for worker shutdown...' }
        : {}),
    });

    if (!terminalStatus) {
      return;
    }

    this.stopPolling();

    if (status.status === 'completed') {
      const report = typeof status.result?.report === 'string' ? status.result.report : '';
      this.revokeCurrentExportUrl();
      this.stateService.updateDiliAgent({
        message: report || '[INFO] Clinical analysis completed.',
        exportUrl: report ? createDownloadUrl(report, REPORT_EXPORT_FILENAME) : null,
        pollIntervalMs: null,
      });
      return;
    }

    this.revokeCurrentExportUrl();
    this.stateService.updateDiliAgent({
      message: status.status === 'failed'
        ? (status.error ? formatErrorMessage(status.error) : '[ERROR] Clinical analysis failed.')
        : '[INFO] Clinical analysis cancelled.',
      exportUrl: null,
      pollIntervalMs: null,
    });
  }

  private resolveTerminalStageMessage(status: JobStatusResponse<ClinicalJobResult>): string {
    if (status.status === 'completed') {
      return 'Clinical analysis completed.';
    }
    if (status.status === 'cancelled') {
      return 'Clinical analysis cancelled.';
    }
    return 'Clinical analysis failed.';
  }

  private handlePollingError(message: string): void {
    this.stopPolling();
    this.revokeCurrentExportUrl();
    this.stateService.updateDiliAgent({
      message: isJobNotFoundError(message)
        ? STALE_JOB_REATTACH_MESSAGE
        : message.startsWith('[ERROR]')
          ? message
          : `[ERROR] ${message}`,
      exportUrl: null,
      jobStatus: isJobNotFoundError(message) ? null : 'failed',
      jobId: isJobNotFoundError(message) ? null : this.stateService.state().diliAgent.jobId,
      jobStage: null,
      jobStageMessage: null,
      isStarting: false,
      isRunning: false,
      jobProgress: isJobNotFoundError(message) ? 0 : this.stateService.state().diliAgent.jobProgress,
      jobStartedAtMs: isJobNotFoundError(message) ? null : this.stateService.state().diliAgent.jobStartedAtMs,
      jobLastProgressAtMs: isJobNotFoundError(message) ? null : this.stateService.state().diliAgent.jobLastProgressAtMs,
      pollIntervalMs: null,
    });
  }

  private schedulePollWatchdog(intervalMs: number): void {
    if (this.pollWatchdogTimer !== null) {
      globalThis.clearInterval(this.pollWatchdogTimer);
    }
    this.pollWatchdogTimer = globalThis.setInterval(() => {
      void this.recoverPollingIfStale();
    }, Math.max(5000, intervalMs));
  }

  private async recoverPollingIfStale(): Promise<void> {
    const vm = this.stateService.state().diliAgent;
    if (this.pollRecoveryInFlight || !vm.isRunning || !vm.jobId) {
      return;
    }

    const pollIntervalMs = Math.max(vm.pollIntervalMs ?? 1000, 250);
    const staleThresholdMs = Math.max(POLL_WATCHDOG_MIN_STALE_MS, pollIntervalMs * 6);
    const lastResponseAgeMs = this.lastPollResponseTimestamp > 0
      ? Date.now() - this.lastPollResponseTimestamp
      : staleThresholdMs + 1;
    if (lastResponseAgeMs < staleThresholdMs) {
      return;
    }

    this.pollRecoveryInFlight = true;
    const jobId = vm.jobId;
    try {
      const status = await fetchClinicalJobStatus(
        jobId,
        `watchdog-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      );
      this.lastPollResponseTimestamp = Date.now();
      this.applyJobStatus(status);
      if (!isTerminalJobStatus(status.status) && this.stateService.state().diliAgent.jobId === jobId) {
        this.beginPolling(jobId, pollIntervalMs);
      }
    } catch {
      // Keep the active UI state; the next poll attempt or a later recovery can still succeed.
    } finally {
      this.pollRecoveryInFlight = false;
    }
  }
}
