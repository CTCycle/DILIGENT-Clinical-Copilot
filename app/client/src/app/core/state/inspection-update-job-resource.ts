import { computed, effect, signal } from '@angular/core';

import {
  INSPECTION_UPDATE_ALL_TARGETS,
  InspectionUpdateAllConfigErrorMap,
  InspectionUpdateAllConfigLoadingMap,
  InspectionUpdateAllConfigMap,
  InspectionUpdateAllStartRequestMap,
  InspectionUpdateAllState,
  InspectionUpdateAllTarget,
  InspectionUpdateOverridesByTarget,
  InspectionUpdateConfigResponse,
  InspectionUpdateJobStatusResponse,
  InspectionUpdateStartRequest,
  InspectionUpdateTarget,
} from '../models/inspection-types';
import { JobStartResponse } from '../models/types';
import { resolvePollIntervalMs } from '../services/clinical-api';
import { JobPollingService } from '../services/job-polling.service';
import { InspectionUpdateJobTrackerService } from './inspection-update-job-tracker.service';
import { isRecord } from '../utils';

type InspectionUpdateTargetActions<TTarget extends InspectionUpdateTarget> = {
  fetchConfig: () => Promise<InspectionUpdateConfigResponse>;
  start: (overrides: InspectionUpdateOverridesByTarget[TTarget]) => Promise<JobStartResponse>;
  status: (jobId: string, timeoutSeconds?: number) => Promise<InspectionUpdateJobStatusResponse>;
  cancel: (jobId: string) => Promise<void>;
  refresh: () => Promise<void>;
};

type InspectionUpdateTargetState = {
  config: Record<string, unknown> | null;
  configError: string | null;
  loading: boolean;
  running: boolean;
  jobId: string | null;
  progress: number;
  message: string;
  error: string | null;
  pollToken: number | null;
};

export type InspectionUpdateTargetSnapshot = {
  running: boolean;
  progress: number;
  message: string;
  error: string | null;
};

type InspectionUpdateTargetSnapshotMap = Record<
  InspectionUpdateTarget,
  InspectionUpdateTargetSnapshot
>;

export type InspectionUpdateTargetActionsMap = {
  [TTarget in InspectionUpdateTarget]: InspectionUpdateTargetActions<TTarget>;
};

function readStringKey(payload: Record<string, unknown>, ...keys: string[]): string | null {
  for (const key of keys) {
    const value = payload[key];
    if (typeof value === 'string' && value.trim()) {
      return value;
    }
  }
  return null;
}

function readNumberKey(payload: Record<string, unknown>, ...keys: string[]): number | null {
  for (const key of keys) {
    const value = payload[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function readRecordKey(payload: Record<string, unknown>, ...keys: string[]): Record<string, unknown> | null {
  for (const key of keys) {
    const value = payload[key];
    if (isRecord(value)) {
      return value;
    }
  }
  return null;
}

function resolveStartJobId(started: JobStartResponse): string | null {
  const payload = started as unknown;
  if (!isRecord(payload)) {
    return null;
  }
  return readStringKey(payload, 'job_id', 'jobId', 'id');
}

function resolveStatusValue(status: InspectionUpdateJobStatusResponse): string {
  const payload = status as unknown;
  if (!isRecord(payload)) {
    return 'failed';
  }
  return readStringKey(payload, 'status') || 'failed';
}

function resolveUpdateProgressMessage(status: InspectionUpdateJobStatusResponse): string {
  const payload = status as unknown;
  if (!isRecord(payload)) {
    return 'Update progress unavailable.';
  }
  const resultPayload = readRecordKey(payload, 'result');
  const statusValue = readStringKey(payload, 'status') || 'unknown';
  if (resultPayload && statusValue === 'completed') {
    const summary = readRecordKey(resultPayload, 'summary');
    const documents = summary ? readNumberKey(summary, 'documents') : null;
    const chunks = summary ? readNumberKey(summary, 'chunks') : null;
    const supportedFiles = summary ? readNumberKey(summary, 'supported_files') : null;
    if (typeof documents === 'number' && typeof chunks === 'number') {
      const supportedSuffix = typeof supportedFiles === 'number'
        ? ` from ${supportedFiles} supported files`
        : '';
      return `RAG embeddings update completed: ${documents} documents, ${chunks} chunks${supportedSuffix}.`;
    }
  }
  const resultMessage = resultPayload
    ? readStringKey(resultPayload, 'progress_message', 'progressMessage')
    : null;
  if (resultMessage) {
    return resultMessage;
  }
  const errorMessage = readStringKey(payload, 'error');
  if (errorMessage) {
    return errorMessage;
  }
  return `Job status: ${statusValue}`;
}

function resolveProgressValue(status: InspectionUpdateJobStatusResponse): number {
  const payload = status as unknown;
  if (!isRecord(payload)) {
    return 0;
  }
  const directProgress = readNumberKey(payload, 'progress');
  if (typeof directProgress === 'number') {
    return directProgress;
  }
  const resultPayload = readRecordKey(payload, 'result');
  const resultProgress = resultPayload ? readNumberKey(resultPayload, 'progress') : null;
  return typeof resultProgress === 'number' ? resultProgress : 0;
}

function resolveErrorValue(status: InspectionUpdateJobStatusResponse): string | null {
  const payload = status as unknown;
  if (!isRecord(payload)) {
    return null;
  }
  return readStringKey(payload, 'error');
}

function resolveStartedMessage(started: JobStartResponse): string {
  const payload = started as unknown;
  if (!isRecord(payload)) {
    return 'Update running...';
  }
  const message =
    readStringKey(payload, 'message') ||
    readStringKey(payload, 'detail');
  return message || 'Update running...';
}

function resolvePollRequestTimeoutSeconds(intervalMs: number): number {
  return Math.min(30, Math.max(5, Math.ceil((Math.max(intervalMs, 250) / 1000) * 4)));
}

function initialUpdateAllConfigMap(): InspectionUpdateAllConfigMap {
  return {
    livertox: null,
    rxnav: null,
    dilirank: null,
  };
}

function initialUpdateAllConfigLoadingMap(): InspectionUpdateAllConfigLoadingMap {
  return {
    livertox: false,
    rxnav: false,
    dilirank: false,
  };
}

function initialUpdateAllConfigErrorMap(): InspectionUpdateAllConfigErrorMap {
  return {
    livertox: null,
    rxnav: null,
    dilirank: null,
  };
}

function initialUpdateAllState(): InspectionUpdateAllState {
  return {
    runId: null,
    phase: 'idle',
    progress: 0,
    message: '',
    cancelRequested: false,
    targets: {
      livertox: {
        started: false,
        jobId: null,
        status: null,
        progress: 0,
        message: 'Waiting to start.',
        error: null,
      },
      rxnav: {
        started: false,
        jobId: null,
        status: null,
        progress: 0,
        message: 'Waiting to start.',
        error: null,
      },
      dilirank: {
        started: false,
        jobId: null,
        status: null,
        progress: 0,
        message: 'Waiting to start.',
        error: null,
      },
    },
  };
}

function isUpdateAllRunning(state: InspectionUpdateAllState): boolean {
  return state.phase === 'starting' || state.phase === 'running';
}

export class InspectionUpdateJobResource {
  readonly targetState = signal<InspectionUpdateTargetSnapshotMap>({
    rxnav: { running: false, progress: 0, message: '', error: null },
    livertox: { running: false, progress: 0, message: '', error: null },
    dilirank: { running: false, progress: 0, message: '', error: null },
    rag: { running: false, progress: 0, message: '', error: null },
  });
  readonly updateAllState = signal<InspectionUpdateAllState>(initialUpdateAllState());
  readonly updateAllConfigs = signal<InspectionUpdateAllConfigMap>(initialUpdateAllConfigMap());
  readonly updateAllConfigLoading = signal<InspectionUpdateAllConfigLoadingMap>(initialUpdateAllConfigLoadingMap());
  readonly updateAllConfigErrors = signal<InspectionUpdateAllConfigErrorMap>(initialUpdateAllConfigErrorMap());
  readonly updateAllValidationError = signal<string | null>(null);
  readonly activeTarget = signal<InspectionUpdateTarget | null>(null);
  readonly updateConfig = signal<Record<string, unknown> | null>(null);
  readonly updateLoading = signal(false);
  readonly updateRunning = signal(false);
  readonly updateJobId = signal<string | null>(null);
  readonly updateProgress = signal(0);
  readonly updateMessage = signal('');
  readonly updateError = signal<string | null>(null);
  readonly updateAllRunning = computed(() => isUpdateAllRunning(this.updateAllState()));
  readonly updateAllCanStart = computed(() => {
    const configs = this.updateAllConfigs();
    const loading = this.updateAllConfigLoading();
    const errors = this.updateAllConfigErrors();
    const trackerStates = this.tracker?.targetState();
    return INSPECTION_UPDATE_ALL_TARGETS.every(
      (target) => configs[target] !== null && !loading[target] && !errors[target],
    ) && !this.updateAllRunning() && !INSPECTION_UPDATE_ALL_TARGETS.some(
      (target) => trackerStates?.[target].running,
    );
  });

  private updatePollToken = 0;
  private readonly targetStates = new Map<InspectionUpdateTarget, InspectionUpdateTargetState>();

  constructor(
    private readonly jobPolling: JobPollingService,
    private readonly actions: InspectionUpdateTargetActionsMap,
    private readonly getRagDocumentsPath: () => string = () => '',
    private readonly tracker: InspectionUpdateJobTrackerService | null = null,
  ) {
    if (this.tracker) {
      this.tracker.configureRefreshers({
        rxnav: this.actions.rxnav.refresh,
        livertox: this.actions.livertox.refresh,
        dilirank: this.actions.dilirank.refresh,
        rag: this.actions.rag.refresh,
      });
      effect(() => {
        const states = this.tracker!.targetState();
        for (const target of ['rxnav', 'livertox', 'dilirank', 'rag'] as InspectionUpdateTarget[]) {
          const state = states[target];
          this.patchTargetState(target, {
            jobId: state.jobId,
            running: state.running,
            progress: state.progress,
            message: state.message,
            error: state.error,
          });
        }
      });
      const updateAllState = this.tracker.updateAllState;
      if (updateAllState) {
        effect(() => {
          this.updateAllState.set(updateAllState());
        });
      }
    }
  }

  dispose(): void {
    this.cancelActivePolling();
  }

  async open(target: InspectionUpdateTarget): Promise<void> {
    if (this.tracker) {
      await this.tracker.discover();
    }
    this.activeTarget.set(target);
    const state = this.getTargetState(target);
    this.applyStateToSignals(state);
    await this.loadConfig(target, { emptyOnFailure: true });
  }

  async openAll(): Promise<void> {
    if (this.tracker) {
      await this.tracker.discover();
    }
    this.updateAllValidationError.set(null);
    await Promise.all(
      INSPECTION_UPDATE_ALL_TARGETS.map((target) => {
        const state = this.getTargetState(target);
        return this.loadConfig(target, {
          force: state.configError !== null,
          emptyOnFailure: false,
        });
      }),
    );
  }

  private async loadConfig(
    target: InspectionUpdateTarget,
    options: { force?: boolean; emptyOnFailure: boolean },
  ): Promise<void> {
    const current = this.getTargetState(target);
    if (!options.force && current.config !== null) {
      return;
    }
    this.patchTargetState(target, {
      loading: true,
      error: null,
      configError: null,
    });
    try {
      const payload = await this.actions[target].fetchConfig();
      const defaults = { ...(payload.defaults ?? undefined) };
      const summary = isRecord(payload.summary) ? { ...payload.summary } : {};
      this.patchTargetState(target, {
        config: payload.read_only ? summary : defaults,
        configError: null,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load update configuration.';
      this.patchTargetState(target, {
        config: options.emptyOnFailure ? {} : null,
        configError: message,
        error: message,
      });
    } finally {
      this.patchTargetState(target, { loading: false });
    }
  }

  close(): void {
    this.activeTarget.set(null);
  }

  setConfigValue(key: string, value: unknown): void {
    const target = this.activeTarget();
    if (!target) {
      return;
    }
    this.patchTargetState(target, {
      config: {
        ...(this.getTargetState(target).config ?? {}),
        [key]: value,
      },
      configError: null,
      error: null,
    });
  }

  setConfigValueForTarget(target: InspectionUpdateAllTarget, key: string, value: unknown): void {
    this.patchTargetState(target, {
      config: {
        ...(this.getTargetState(target).config ?? {}),
        [key]: value,
      },
      configError: null,
      error: null,
    });
    this.updateAllValidationError.set(null);
  }

  async startAll(): Promise<void> {
    const validationError = this.validateUpdateAllConfiguration();
    if (validationError) {
      this.updateAllValidationError.set(validationError);
      return;
    }
    if (!this.tracker) {
      this.updateAllValidationError.set('Combined source updates are unavailable.');
      return;
    }
    this.updateAllValidationError.set(null);
    await this.tracker.startAll(this.buildUpdateAllStartRequests());
  }

  async cancelAll(): Promise<void> {
    if (!this.tracker) {
      return;
    }
    await this.tracker.cancelAll();
  }

  closeAll(): void {
    this.updateAllValidationError.set(null);
  }

  private validateUpdateAllConfiguration(): string | null {
    const configs = this.updateAllConfigs();
    const loading = this.updateAllConfigLoading();
    const errors = this.updateAllConfigErrors();
    const loadingTarget = INSPECTION_UPDATE_ALL_TARGETS.find((target) => loading[target]);
    if (loadingTarget) {
      return 'Wait for all source configurations to finish loading.';
    }
    const failedTarget = INSPECTION_UPDATE_ALL_TARGETS.find((target) => errors[target]);
    if (failedTarget) {
      return 'Resolve the source configuration errors before starting the combined update.';
    }
    const missingTarget = INSPECTION_UPDATE_ALL_TARGETS.find((target) => configs[target] === null);
    if (missingTarget) {
      return 'Load configuration for all three sources before starting the combined update.';
    }
    const runningTarget = INSPECTION_UPDATE_ALL_TARGETS.find(
      (target) => this.tracker?.targetState()[target].running,
    );
    if (runningTarget) {
      return 'An update is already running for one of the selected sources.';
    }
    return null;
  }

  private buildUpdateAllStartRequests(): InspectionUpdateAllStartRequestMap {
    return {
      livertox: {
        target: 'livertox',
        payload: this.buildStartPayload('livertox'),
      },
      rxnav: {
        target: 'rxnav',
        payload: this.buildStartPayload('rxnav'),
      },
      dilirank: {
        target: 'dilirank',
        payload: this.buildStartPayload('dilirank'),
      },
    };
  }

  async start(): Promise<void> {
    const target = this.activeTarget();
    if (!target) {
      return;
    }
    this.updateError.set(null);
    if (this.tracker) {
      try {
        await this.tracker.start(this.buildStartRequest(target));
      } catch (error) {
        this.patchTargetState(target, {
          running: false,
          error: error instanceof Error ? error.message : 'Failed to start update job.',
        });
      }
      return;
    }
    this.patchTargetState(target, {
      error: null,
      running: true,
      progress: 0,
      message: 'Starting update job...',
    });

    try {
      const started = await this.actions[target].start(this.buildStartPayload(target));
      const startedJobId = resolveStartJobId(started);
      if (!startedJobId) {
        throw new Error('Update job started but no job id was returned.');
      }
      const intervalMs = resolvePollIntervalMs(started.poll_interval);
      const pollToken = this.beginPolling();
      this.patchTargetState(target, {
        jobId: startedJobId,
        message: resolveStartedMessage(started),
        pollToken,
      });
      void this.pollUpdateJob(target, startedJobId, pollToken, intervalMs);
    } catch (error) {
      this.patchTargetState(target, {
        running: false,
        error: error instanceof Error ? error.message : 'Failed to start update job.',
      });
    }
  }

  async cancel(): Promise<void> {
    const target = this.activeTarget();
    const jobId = this.updateJobId();
    if (!target || !jobId) {
      return;
    }
    if (this.tracker) {
      await this.tracker.cancel(target);
      return;
    }
    try {
      await this.actions[target].cancel(jobId);
      this.patchTargetState(target, { message: 'Cancellation requested.' });
    } catch (error) {
      this.patchTargetState(target, {
        error: error instanceof Error ? error.message : 'Failed to cancel update job.',
      });
    }
  }

  private buildStartPayload<TTarget extends InspectionUpdateTarget>(
    target: TTarget,
  ): InspectionUpdateOverridesByTarget[TTarget] {
    if (target === 'rag') {
      const requestedPath = this.getRagDocumentsPath().trim();
      return {
        documents_path: requestedPath || undefined,
      } as InspectionUpdateOverridesByTarget[TTarget];
    }
    return {
      ...(this.getTargetState(target).config ?? {}),
    } as InspectionUpdateOverridesByTarget[TTarget];
  }

  private buildStartRequest(target: InspectionUpdateTarget): InspectionUpdateStartRequest {
    if (target === 'rxnav') {
      return { target, payload: this.buildStartPayload(target) };
    }
    if (target === 'livertox') {
      return { target, payload: this.buildStartPayload(target) };
    }
    if (target === 'dilirank') {
      return { target, payload: this.buildStartPayload(target) };
    }
    return { target, payload: this.buildStartPayload(target) };
  }

  private async pollUpdateJob(
    target: InspectionUpdateTarget,
    jobId: string,
    pollToken: number,
    intervalMs: number,
  ): Promise<void> {
    try {
      const safeIntervalMs = Math.max(intervalMs, 250);
      const requestTimeoutSeconds = resolvePollRequestTimeoutSeconds(safeIntervalMs);
      await this.jobPolling.run({
        intervalMs: safeIntervalMs,
        isCancelled: () => !this.isPollingActive(target, pollToken),
        pollStep: async () => {
          const status = await this.actions[target].status(jobId, requestTimeoutSeconds);
          if (!this.isPollingActive(target, pollToken)) {
            return false;
          }
          const statusValue = resolveStatusValue(status);
          this.patchTargetState(target, {
            progress: resolveProgressValue(status),
            message: resolveUpdateProgressMessage(status),
          });

          if (statusValue === 'completed') {
            this.patchTargetState(target, {
              running: false,
              progress: 100,
              pollToken: null,
            });
            await this.actions[target].refresh();
            return false;
          }
          if (statusValue === 'failed' || statusValue === 'cancelled') {
            this.patchTargetState(target, {
              running: false,
              error: resolveErrorValue(status) || `Update job ${statusValue}.`,
              pollToken: null,
            });
            return false;
          }

          return true;
        },
      });
    } catch (error) {
      if (!this.isPollingActive(target, pollToken)) {
        return;
      }
      this.patchTargetState(target, {
        running: false,
        error: error instanceof Error ? error.message : 'Failed to poll update job.',
        pollToken: null,
      });
    }
  }

  private beginPolling(): number {
    this.updatePollToken += 1;
    return this.updatePollToken;
  }

  private cancelActivePolling(): void {
    if (this.tracker) {
      return;
    }
    this.updatePollToken += 1;
    for (const [target, state] of this.targetStates) {
      if (state.running) {
        this.patchTargetState(target, { running: false, pollToken: null });
      }
    }
  }

  private isPollingActive(target: InspectionUpdateTarget, pollToken: number): boolean {
    return this.getTargetState(target).pollToken === pollToken;
  }

  private getTargetState(target: InspectionUpdateTarget): InspectionUpdateTargetState {
    const existing = this.targetStates.get(target);
    if (existing) {
      return existing;
    }
    const initial: InspectionUpdateTargetState = {
      config: null,
      configError: null,
      loading: false,
      running: false,
      jobId: null,
      progress: 0,
      message: '',
      error: null,
      pollToken: null,
    };
    this.targetStates.set(target, initial);
    return initial;
  }

  private patchTargetState(
    target: InspectionUpdateTarget,
    patch: Partial<InspectionUpdateTargetState>,
  ): InspectionUpdateTargetState {
    const next = {
      ...this.getTargetState(target),
      ...patch,
    };
    this.targetStates.set(target, next);
    if (target === 'livertox' || target === 'rxnav' || target === 'dilirank') {
      this.updateAllConfigs.update((configs) => ({ ...configs, [target]: next.config }));
      this.updateAllConfigLoading.update((loading) => ({ ...loading, [target]: next.loading }));
      this.updateAllConfigErrors.update((errors) => ({ ...errors, [target]: next.configError }));
    }
    this.targetState.update((current) => ({
      ...current,
      [target]: {
        running: next.running,
        progress: next.progress,
        message: next.message,
        error: next.error || next.configError,
      },
    }));
    if (this.activeTarget() === target) {
      this.applyStateToSignals(next);
    }
    return next;
  }

  private applyStateToSignals(state: InspectionUpdateTargetState): void {
    this.updateConfig.set(state.config);
    this.updateLoading.set(state.loading);
    this.updateRunning.set(state.running);
    this.updateJobId.set(state.jobId);
    this.updateProgress.set(state.progress);
    this.updateMessage.set(state.message);
    this.updateError.set(state.error || state.configError);
  }
}
