import { signal } from '@angular/core';

import {
  InspectionLiverToxOverrideRequest,
  InspectionRagOverrideRequest,
  InspectionRxNavOverrideRequest,
  InspectionUpdateConfigResponse,
  InspectionUpdateJobStatusResponse,
  InspectionUpdateTarget,
  JobStartResponse,
} from '../../core/models/types';
import { JobPollingService } from '../../core/services/job-polling.service';

type InspectionUpdateOverridesByTarget = {
  rxnav: InspectionRxNavOverrideRequest;
  livertox: InspectionLiverToxOverrideRequest;
  rag: InspectionRagOverrideRequest;
};

type InspectionUpdateTargetActions<TTarget extends InspectionUpdateTarget> = {
  fetchConfig: () => Promise<InspectionUpdateConfigResponse>;
  start: (overrides: InspectionUpdateOverridesByTarget[TTarget]) => Promise<JobStartResponse>;
  status: (jobId: string) => Promise<InspectionUpdateJobStatusResponse>;
  cancel: (jobId: string) => Promise<void>;
  refresh: () => Promise<void>;
};

type InspectionUpdateTargetState = {
  config: Record<string, unknown> | null;
  configText: string;
  loading: boolean;
  running: boolean;
  jobId: string | null;
  progress: number;
  message: string;
  error: string | null;
  pollToken: number | null;
};

export type InspectionUpdateTargetActionsMap = {
  [TTarget in InspectionUpdateTarget]: InspectionUpdateTargetActions<TTarget>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

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
  const statusValue = readStringKey(payload, 'status') || 'unknown';
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

export class InspectionUpdateJobResource {
  readonly activeTarget = signal<InspectionUpdateTarget | null>(null);
  readonly updateConfig = signal<Record<string, unknown> | null>(null);
  readonly updateConfigText = signal('{}');
  readonly updateLoading = signal(false);
  readonly updateRunning = signal(false);
  readonly updateJobId = signal<string | null>(null);
  readonly updateProgress = signal(0);
  readonly updateMessage = signal('');
  readonly updateError = signal<string | null>(null);

  private updatePollToken = 0;
  private readonly targetStates = new Map<InspectionUpdateTarget, InspectionUpdateTargetState>();

  constructor(
    private readonly jobPolling: JobPollingService,
    private readonly actions: InspectionUpdateTargetActionsMap,
    private readonly getRagDocumentsPath: () => string = () => '',
  ) {}

  dispose(): void {
    this.cancelActivePolling();
  }

  async open(target: InspectionUpdateTarget): Promise<void> {
    this.activeTarget.set(target);
    const state = this.getTargetState(target);
    this.applyStateToSignals(state);

    if (state.config !== null) {
      return;
    }

    this.patchTargetState(target, { loading: true, error: null });
    try {
      const payload = await this.actions[target].fetchConfig();
      const defaults = { ...(payload.defaults ?? undefined) };
      if (target === 'rag' && this.getRagDocumentsPath().trim()) {
        defaults['documents_path'] = this.getRagDocumentsPath().trim();
      }
      this.patchTargetState(target, {
        config: defaults,
        configText: JSON.stringify(defaults, null, 2),
      });
    } catch (error) {
      this.patchTargetState(target, {
        config: {},
        configText: '{}',
        error: error instanceof Error ? error.message : 'Failed to load update configuration.',
      });
    } finally {
      this.patchTargetState(target, { loading: false });
    }
  }

  close(): void {
    this.activeTarget.set(null);
  }

  setConfigText(value: string): void {
    const target = this.activeTarget();
    if (!target) {
      return;
    }
    this.patchTargetState(target, { configText: value });
  }

  async start(): Promise<void> {
    const target = this.activeTarget();
    if (!target) {
      return;
    }
    this.updateError.set(null);
    this.patchTargetState(target, {
      error: null,
      running: true,
      progress: 0,
      message: 'Starting update job...',
    });

    const parsedOverrides = this.parseOverrides(target, this.updateConfigText());
    if (parsedOverrides.error) {
      this.patchTargetState(target, {
        error: parsedOverrides.error,
        running: false,
      });
      return;
    }

    try {
      const started = await this.actions[target].start(parsedOverrides.value);
      const startedJobId = resolveStartJobId(started);
      if (!startedJobId) {
        throw new Error('Update job started but no job id was returned.');
      }
      const pollToken = this.beginPolling();
      this.patchTargetState(target, {
        jobId: startedJobId,
        message: resolveStartedMessage(started),
        pollToken,
      });
      void this.pollUpdateJob(target, startedJobId, pollToken);
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
    try {
      await this.actions[target].cancel(jobId);
      this.patchTargetState(target, { message: 'Cancellation requested.' });
    } catch (error) {
      this.patchTargetState(target, {
        error: error instanceof Error ? error.message : 'Failed to cancel update job.',
      });
    }
  }

  private parseOverrides<TTarget extends InspectionUpdateTarget>(
    target: TTarget,
    raw: string,
  ): {
    value: InspectionUpdateOverridesByTarget[TTarget];
    error: string | null;
  } {
    const normalized = raw.trim();
    if (!normalized) {
      return { value: this.applyTargetDefaults(target, {}), error: null };
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(normalized);
    } catch {
      return { value: this.applyTargetDefaults(target, {}), error: 'Invalid JSON overrides.' };
    }
    if (!isRecord(parsed)) {
      return { value: this.applyTargetDefaults(target, {}), error: 'Overrides must be a JSON object.' };
    }
    return { value: this.applyTargetDefaults(target, parsed), error: null };
  }

  private applyTargetDefaults<TTarget extends InspectionUpdateTarget>(
    target: TTarget,
    overrides: Record<string, unknown>,
  ): InspectionUpdateOverridesByTarget[TTarget] {
    if (target === 'rag') {
      return {
        ...overrides,
        documents_path: this.getRagDocumentsPath().trim() || undefined,
      } as InspectionUpdateOverridesByTarget[TTarget];
    }
    return overrides as InspectionUpdateOverridesByTarget[TTarget];
  }

  private async pollUpdateJob(
    target: InspectionUpdateTarget,
    jobId: string,
    pollToken: number,
  ): Promise<void> {
    try {
      await this.jobPolling.run({
        intervalMs: 1200,
        isCancelled: () => !this.isPollingActive(target, pollToken),
        pollStep: async () => {
          const status = await this.actions[target].status(jobId);
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
      configText: '{}',
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
    if (this.activeTarget() === target) {
      this.applyStateToSignals(next);
    }
    return next;
  }

  private applyStateToSignals(state: InspectionUpdateTargetState): void {
    this.updateConfig.set(state.config);
    this.updateConfigText.set(state.configText);
    this.updateLoading.set(state.loading);
    this.updateRunning.set(state.running);
    this.updateJobId.set(state.jobId);
    this.updateProgress.set(state.progress);
    this.updateMessage.set(state.message);
    this.updateError.set(state.error);
  }
}
