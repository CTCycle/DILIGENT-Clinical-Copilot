import { Injectable, inject, signal } from '@angular/core';

import {
  INSPECTION_UPDATE_ALL_TARGETS,
  InspectionUpdateAllStartRequestMap,
  InspectionUpdateAllState,
  InspectionUpdateAllTarget,
  InspectionUpdateAllTargetState,
  InspectionUpdateJobStatusResponse,
  InspectionUpdateStartRequest,
  InspectionUpdateTarget,
} from '../models/inspection-types';
import { JobStartResponse, JobStatus } from '../models/types';
import {
  cancelInspectionDiliRankUpdateJob,
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRagUpdateJob,
  cancelInspectionRxNavUpdateJob,
  fetchInspectionDiliRankUpdateJobStatus,
  fetchInspectionLiverToxUpdateJobStatus,
  fetchInspectionRagUpdateJobStatus,
  fetchInspectionRxNavUpdateJobStatus,
  fetchInspectionUpdateJobs,
  startInspectionDiliRankUpdateJob,
  startInspectionLiverToxUpdateJob,
  startInspectionRagUpdateJob,
  startInspectionRxNavUpdateJob,
} from '../services/inspection-jobs-api';
import { resolvePollIntervalMs } from '../services/clinical-api';
import { JobPollingService } from '../services/job-polling.service';

export type InspectionUpdateTargetState = {
  jobId: string | null;
  status: JobStatus | null;
  running: boolean;
  progress: number;
  message: string;
  error: string | null;
  version: number;
};

export type InspectionUpdateTargetStateMap = Record<InspectionUpdateTarget, InspectionUpdateTargetState>;

const TERMINAL = new Set<JobStatus>(['completed', 'failed', 'cancelled']);
const UPDATE_ALL_ACTIVE_PHASES = new Set<InspectionUpdateAllState['phase']>(['starting', 'running']);
const JOB_TYPES: Record<InspectionUpdateTarget, string> = {
  rxnav: 'rxnav_update',
  livertox: 'livertox_update',
  dilirank: 'dilirank_update',
  rag: 'rag_update',
};

function initialState(): InspectionUpdateTargetState {
  return { jobId: null, status: null, running: false, progress: 0, message: '', error: null, version: -1 };
}

function initialUpdateAllTargetState(): InspectionUpdateAllTargetState {
  return {
    started: false,
    jobId: null,
    status: null,
    progress: 0,
    message: 'Waiting to start.',
    error: null,
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
      livertox: initialUpdateAllTargetState(),
      rxnav: initialUpdateAllTargetState(),
      dilirank: initialUpdateAllTargetState(),
    },
  };
}

function clampProgress(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, value));
}

function updateErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback;
}

function statusMessage(status: InspectionUpdateJobStatusResponse): string {
  const result = status.result;
  if (result?.progress_message) return result.progress_message;
  if (status.error) return status.error;
  if (status.status === 'completed') return 'Update completed.';
  if (status.status === 'failed') return 'Update failed.';
  if (status.status === 'cancelled') return 'Update cancelled.';
  return 'Update running...';
}

@Injectable({ providedIn: 'root' })
export class InspectionUpdateJobTrackerService {
  private readonly polling = inject(JobPollingService);
  readonly targetState = signal<InspectionUpdateTargetStateMap>({
    rxnav: initialState(),
    livertox: initialState(),
    dilirank: initialState(),
    rag: initialState(),
  });
  readonly updateAllState = signal<InspectionUpdateAllState>(initialUpdateAllState());
  private readonly pollTokens = new Map<InspectionUpdateTarget, number>();
  private readonly refreshedJobKeys = new Set<string>();
  private readonly updateAllStartRunByTarget = new Map<InspectionUpdateAllTarget, number>();
  private updateAllRunSequence = 0;
  private refreshers: Partial<Record<InspectionUpdateTarget, () => Promise<void>>> = {};

  constructor() {
    void this.discover();
  }

  configureRefreshers(refreshers: Partial<Record<InspectionUpdateTarget, () => Promise<void>>>): void {
    this.refreshers = refreshers;
  }

  async discover(): Promise<void> {
    try {
      const response = await fetchInspectionUpdateJobs();
      for (const snapshot of response.jobs) {
        const target = (Object.keys(JOB_TYPES) as InspectionUpdateTarget[])
          .find((candidate) => JOB_TYPES[candidate] === snapshot.job_type);
        if (!target || !this.applySnapshot(target, snapshot)) continue;
        if (!TERMINAL.has(snapshot.status)) {
          this.startPolling(target, snapshot.job_id, 1000);
        } else {
          await this.refreshIfNeeded(target, snapshot);
        }
      }
    } catch {
      // Discovery is best effort; a later Data Inspection open retries it.
    }
  }

  async start(request: InspectionUpdateStartRequest): Promise<void> {
    try {
      const started = await this.startRequest(request);
      this.applyStarted(request.target, started);
      this.startPolling(request.target, started.job_id, resolvePollIntervalMs(started.poll_interval));
    } catch (error) {
      await this.discover();
      if (this.targetState()[request.target].running) return;
      throw error;
    }
  }

  async startAll(requests: InspectionUpdateAllStartRequestMap): Promise<void> {
    const current = this.updateAllState();
    if (UPDATE_ALL_ACTIVE_PHASES.has(current.phase)) {
      throw new Error('All source updates are already running.');
    }

    const runId = ++this.updateAllRunSequence;
    const initialTargets = {
      livertox: initialUpdateAllTargetState(),
      rxnav: initialUpdateAllTargetState(),
      dilirank: initialUpdateAllTargetState(),
    } satisfies Record<InspectionUpdateAllTarget, InspectionUpdateAllTargetState>;
    this.updateAllState.set({
      runId,
      phase: 'starting',
      progress: 0,
      message: 'Starting LiverTox, RxNav, and DILIrank updates.',
      cancelRequested: false,
      targets: initialTargets,
    });
    await this.discover();

    const startTargets = INSPECTION_UPDATE_ALL_TARGETS.filter((target) => {
      if (!this.targetState()[target].running) return true;
      this.markUpdateAllFailure(target, runId, 'An update is already running for this source.');
      return false;
    });

    for (const target of startTargets) {
      this.updateAllStartRunByTarget.set(target, runId);
    }

    try {
      const results = await Promise.allSettled(
        startTargets.map((target) => this.startAllTarget(target, requests[target], runId)),
      );
      results.forEach((result, index) => {
        const target = startTargets[index];
        if (!target || result.status !== 'rejected') return;
        this.markUpdateAllFailure(
          target,
          runId,
          updateErrorMessage(result.reason, 'Failed to start update job.'),
          true,
        );
      });
      for (const target of startTargets) {
        const targetState = this.updateAllState().targets[target];
        if (!targetState.started && !targetState.error) {
          this.markUpdateAllFailure(target, runId, 'The source update did not start.', true);
        }
      }
      this.syncUpdateAllState();
    } finally {
      for (const target of startTargets) {
        if (this.updateAllStartRunByTarget.get(target) === runId) {
          this.updateAllStartRunByTarget.delete(target);
        }
      }
    }
  }

  async cancel(target: InspectionUpdateTarget): Promise<void> {
    const jobId = this.targetState()[target].jobId;
    if (!jobId) return;
    await this.cancelRequest(target, jobId);
  }

  async cancelAll(): Promise<void> {
    const current = this.updateAllState();
    if (!current.runId || !UPDATE_ALL_ACTIVE_PHASES.has(current.phase)) return;

    this.updateAllState.update((state) => ({
      ...state,
      cancelRequested: true,
      message: 'Cancellation requested for the source updates.',
    }));

    const cancellableTargets = INSPECTION_UPDATE_ALL_TARGETS.filter((target) => {
      const targetState = this.updateAllState().targets[target];
      return targetState.started && this.targetState()[target].running;
    });
    const results = await Promise.allSettled(
      cancellableTargets.map(async (target) => {
        await this.cancel(target);
      }),
    );
    results.forEach((result, index) => {
      const target = cancellableTargets[index];
      if (!target || result.status !== 'rejected') return;
      this.markUpdateAllFailure(
        target,
        current.runId!,
        updateErrorMessage(result.reason, 'Failed to request source update cancellation.'),
        true,
      );
    });
    this.syncUpdateAllState();
  }

  private async startAllTarget<TTarget extends InspectionUpdateAllTarget>(
    target: TTarget,
    request: InspectionUpdateAllStartRequestMap[TTarget],
    runId: number,
  ): Promise<void> {
    await this.start(request);
    if (!this.isCurrentUpdateAllRun(runId)) return;
    if (this.updateAllState().cancelRequested && this.targetState()[target].running) {
      await this.cancel(target);
    }
  }

  private async startRequest(request: InspectionUpdateStartRequest): Promise<JobStartResponse> {
    if (request.target === 'rxnav') return startInspectionRxNavUpdateJob(request.payload);
    if (request.target === 'livertox') return startInspectionLiverToxUpdateJob(request.payload);
    if (request.target === 'dilirank') return startInspectionDiliRankUpdateJob(request.payload);
    return startInspectionRagUpdateJob(request.payload);
  }

  private async cancelRequest(target: InspectionUpdateTarget, jobId: string): Promise<void> {
    if (target === 'rxnav') await cancelInspectionRxNavUpdateJob(jobId);
    else if (target === 'livertox') await cancelInspectionLiverToxUpdateJob(jobId);
    else if (target === 'dilirank') await cancelInspectionDiliRankUpdateJob(jobId);
    else await cancelInspectionRagUpdateJob(jobId);
    this.patch(target, { message: 'Cancellation requested.' });
  }

  private statusRequest(target: InspectionUpdateTarget, jobId: string): Promise<InspectionUpdateJobStatusResponse> {
    if (target === 'rxnav') return fetchInspectionRxNavUpdateJobStatus(jobId);
    if (target === 'livertox') return fetchInspectionLiverToxUpdateJobStatus(jobId);
    if (target === 'dilirank') return fetchInspectionDiliRankUpdateJobStatus(jobId);
    return fetchInspectionRagUpdateJobStatus(jobId);
  }

  private applyStarted(target: InspectionUpdateTarget, started: JobStartResponse): void {
    this.patch(target, {
      jobId: started.job_id, status: started.status, running: !TERMINAL.has(started.status),
      progress: 0, message: started.message || 'Update running.', error: null, version: -1,
    });
    if (target === 'rag') return;
    const runId = this.updateAllStartRunByTarget.get(target);
    if (runId !== undefined) {
      this.markUpdateAllStarted(target, runId);
    }
  }

  private applySnapshot(target: InspectionUpdateTarget, snapshot: InspectionUpdateJobStatusResponse): boolean {
    const current = this.targetState()[target];
    const version = typeof snapshot.version === 'number' ? snapshot.version : -1;
    if (current.jobId === snapshot.job_id && version >= 0 && version < current.version) return false;
    this.patch(target, {
      jobId: snapshot.job_id, status: snapshot.status, running: !TERMINAL.has(snapshot.status),
      progress: snapshot.progress, message: statusMessage(snapshot), error: snapshot.error,
      version,
    });
    return true;
  }

  private startPolling(target: InspectionUpdateTarget, jobId: string, intervalMs: number): void {
    const token = (this.pollTokens.get(target) ?? 0) + 1;
    this.pollTokens.set(target, token);
    void this.polling.run({
      intervalMs: Math.max(intervalMs, 250),
      isCancelled: () => this.pollTokens.get(target) !== token || this.targetState()[target].jobId !== jobId,
      pollStep: async () => {
        try {
          const snapshot = await this.statusRequest(target, jobId);
          if (this.pollTokens.get(target) !== token) return false;
          const applied = this.applySnapshot(target, snapshot);
          if (applied && TERMINAL.has(snapshot.status)) {
            this.pollTokens.delete(target);
            await this.refreshIfNeeded(target, snapshot);
            return false;
          }
          return !TERMINAL.has(snapshot.status);
        } catch {
          // A transient poll failure must not fabricate a terminal backend state.
          return true;
        }
      },
    });
  }

  private async refreshIfNeeded(target: InspectionUpdateTarget, snapshot: InspectionUpdateJobStatusResponse): Promise<void> {
    if (snapshot.status !== 'completed') return;
    const key = `${target}:${snapshot.job_id}:${snapshot.version ?? -1}`;
    if (this.refreshedJobKeys.has(key)) return;
    this.refreshedJobKeys.add(key);
    await this.refreshers[target]?.();
  }

  private patch(target: InspectionUpdateTarget, patch: Partial<InspectionUpdateTargetState>): void {
    const current = this.targetState()[target];
    this.targetState.update((states) => ({ ...states, [target]: { ...current, ...patch } }));
    this.syncUpdateAllTarget(target);
  }

  private markUpdateAllStarted(target: InspectionUpdateAllTarget, runId: number): void {
    if (!this.isCurrentUpdateAllRun(runId)) return;
    const sourceState = this.targetState()[target];
    this.updateAllState.update((state) => {
      if (state.runId !== runId) return state;
      return {
        ...state,
        targets: {
          ...state.targets,
          [target]: {
            ...state.targets[target],
            started: true,
            jobId: sourceState.jobId,
            status: sourceState.status,
            progress: clampProgress(sourceState.progress),
            message: sourceState.message || 'Update running.',
            error: sourceState.error,
          },
        },
      };
    });
    this.syncUpdateAllState();
  }

  private markUpdateAllFailure(
    target: InspectionUpdateAllTarget,
    runId: number,
    message: string,
    preserveActiveJob = false,
  ): void {
    if (!this.isCurrentUpdateAllRun(runId)) return;
    const sourceState = this.targetState()[target];
    const keepActiveJob = preserveActiveJob && sourceState.running && !!sourceState.jobId;
    this.updateAllState.update((state) => {
      if (state.runId !== runId) return state;
      return {
        ...state,
        targets: {
          ...state.targets,
          [target]: {
            ...state.targets[target],
            started: keepActiveJob,
            jobId: keepActiveJob ? sourceState.jobId : null,
            status: keepActiveJob ? sourceState.status : 'failed',
            progress: keepActiveJob ? sourceState.progress : 0,
            message,
            error: message,
          },
        },
      };
    });
    this.syncUpdateAllState();
  }

  private syncUpdateAllTarget(target: InspectionUpdateTarget): void {
    if (target === 'rag') return;
    const group = this.updateAllState();
    if (!group.runId || !UPDATE_ALL_ACTIVE_PHASES.has(group.phase)) return;
    const groupTarget = group.targets[target];
    if (!groupTarget.started || groupTarget.status === 'failed') return;
    const sourceState = this.targetState()[target];
    if (sourceState.jobId !== groupTarget.jobId) return;
    this.updateAllState.update((state) => {
      if (state.runId !== group.runId) return state;
      return {
        ...state,
        targets: {
          ...state.targets,
          [target]: {
            ...state.targets[target],
            status: sourceState.status,
            progress: clampProgress(sourceState.status === 'completed' ? 100 : sourceState.progress),
            message: sourceState.message,
            error: state.targets[target].error || sourceState.error,
          },
        },
      };
    });
    this.syncUpdateAllState();
  }

  private syncUpdateAllState(): void {
    const current = this.updateAllState();
    if (!current.runId) return;
    const targets = INSPECTION_UPDATE_ALL_TARGETS.map((target) => current.targets[target]);
    const allTerminal = targets.every((target) => target.status !== null && TERMINAL.has(target.status));
    const anyFailed = targets.some((target) => target.status === 'failed' || !!target.error);
    const anyCancelled = targets.some((target) => target.status === 'cancelled');
    const completedCount = targets.filter((target) => target.status === 'completed').length;
    const startedCount = targets.filter((target) => target.started).length;
    const progress = Math.round(
      targets.reduce(
        (total, target) => total + clampProgress(target.status === 'completed' ? 100 : target.progress),
        0,
      ) / targets.length,
    );

    let phase: InspectionUpdateAllState['phase'];
    let message: string;
    if (!allTerminal) {
      phase = startedCount < targets.length ? 'starting' : 'running';
      message = current.cancelRequested
        ? 'Cancellation requested for the source updates.'
        : phase === 'starting'
          ? 'Starting source updates...'
          : `${completedCount} of ${targets.length} source updates are running or queued.`;
    } else if (anyFailed) {
      phase = 'partial_failure';
      message = `${completedCount} of ${targets.length} source updates completed. One or more sources failed.`;
    } else if (anyCancelled) {
      phase = 'cancelled';
      message = `${completedCount} of ${targets.length} source updates completed. The operation was cancelled.`;
    } else {
      phase = 'completed';
      message = 'All source updates completed.';
    }

    if (
      current.phase === phase &&
      current.progress === progress &&
      current.message === message
    ) {
      return;
    }
    this.updateAllState.update((state) => ({ ...state, phase, progress, message }));
  }

  private isCurrentUpdateAllRun(runId: number): boolean {
    return this.updateAllState().runId === runId;
  }
}
