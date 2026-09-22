import { Injectable, inject, signal } from '@angular/core';

import {
  INSPECTION_UPDATE_ALL_TARGETS,
  InspectionUpdateAllStartRequestMap,
  InspectionUpdateAllState,
  InspectionUpdateAllTarget,
  InspectionUpdateAllTargetState,
  InspectionUpdateJobStatusResponse,
  InspectionStructuredSourcesUpdateRequest,
  InspectionUpdateStartRequest,
  InspectionUpdateTarget,
} from '../models/inspection-types';
import { JobStartResponse, JobStatus } from '../models/types';
import {
  cancelInspectionDiliRankUpdateJob,
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRagUpdateJob,
  cancelInspectionRxNavUpdateJob,
  cancelInspectionStructuredSourcesUpdateJob,
  fetchInspectionDiliRankUpdateJobStatus,
  fetchInspectionLiverToxUpdateJobStatus,
  fetchInspectionRagUpdateJobStatus,
  fetchInspectionRxNavUpdateJobStatus,
  fetchInspectionStructuredSourcesUpdateJobStatus,
  fetchInspectionUpdateJobs,
  startInspectionDiliRankUpdateJob,
  startInspectionLiverToxUpdateJob,
  startInspectionRagUpdateJob,
  startInspectionRxNavUpdateJob,
  startInspectionStructuredSourcesUpdateJob,
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
const STRUCTURED_SOURCES_JOB_TYPE = 'structured_sources_update';
const STRUCTURED_SOURCE_JOB_TYPES = new Set([
  'rxnav_update',
  'livertox_update',
  'dilirank_update',
]);
const MAX_POLL_FAILURES = 5;

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
  private combinedPollToken = 0;
  private combinedJobId: string | null = null;
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
      const combinedSnapshot = response.jobs.find(
        (snapshot) => snapshot.job_type === STRUCTURED_SOURCES_JOB_TYPE,
      );
      if (combinedSnapshot) {
        this.applyCombinedSnapshot(combinedSnapshot);
        if (!TERMINAL.has(combinedSnapshot.status)) {
          this.startCombinedPolling(
            combinedSnapshot.job_id,
            1000,
          );
        } else {
          await this.refreshCombinedIfNeeded(combinedSnapshot);
        }
      }
      for (const snapshot of response.jobs) {
        if (snapshot.job_type === STRUCTURED_SOURCES_JOB_TYPE) continue;
        if (combinedSnapshot && STRUCTURED_SOURCE_JOB_TYPES.has(snapshot.job_type)) {
          continue;
        }
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
    if (INSPECTION_UPDATE_ALL_TARGETS.some((target) => this.targetState()[target].running)) {
      throw new Error('A structured source update is already running.');
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
      message: 'Starting the RxNav, LiverTox, and DILIrank update sequence.',
      cancelRequested: false,
      targets: initialTargets,
    });
    await this.discover();
    if (INSPECTION_UPDATE_ALL_TARGETS.some((target) => this.targetState()[target].running)) {
      this.syncUpdateAllState();
      throw new Error('A structured source update is already running.');
    }

    const payload: InspectionStructuredSourcesUpdateRequest = {
      rxnav: requests.rxnav.payload,
      livertox: requests.livertox.payload,
      dilirank: requests.dilirank.payload,
    };
    try {
      const started = await startInspectionStructuredSourcesUpdateJob(payload);
      this.applyCombinedStarted(started, runId);
      this.startCombinedPolling(
        started.job_id,
        resolvePollIntervalMs(started.poll_interval),
      );
    } catch (error) {
      await this.discover();
      if (INSPECTION_UPDATE_ALL_TARGETS.some((target) => this.targetState()[target].running)) {
        return;
      }
      throw error;
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

    const jobId = INSPECTION_UPDATE_ALL_TARGETS
      .map((target) => this.updateAllState().targets[target].jobId)
      .find((candidate): candidate is string => !!candidate);
    if (!jobId) {
      this.syncUpdateAllState();
      return;
    }
    try {
      await cancelInspectionStructuredSourcesUpdateJob(jobId);
    } catch (error) {
      const message = updateErrorMessage(
        error,
        'Failed to request source update cancellation.',
      );
      this.updateAllState.update((state) => ({
        ...state,
        message,
        targets: Object.fromEntries(
          INSPECTION_UPDATE_ALL_TARGETS.map((target) => [
            target,
            {
              ...state.targets[target],
              error: message,
              message,
            },
          ]),
        ) as InspectionUpdateAllState['targets'],
      }));
    }
    this.syncUpdateAllState();
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
  }

  private applyCombinedStarted(started: JobStartResponse, runId: number): void {
    this.combinedJobId = started.job_id;
    const status = started.status as JobStatus;
    this.updateAllState.update((state) => {
      if (state.runId !== runId) return state;
      return {
        ...state,
        targets: Object.fromEntries(
          INSPECTION_UPDATE_ALL_TARGETS.map((target) => [
            target,
            {
              ...state.targets[target],
              started: true,
              jobId: started.job_id,
              status,
              progress: 0,
              message: started.message || 'Update queued.',
              error: null,
            },
          ]),
        ) as InspectionUpdateAllState['targets'],
      };
    });
    this.syncUpdateAllState();
  }

  private applyCombinedSnapshot(
    snapshot: InspectionUpdateJobStatusResponse,
  ): boolean {
    this.ensureUpdateAllRunForCombined();
    this.combinedJobId = snapshot.job_id;
    const sourceSnapshots = snapshot.result?.sources;
    const fallbackStatus = snapshot.status === 'completed'
      ? 'completed'
      : snapshot.status === 'failed'
        ? 'failed'
        : snapshot.status === 'cancelled'
          ? 'cancelled'
          : 'pending';
    this.updateAllState.update((state) => ({
      ...state,
      targets: Object.fromEntries(
        INSPECTION_UPDATE_ALL_TARGETS.map((target) => [
          target,
          {
            ...state.targets[target],
            started: true,
            jobId: snapshot.job_id,
            status: (sourceSnapshots?.[target]?.status || fallbackStatus) as JobStatus,
            error: sourceSnapshots?.[target]?.error || snapshot.error,
          },
        ]),
      ) as InspectionUpdateAllState['targets'],
    }));

    let applied = false;
    for (const target of INSPECTION_UPDATE_ALL_TARGETS) {
      const source = sourceSnapshots?.[target];
      const status = (source?.status || fallbackStatus) as JobStatus;
      const progress = source?.progress ?? (
        status === 'completed' ? 100 : status === 'pending' ? 0 : snapshot.progress
      );
      this.patch(target, {
        jobId: snapshot.job_id,
        status,
        running: !TERMINAL.has(status),
        progress,
        message: source?.message || statusMessage(snapshot),
        error: source?.error || snapshot.error,
        version: typeof snapshot.version === 'number' ? snapshot.version : -1,
      });
      applied = true;
    }
    this.syncUpdateAllState();
    return applied;
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
    let consecutiveFailures = 0;
    let rediscoveryAttempted = false;
    void this.polling.run({
      intervalMs: Math.max(intervalMs, 250),
      isCancelled: () => this.pollTokens.get(target) !== token || this.targetState()[target].jobId !== jobId,
      pollStep: async () => {
        try {
          const snapshot = await this.statusRequest(target, jobId);
          consecutiveFailures = 0;
          if (this.pollTokens.get(target) !== token) return false;
          const applied = this.applySnapshot(target, snapshot);
          if (applied && TERMINAL.has(snapshot.status)) {
            this.pollTokens.delete(target);
            await this.refreshIfNeeded(target, snapshot);
            return false;
          }
          return !TERMINAL.has(snapshot.status);
        } catch (error) {
          if (this.isJobNotFoundError(error)) {
            if (!rediscoveryAttempted) {
              rediscoveryAttempted = true;
              await this.discover();
              if (this.pollTokens.get(target) !== token) return false;
            }
            this.markTargetStale(target, jobId);
            this.pollTokens.delete(target);
            return false;
          }
          consecutiveFailures += 1;
          if (consecutiveFailures >= MAX_POLL_FAILURES) {
            this.markTargetStale(target, jobId);
            this.pollTokens.delete(target);
            return false;
          }
          return true;
        }
      },
    });
  }

  private startCombinedPolling(jobId: string, intervalMs: number): void {
    const token = ++this.combinedPollToken;
    this.combinedJobId = jobId;
    let consecutiveFailures = 0;
    let rediscoveryAttempted = false;
    void this.polling.run({
      intervalMs: Math.max(intervalMs, 250),
      isCancelled: () => this.combinedPollToken !== token || this.combinedJobId !== jobId,
      pollStep: async () => {
        try {
          const snapshot = await fetchInspectionStructuredSourcesUpdateJobStatus(jobId);
          consecutiveFailures = 0;
          if (this.combinedPollToken !== token) return false;
          this.applyCombinedSnapshot(snapshot);
          if (TERMINAL.has(snapshot.status)) {
            this.combinedPollToken += 1;
            await this.refreshCombinedIfNeeded(snapshot);
            return false;
          }
          return true;
        } catch (error) {
          if (this.isJobNotFoundError(error)) {
            if (!rediscoveryAttempted) {
              rediscoveryAttempted = true;
              await this.discover();
              if (this.combinedPollToken !== token || this.combinedJobId !== jobId) {
                return false;
              }
            }
            this.markCombinedStale(jobId);
            this.combinedPollToken += 1;
            return false;
          }
          consecutiveFailures += 1;
          if (consecutiveFailures >= MAX_POLL_FAILURES) {
            this.markCombinedStale(jobId);
            this.combinedPollToken += 1;
            return false;
          }
          return true;
        }
      },
    });
  }

  private isJobNotFoundError(error: unknown): boolean {
    return error instanceof Error && /not found|requested data was not found/i.test(error.message);
  }

  private markTargetStale(target: InspectionUpdateTarget, jobId: string): void {
    const message = 'The update job was lost after backend recovery. Retry the update.';
    if (this.targetState()[target].jobId !== jobId) return;
    this.patch(target, {
      status: 'failed',
      running: false,
      message,
      error: message,
    });
  }

  private markCombinedStale(jobId: string): void {
    const message = 'The structured source update was lost after backend recovery. Retry the update.';
    if (this.combinedJobId !== jobId) return;
    this.updateAllState.update((state) => ({
      ...state,
      message,
      targets: Object.fromEntries(
        INSPECTION_UPDATE_ALL_TARGETS.map((target) => [
          target,
          {
            ...state.targets[target],
            status: 'failed',
            message,
            error: message,
          },
        ]),
      ) as InspectionUpdateAllState['targets'],
    }));
    for (const target of INSPECTION_UPDATE_ALL_TARGETS) {
      if (this.targetState()[target].jobId !== jobId) continue;
      this.patch(target, {
        status: 'failed',
        running: false,
        message,
        error: message,
      });
    }
    this.combinedJobId = null;
    this.syncUpdateAllState();
  }

  private ensureUpdateAllRunForCombined(): void {
    if (this.updateAllState().runId !== null) return;
    const runId = ++this.updateAllRunSequence;
    this.updateAllState.set({
      ...initialUpdateAllState(),
      runId,
      phase: 'starting',
      message: 'Recovering the structured source update status.',
    });
  }

  private async refreshCombinedIfNeeded(
    snapshot: InspectionUpdateJobStatusResponse,
  ): Promise<void> {
    if (snapshot.status !== 'completed') return;
    for (const target of INSPECTION_UPDATE_ALL_TARGETS) {
      const child = snapshot.result?.sources?.[target];
      if (child?.status !== 'completed') continue;
      await this.refreshIfNeeded(target, {
        ...snapshot,
        job_id: snapshot.job_id,
        status: 'completed',
      });
    }
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
          : `${completedCount} of ${targets.length} source updates completed.`;
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

}
