import { Injectable, inject, signal } from '@angular/core';

import {
  InspectionUpdateJobStatusResponse,
  InspectionUpdateStartRequest,
  InspectionUpdateTarget,
  JobStartResponse,
  JobStatus,
} from '../models/types';
import {
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRagUpdateJob,
  cancelInspectionRxNavUpdateJob,
  fetchInspectionLiverToxUpdateJobStatus,
  fetchInspectionRagUpdateJobStatus,
  fetchInspectionRxNavUpdateJobStatus,
  fetchInspectionUpdateJobs,
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
const JOB_TYPES: Record<InspectionUpdateTarget, string> = {
  rxnav: 'rxnav_update',
  livertox: 'livertox_update',
  rag: 'rag_update',
};

function initialState(): InspectionUpdateTargetState {
  return { jobId: null, status: null, running: false, progress: 0, message: '', error: null, version: -1 };
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
    rxnav: initialState(), livertox: initialState(), rag: initialState(),
  });
  private readonly pollTokens = new Map<InspectionUpdateTarget, number>();
  private readonly refreshedJobKeys = new Set<string>();
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

  async cancel(target: InspectionUpdateTarget): Promise<void> {
    const jobId = this.targetState()[target].jobId;
    if (!jobId) return;
    await this.cancelRequest(target, jobId);
  }

  private async startRequest(request: InspectionUpdateStartRequest): Promise<JobStartResponse> {
    if (request.target === 'rxnav') return startInspectionRxNavUpdateJob(request.payload);
    if (request.target === 'livertox') return startInspectionLiverToxUpdateJob(request.payload);
    return startInspectionRagUpdateJob(request.payload);
  }

  private async cancelRequest(target: InspectionUpdateTarget, jobId: string): Promise<void> {
    if (target === 'rxnav') await cancelInspectionRxNavUpdateJob(jobId);
    else if (target === 'livertox') await cancelInspectionLiverToxUpdateJob(jobId);
    else await cancelInspectionRagUpdateJob(jobId);
    this.patch(target, { message: 'Cancellation requested.' });
  }

  private statusRequest(target: InspectionUpdateTarget, jobId: string): Promise<InspectionUpdateJobStatusResponse> {
    if (target === 'rxnav') return fetchInspectionRxNavUpdateJobStatus(jobId);
    if (target === 'livertox') return fetchInspectionLiverToxUpdateJobStatus(jobId);
    return fetchInspectionRagUpdateJobStatus(jobId);
  }

  private applyStarted(target: InspectionUpdateTarget, started: JobStartResponse): void {
    this.patch(target, {
      jobId: started.job_id, status: started.status, running: !TERMINAL.has(started.status),
      progress: 0, message: started.message || 'Update running.', error: null, version: -1,
    });
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
  }
}
