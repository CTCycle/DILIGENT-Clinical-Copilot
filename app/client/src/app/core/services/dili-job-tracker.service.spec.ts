import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AppStateService } from '../state/app-state.service';
import { DiliJobTrackerService } from './dili-job-tracker.service';

async function flushAsyncWork(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise((resolve) => globalThis.setTimeout(resolve, 0));
}

describe('DiliJobTrackerService', () => {
  let appState: AppStateService;

  beforeEach(async () => {
    vi.restoreAllMocks();
    localStorage.clear();
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:report-url');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    await TestBed.configureTestingModule({}).compileComponents();
    appState = TestBed.inject(AppStateService);
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('does not bootstrap polling from an in-memory active job', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockRejectedValue(
      new Error('unexpected polling request'),
    );
    appState.updateDiliAgent({
      jobId: 'job-cancelling',
      jobStatus: 'running',
      isStarting: false,
      isRunning: true,
      jobStartedAtMs: Date.now() - 5_000,
      jobLastProgressAtMs: Date.now() - 5_000,
      pollIntervalMs: 250,
    });

    TestBed.inject(DiliJobTrackerService);
    await flushAsyncWork();

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(appState.state().diliAgent.jobId).toBe('job-cancelling');
    expect(appState.state().diliAgent.isRunning).toBeTruthy();
  });

  it('keeps a stop-requested worker active while waiting for shutdown', () => {
    const tracker = TestBed.inject(DiliJobTrackerService);
    appState.updateDiliAgent({
      jobId: 'job-cancelling',
      jobStatus: 'running',
      isStarting: false,
      isRunning: true,
      jobStartedAtMs: Date.now() - 5_000,
      jobLastProgressAtMs: Date.now() - 5_000,
      pollIntervalMs: 250,
    });

    const applyJobStatus = (tracker as unknown as {
      applyJobStatus: (status: {
        job_id: string;
        job_type: 'clinical';
        status: 'running';
        progress: number;
        result: null;
        error: null;
        stop_requested: boolean;
        version: number;
      }) => void;
    }).applyJobStatus.bind(tracker);
    applyJobStatus({
      job_id: 'job-cancelling',
      job_type: 'clinical',
      status: 'running',
      stop_requested: true,
      progress: 48,
      result: null,
      error: null,
      version: 4,
    });

    expect(appState.state().diliAgent.jobStatus).toBe('running');
    expect(appState.state().diliAgent.isRunning).toBeTruthy();
    expect(appState.state().diliAgent.message).toContain('Waiting for worker shutdown');

    tracker.clearJobState();
  });
});
