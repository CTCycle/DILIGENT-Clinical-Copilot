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

  it('reattaches an active persisted job on bootstrap and refreshes progress', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          job_id: 'job-123',
          job_type: 'clinical',
          status: 'running',
          progress: 48,
          result: {
            progress_stage: 'retrieval.evidence',
            progress_message: 'Gathering evidence',
          },
          error: null,
          version: 3,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    );

    appState.updateDiliAgent({
      jobId: 'job-123',
      jobStatus: 'running',
      jobProgress: 12,
      jobStage: 'drugs.extracting',
      jobStageMessage: 'Extracting drugs',
      isStarting: false,
      isRunning: true,
      jobStartedAtMs: Date.now() - 5_000,
      jobLastProgressAtMs: Date.now() - 5_000,
      pollIntervalMs: 250,
    });

    const tracker = TestBed.inject(DiliJobTrackerService);
    await flushAsyncWork();

    expect(fetchSpy).toHaveBeenCalled();
    expect(appState.state().diliAgent.jobId).toBe('job-123');
    expect(appState.state().diliAgent.jobProgress).toBe(48);
    expect(appState.state().diliAgent.jobStage).toBe('retrieval.evidence');
    expect(appState.state().diliAgent.jobStageMessage).toBe('Gathering evidence');
    expect(appState.state().diliAgent.isRunning).toBeTruthy();

    tracker.clearJobState();
  });

  it('clears stale persisted job linkage when the backend no longer has the job', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({ detail: 'Job not found.' }),
        {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    );

    appState.updateDiliAgent({
      jobId: 'job-stale',
      jobStatus: 'running',
      isStarting: false,
      isRunning: true,
      jobStartedAtMs: Date.now() - 5_000,
      jobLastProgressAtMs: Date.now() - 5_000,
      pollIntervalMs: 250,
    });

    TestBed.inject(DiliJobTrackerService);
    await flushAsyncWork();

    expect(appState.state().diliAgent.jobId).toBeNull();
    expect(appState.state().diliAgent.isRunning).toBeFalsy();
    expect(appState.state().diliAgent.message).toContain('no longer available');
  });

  it('hydrates the final report and export url after reattaching to a completed job', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          job_id: 'job-456',
          job_type: 'clinical',
          status: 'completed',
          progress: 100,
          result: {
            report: 'Recovered final report',
            progress_stage: 'completed',
            progress_message: 'Clinical analysis completed.',
          },
          error: null,
          version: 7,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    );

    appState.updateDiliAgent({
      jobId: 'job-456',
      jobStatus: 'running',
      isStarting: false,
      isRunning: true,
      jobStartedAtMs: Date.now() - 5_000,
      jobLastProgressAtMs: Date.now() - 5_000,
      pollIntervalMs: 250,
    });

    const tracker = TestBed.inject(DiliJobTrackerService);
    await flushAsyncWork();

    expect(appState.state().diliAgent.jobStatus).toBe('completed');
    expect(appState.state().diliAgent.isRunning).toBeFalsy();
    expect(appState.state().diliAgent.message).toContain('Recovered final report');
    expect(appState.state().diliAgent.exportUrl).toBe('blob:report-url');

    tracker.clearJobState();
  });

  it('keeps a stop-requested worker active while waiting for shutdown', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          job_id: 'job-cancelling',
          job_type: 'clinical',
          status: 'running',
          stop_requested: true,
          progress: 48,
          result: null,
          error: null,
          version: 4,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
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

    const tracker = TestBed.inject(DiliJobTrackerService);
    await flushAsyncWork();

    expect(appState.state().diliAgent.jobStatus).toBe('running');
    expect(appState.state().diliAgent.isRunning).toBeTruthy();
    expect(appState.state().diliAgent.message).toContain('Waiting for worker shutdown');

    tracker.clearJobState();
  });
});
