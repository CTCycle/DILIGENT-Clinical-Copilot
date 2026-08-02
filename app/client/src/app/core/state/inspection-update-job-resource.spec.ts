import { describe, expect, it, vi } from 'vitest';
import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import {
  InspectionUpdateJobResource,
  InspectionUpdateTargetActionsMap,
} from './inspection-update-job-resource';
import { JobPollingService } from '../services/job-polling.service';

describe('InspectionUpdateJobResource', () => {
  it('uses the backend poll interval and a bounded request timeout for update jobs', async () => {
    let capturedIntervalMs: number | null = null;
    const statusSpy = vi.fn().mockResolvedValue({
      job_id: 'job-1',
      job_type: 'rxnav_update',
      status: 'completed',
      progress: 100,
      result: {
        progress_message: 'Completed.',
      },
      error: null,
    });

    const jobPolling = {
      run: vi.fn(async (options: { intervalMs: number; pollStep: () => Promise<boolean> }) => {
        capturedIntervalMs = options.intervalMs;
        await options.pollStep();
      }),
    } as unknown as JobPollingService;

    const actions: InspectionUpdateTargetActionsMap = {
      rxnav: {
        fetchConfig: vi.fn(),
        start: vi.fn().mockResolvedValue({
          job_id: 'job-1',
          job_type: 'rxnav_update',
          status: 'pending',
          message: 'Started.',
          poll_interval: 2.5,
        }),
        status: statusSpy,
        cancel: vi.fn(),
        refresh: vi.fn(),
      },
      livertox: {
        fetchConfig: vi.fn(),
        start: vi.fn(),
        status: vi.fn(),
        cancel: vi.fn(),
        refresh: vi.fn(),
      },
      rag: {
        fetchConfig: vi.fn(),
        start: vi.fn(),
        status: vi.fn(),
        cancel: vi.fn(),
        refresh: vi.fn(),
      },
    };

    const resource = new InspectionUpdateJobResource(jobPolling, actions);
    resource.activeTarget.set('rxnav');

    await resource.start();

    expect(capturedIntervalMs).toBe(2500);
    expect(statusSpy).toHaveBeenCalledWith('job-1', 10);
    expect(actions.rxnav.refresh).toHaveBeenCalled();
    expect(resource.updateRunning()).toBe(false);
    expect(resource.updateProgress()).toBe(100);
  });

  it('passes a discriminated start request to the shared tracker', async () => {
    const actions = {
      rxnav: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
      livertox: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
      rag: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
    } as unknown as InspectionUpdateTargetActionsMap;
    const tracker = {
      configureRefreshers: vi.fn(),
      targetState: signal({
        rxnav: { jobId: null, running: false, progress: 0, message: '', error: null },
        livertox: { jobId: null, running: false, progress: 0, message: '', error: null },
        rag: { jobId: null, running: false, progress: 0, message: '', error: null },
      }),
      start: vi.fn().mockResolvedValue(undefined),
    };
    const resource = TestBed.runInInjectionContext(() => new InspectionUpdateJobResource(
      {} as JobPollingService,
      actions,
      () => 'C:\\clinical-documents',
      tracker as never,
    ));
    resource.activeTarget.set('rag');

    await resource.start();

    expect(tracker.start).toHaveBeenCalledWith({
      target: 'rag',
      payload: { documents_path: 'C:\\clinical-documents' },
    });
  });
});
