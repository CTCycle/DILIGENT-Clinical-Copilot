import { describe, expect, it, vi } from 'vitest';

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
});
