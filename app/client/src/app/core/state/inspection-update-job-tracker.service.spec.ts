import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  InspectionUpdateAllStartRequestMap,
  InspectionUpdateJobStatusResponse,
} from '../models/inspection-types';
import { JobPollingService } from '../services/job-polling.service';
import { InspectionUpdateJobTrackerService } from './inspection-update-job-tracker.service';

describe('InspectionUpdateJobTrackerService combined updates', () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  let polling: { run: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    polling = { run: vi.fn().mockResolvedValue(undefined) };
    fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || 'GET';
      if (url.endsWith('/inspection/jobs')) {
        return jsonResponse({ jobs: [] });
      }
      if (method === 'POST') {
        const target = url.split('/').at(-2) || 'source';
        return jsonResponse({
          job_id: `job-${target}`,
          job_type: `${target}_update`,
          status: 'pending',
          message: `${target} queued`,
          poll_interval: 60,
        });
      }
      if (method === 'DELETE') {
        return jsonResponse({ success: true, message: 'Cancellation requested.' });
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    await TestBed.configureTestingModule({
      providers: [{ provide: JobPollingService, useValue: polling }],
    }).compileComponents();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function jsonResponse(payload: unknown, status = 200): Response {
    return new Response(JSON.stringify(payload), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  function requests(): InspectionUpdateAllStartRequestMap {
    return {
      livertox: { target: 'livertox', payload: {} },
      rxnav: { target: 'rxnav', payload: {} },
      dilirank: { target: 'dilirank', payload: {} },
    };
  }

  function applySnapshot(
    tracker: InspectionUpdateJobTrackerService,
    target: 'livertox' | 'rxnav' | 'dilirank',
    jobId: string,
    status: 'completed' | 'failed' | 'cancelled',
    progress: number,
  ): void {
    const privateTracker = tracker as unknown as {
      applySnapshot: (target: string, snapshot: InspectionUpdateJobStatusResponse) => boolean;
    };
    privateTracker.applySnapshot(target, {
      job_id: jobId,
      job_type: 'rxnav_update',
      status,
      progress,
      result: { progress_message: status },
      error: status === 'failed' ? 'Source failed.' : null,
      version: 1,
    });
  }

  it('starts the three existing source jobs in parallel and excludes RAG', async () => {
    const tracker = TestBed.inject(InspectionUpdateJobTrackerService);

    await tracker.startAll(requests());

    const postUrls = fetchMock.mock.calls
      .filter((call) => (call[1] as RequestInit | undefined)?.method === 'POST')
      .map((call) => String(call[0]));
    expect(postUrls).toHaveLength(3);
    expect(postUrls.some((url) => url.includes('/livertox/jobs'))).toBe(true);
    expect(postUrls.some((url) => url.includes('/rxnav/jobs'))).toBe(true);
    expect(postUrls.some((url) => url.includes('/dilirank/jobs'))).toBe(true);
    expect(postUrls.some((url) => url.includes('/rag/jobs'))).toBe(false);
    expect(tracker.updateAllState().phase).toBe('running');
    expect(tracker.updateAllState().progress).toBe(0);
    expect(tracker.updateAllState().targets.livertox.started).toBe(true);
    expect(tracker.updateAllState().targets.rxnav.started).toBe(true);
    expect(tracker.updateAllState().targets.dilirank.started).toBe(true);

    await tracker.cancelAll();

    const deleteUrls = fetchMock.mock.calls
      .filter((call) => (call[1] as RequestInit | undefined)?.method === 'DELETE')
      .map((call) => String(call[0]));
    expect(deleteUrls).toHaveLength(3);
    expect(deleteUrls.some((url) => url.includes('/rag/'))).toBe(false);
    expect(tracker.updateAllState().cancelRequested).toBe(true);
  });

  it('derives equal-weight completion progress and preserves partial failures', async () => {
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/inspection/jobs')) return jsonResponse({ jobs: [] });
      if (url.includes('/dilirank/jobs') && init?.method === 'POST') {
        return jsonResponse({ detail: 'DILIrank unavailable' }, 422);
      }
      if (init?.method === 'POST') {
        const target = url.split('/').at(-2) || 'source';
        return jsonResponse({
          job_id: `job-${target}`,
          job_type: `${target}_update`,
          status: 'pending',
          message: `${target} queued`,
          poll_interval: 60,
        });
      }
      throw new Error(`Unexpected request: ${init?.method || 'GET'} ${url}`);
    });

    const tracker = TestBed.inject(InspectionUpdateJobTrackerService);
    await tracker.startAll(requests());

    applySnapshot(tracker, 'livertox', 'job-livertox', 'completed', 100);
    applySnapshot(tracker, 'rxnav', 'job-rxnav', 'completed', 100);

    const state = tracker.updateAllState();
    expect(state.phase).toBe('partial_failure');
    expect(state.progress).toBe(67);
    expect(state.targets.dilirank.status).toBe('failed');
    expect(state.targets.dilirank.error).toContain('DILIrank unavailable');
    expect(state.message).toContain('One or more sources failed');
  });
});
