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
        return jsonResponse({
          job_id: 'job-structured-sources',
          job_type: 'structured_sources_update',
          status: 'pending',
          message: 'structured sources queued',
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

  function applyCombinedSnapshot(
    tracker: InspectionUpdateJobTrackerService,
    sources: NonNullable<InspectionUpdateJobStatusResponse['result']>['sources'],
  ): void {
    const privateTracker = tracker as unknown as {
      applyCombinedSnapshot: (snapshot: InspectionUpdateJobStatusResponse) => boolean;
    };
    privateTracker.applyCombinedSnapshot({
      job_id: 'job-structured-sources',
      job_type: 'structured_sources_update',
      status: sources?.dilirank?.status === 'failed' ? 'failed' : 'running',
      progress: 67,
      result: { sources },
      error: null,
      version: 1,
    });
  }

  it('starts one ordered structured-source job and excludes RAG', async () => {
    const tracker = TestBed.inject(InspectionUpdateJobTrackerService);

    await tracker.startAll(requests());

    const postUrls = fetchMock.mock.calls
      .filter((call) => (call[1] as RequestInit | undefined)?.method === 'POST')
      .map((call) => String(call[0]));
    expect(postUrls).toHaveLength(1);
    expect(postUrls[0]).toContain('/structured-sources/jobs');
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
    expect(deleteUrls).toHaveLength(1);
    expect(deleteUrls[0]).toContain('/structured-sources/jobs/');
    expect(deleteUrls.some((url) => url.includes('/rag/'))).toBe(false);
    expect(tracker.updateAllState().cancelRequested).toBe(true);
  });

  it('derives equal-weight completion progress and preserves partial failures', async () => {
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/inspection/jobs')) return jsonResponse({ jobs: [] });
      if (init?.method === 'POST') {
        return jsonResponse({
          job_id: 'job-structured-sources',
          job_type: 'structured_sources_update',
          status: 'pending',
          message: 'structured sources queued',
          poll_interval: 60,
        });
      }
      throw new Error(`Unexpected request: ${init?.method || 'GET'} ${url}`);
    });

    const tracker = TestBed.inject(InspectionUpdateJobTrackerService);
    await tracker.startAll(requests());

    applyCombinedSnapshot(tracker, {
      livertox: {
        status: 'completed',
        progress: 100,
        message: 'LiverTox complete',
        error: null,
      },
      rxnav: {
        status: 'completed',
        progress: 100,
        message: 'RxNav complete',
        error: null,
      },
      dilirank: {
        status: 'failed',
        progress: 0,
        message: 'DILIrank unavailable',
        error: 'DILIrank unavailable',
      },
    });

    const state = tracker.updateAllState();
    expect(state.phase).toBe('partial_failure');
    expect(state.progress).toBe(67);
    expect(state.targets.dilirank.status).toBe('failed');
    expect(state.targets.dilirank.error).toContain('DILIrank unavailable');
    expect(state.message).toContain('One or more sources failed');
  });
});
