// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { InspectionUpdateJobResource, InspectionUpdateTargetActionsMap } from './inspection-update-job-resource';
import { JobPollingService } from '../services/job-polling.service';
import { InspectionUpdateAllStartRequestMap, InspectionUpdateJobStatusResponse } from '../models/inspection-types';
import { InspectionUpdateJobTrackerService } from './inspection-update-job-tracker.service';

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
      terminology: {
        fetchConfig: vi.fn(),
        start: vi.fn(),
        status: vi.fn(),
        cancel: vi.fn(),
        refresh: vi.fn(),
      },
      dilirank: {
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
      terminology: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
      dilirank: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
      rag: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
    } as unknown as InspectionUpdateTargetActionsMap;
    const tracker = {
      configureRefreshers: vi.fn(),
      targetState: signal({
        rxnav: { jobId: null, running: false, progress: 0, message: '', error: null },
        livertox: { jobId: null, running: false, progress: 0, message: '', error: null },
        terminology: { jobId: null, running: false, progress: 0, message: '', error: null },
        dilirank: { jobId: null, running: false, progress: 0, message: '', error: null },
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

  it('passes DILIrank refresh options through the shared tracker', async () => {
    const actions = {
      rxnav: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
      livertox: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
      terminology: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
      dilirank: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
      rag: { fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn() },
    } as unknown as InspectionUpdateTargetActionsMap;
    const tracker = {
      configureRefreshers: vi.fn(),
      discover: vi.fn().mockResolvedValue(undefined),
      targetState: signal({
        rxnav: { jobId: null, running: false, progress: 0, message: '', error: null },
        livertox: { jobId: null, running: false, progress: 0, message: '', error: null },
        terminology: { jobId: null, running: false, progress: 0, message: '', error: null },
        dilirank: { jobId: null, running: false, progress: 0, message: '', error: null },
        rag: { jobId: null, running: false, progress: 0, message: '', error: null },
      }),
      start: vi.fn().mockResolvedValue(undefined),
    };
    const resource = TestBed.runInInjectionContext(() => new InspectionUpdateJobResource(
      {} as JobPollingService,
      actions,
      () => '',
      tracker as never,
    ));

    await resource.open('dilirank');
    resource.setConfigValue('redownload', true);
    await resource.start();

    expect(tracker.start).toHaveBeenCalledWith({
      target: 'dilirank',
      payload: { redownload: true },
    });
  });

  it('loads and submits the four structured configurations without including RAG', async () => {
    const config = (target: 'livertox' | 'rxnav' | 'terminology' | 'dilirank', defaults: Record<string, unknown>) => ({
      target,
      defaults,
      allowed_fields: Object.keys(defaults),
      summary: {},
      read_only: false,
    });
    const actions = {
      rxnav: {
        fetchConfig: vi.fn().mockResolvedValue(config('rxnav', { force_full: false })),
        start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn(),
      },
      livertox: {
        fetchConfig: vi.fn().mockResolvedValue(config('livertox', {
          livertox_monograph_max_workers: 4,
          livertox_archive: 'livertox.tar.gz',
          redownload: false,
        })),
        start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn(),
      },
      terminology: {
        fetchConfig: vi.fn().mockResolvedValue(config('terminology', { force_full: false, chembl_enabled: true })),
        start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn(),
      },
      dilirank: {
        fetchConfig: vi.fn().mockResolvedValue(config('dilirank', { redownload: false })),
        start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn(),
      },
      rag: {
        fetchConfig: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), refresh: vi.fn(),
      },
    } as unknown as InspectionUpdateTargetActionsMap;
    const tracker = {
      configureRefreshers: vi.fn(),
      discover: vi.fn().mockResolvedValue(undefined),
      targetState: signal({
        rxnav: { jobId: null, running: false, progress: 0, message: '', error: null },
        livertox: { jobId: null, running: false, progress: 0, message: '', error: null },
        terminology: { jobId: null, running: false, progress: 0, message: '', error: null },
        dilirank: { jobId: null, running: false, progress: 0, message: '', error: null },
        rag: { jobId: null, running: false, progress: 0, message: '', error: null },
      }),
      updateAllState: signal({
        runId: null,
        phase: 'idle',
        progress: 0,
        message: '',
        cancelRequested: false,
        targets: {
          livertox: { started: false, jobId: null, status: null, progress: 0, message: '', error: null },
          rxnav: { started: false, jobId: null, status: null, progress: 0, message: '', error: null },
          terminology: { started: false, jobId: null, status: null, progress: 0, message: '', error: null },
          dilirank: { started: false, jobId: null, status: null, progress: 0, message: '', error: null },
        },
      }),
      start: vi.fn(),
      startAll: vi.fn().mockResolvedValue(undefined),
      cancelAll: vi.fn(),
    };
    const resource = TestBed.runInInjectionContext(() => new InspectionUpdateJobResource(
      {} as JobPollingService,
      actions,
      () => '',
      tracker as never,
    ));

    await resource.openAll();
    resource.setConfigValueForTarget('livertox', 'redownload', true);
    await resource.startAll();

    expect(tracker.discover).toHaveBeenCalled();
    expect(actions.rag.fetchConfig).not.toHaveBeenCalled();
    expect(resource.updateAllConfigs().livertox?.['redownload']).toBe(true);
    expect(tracker.startAll).toHaveBeenCalledWith({
      livertox: {
        target: 'livertox',
        payload: {
          livertox_monograph_max_workers: 4,
          livertox_archive: 'livertox.tar.gz',
          redownload: true,
        },
      },
      rxnav: {
        target: 'rxnav',
        payload: { force_full: false },
      },
      terminology: {
        target: 'terminology',
        payload: { force_full: false, chembl_enabled: true },
      },
      dilirank: {
        target: 'dilirank',
        payload: { redownload: false },
      },
    });
  });
});

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
      terminology: { target: 'terminology', payload: {} },
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
    expect(tracker.updateAllState().message).toBe('0 of 4 source updates completed.');
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
      terminology: {
        status: 'completed',
        progress: 100,
        message: 'Terminology complete',
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
    expect(state.progress).toBe(75);
    expect(state.targets.dilirank.status).toBe('failed');
    expect(state.targets.dilirank.error).toContain('DILIrank unavailable');
    expect(state.message).toContain('One or more sources failed');
  });

  it('marks a known job stale when backend recovery makes it unavailable', async () => {
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || 'GET';
      if (url.endsWith('/inspection/jobs')) return jsonResponse({ jobs: [] });
      if (method === 'POST') {
        return jsonResponse({
          job_id: 'rxnav-lost',
          job_type: 'rxnav_update',
          status: 'pending',
          message: 'RxNav queued',
          poll_interval: 1,
        });
      }
      if (url.endsWith('/inspection/rxnav/jobs/rxnav-lost')) {
        return jsonResponse({ detail: 'Job not found after backend restart.' }, 404);
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });

    const tracker = TestBed.inject(InspectionUpdateJobTrackerService);
    await tracker.start({ target: 'rxnav', payload: {} });
    const pollStep = (polling.run.mock.calls[0]?.[0] as { pollStep: () => Promise<boolean> })
      .pollStep;

    expect(await pollStep()).toBe(false);
    expect(tracker.targetState().rxnav.running).toBe(false);
    expect(tracker.targetState().rxnav.status).toBe('failed');
    expect(tracker.targetState().rxnav.message).toContain('lost after backend recovery');
  });

  it('retries transient polling failures and accepts eventual completion', async () => {
    let statusCalls = 0;
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || 'GET';
      if (url.endsWith('/inspection/jobs')) return jsonResponse({ jobs: [] });
      if (method === 'POST') {
        return jsonResponse({
          job_id: 'rxnav-recovery',
          job_type: 'rxnav_update',
          status: 'pending',
          message: 'RxNav queued',
          poll_interval: 1,
        });
      }
      if (url.endsWith('/inspection/rxnav/jobs/rxnav-recovery')) {
        statusCalls += 1;
        if (statusCalls < 3) throw new Error('temporary network outage');
        return jsonResponse({
          job_id: 'rxnav-recovery',
          job_type: 'rxnav_update',
          status: 'completed',
          progress: 100,
          result: { progress_message: 'RxNav complete' },
          error: null,
          version: 3,
        });
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });

    const tracker = TestBed.inject(InspectionUpdateJobTrackerService);
    await tracker.start({ target: 'rxnav', payload: {} });
    const pollStep = (polling.run.mock.calls[0]?.[0] as { pollStep: () => Promise<boolean> })
      .pollStep;

    expect(await pollStep()).toBe(true);
    expect(await pollStep()).toBe(true);
    expect(await pollStep()).toBe(false);
    expect(statusCalls).toBe(3);
    expect(tracker.targetState().rxnav.status).toBe('completed');
    expect(tracker.targetState().rxnav.running).toBe(false);
    expect(tracker.targetState().rxnav.progress).toBe(100);
  });
});
