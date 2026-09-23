import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ClinicalSessionDetail, InspectionSessionTimelinePreview } from '../../../core/models/inspection-types';
import { JobPollingService } from '../../../core/services/job-polling.service';
import { ClinicalSessionTimelineWorkspaceComponent } from './clinical-session-timeline-workspace.component';

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => { resolve = next; });
  return { promise, resolve };
}

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

function session(sessionId: number): ClinicalSessionDetail {
  return {
    session_id: sessionId,
    patient_name: `Patient ${sessionId}`,
    visit_date: '2026-09-17',
    session_timestamp: '2026-09-17T08:00:00Z',
    version: 1,
    original_session_id: null,
    status: 'successful',
    text_extraction_model: 'parser',
    clinical_model: 'clinical',
    metadata: {},
    sections: {},
    session_text: '',
    source_clinical_text: '',
    result_payload: {},
    report: null,
    official_report_text: null,
    manual_edit_history: [],
  };
}

function preview(sessionId: number, timelineId: number): InspectionSessionTimelinePreview {
  return {
    timeline_id: timelineId,
    session_id: sessionId,
    generated_at: `2026-09-${timelineId}T08:00:00Z`,
    generation_status: 'llm_generated',
    generation_note: null,
    source_model: 'timeline-model',
    source_kind: 'local',
    model_provider: 'ollama',
    event_count: 0,
    start_date: null,
    end_date: null,
    title: null,
    source_evidence_event_count: 0,
    missing_evidence_event_count: 0,
    uncertain_event_count: 0,
    undated_event_count: 0,
  };
}

describe('ClinicalSessionTimelineWorkspaceComponent request generations', () => {
  let fixture: ComponentFixture<ClinicalSessionTimelineWorkspaceComponent>;
  let component: ClinicalSessionTimelineWorkspaceComponent;

  function attachActiveJob(): { pollStep: () => Promise<boolean> } {
    component.ngOnInit = () => undefined;
    fixture.componentRef.setInput('session', session(1));
    fixture.detectChanges();
    component.generationJobId.set('job-1');
    component.generationRunning.set(true);
    component.generationStatus.set('Generating timeline…');

    let pollStep: (() => Promise<boolean>) | undefined;
    vi.spyOn(TestBed.inject(JobPollingService), 'run').mockImplementation(async (options) => {
      pollStep = options.pollStep;
    });
    const attachToTimelineJob = (component as unknown as {
      attachToTimelineJob: (
        jobId: string,
        pollIntervalSeconds: number,
        sessionId: number,
        loadGeneration: number,
      ) => void;
    }).attachToTimelineJob.bind(component);
    const loadGeneration = (component as unknown as { timelineLoadGeneration: number }).timelineLoadGeneration;
    attachToTimelineJob('job-1', 1, 1, loadGeneration);
    fixture.detectChanges();
    return {
      get pollStep() {
        if (!pollStep) throw new Error('Timeline job polling was not attached.');
        return pollStep;
      },
    };
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ClinicalSessionTimelineWorkspaceComponent],
      providers: [provideRouter([])],
    }).compileComponents();
    fixture = TestBed.createComponent(ClinicalSessionTimelineWorkspaceComponent);
    component = fixture.componentInstance;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    fixture.destroy();
  });

  it('keeps the newest timeline history when session responses complete out of order', async () => {
    const firstResponse = deferred<Response>();
    const secondResponse = deferred<Response>();
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/sessions/1/timelines')) return firstResponse.promise;
      if (url.endsWith('/sessions/2/timelines')) return secondResponse.promise;
      throw new Error(`Unexpected request: ${url}`);
    });

    component.session = session(1);
    (component as unknown as { timelineLoadGeneration: number }).timelineLoadGeneration = 1;
    const firstLoad = component.loadTimelineHistory(1, 1);
    await Promise.resolve();

    component.session = session(2);
    (component as unknown as { timelineLoadGeneration: number }).timelineLoadGeneration = 2;
    const secondLoad = component.loadTimelineHistory(2, 2);
    await Promise.resolve();

    secondResponse.resolve(jsonResponse({ items: [preview(2, 22)] }));
    await secondLoad;
    firstResponse.resolve(jsonResponse({ items: [preview(1, 11)] }));
    await firstLoad;

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(component.timelinePreviews().map((item) => item.timeline_id)).toEqual([22]);
    expect(component.timelineListLoading()).toBe(false);
    expect(component.timelineListError()).toBeNull();
  });

  it('ignores a late timeline polling result after the session generation changes', async () => {
    const statusResponse = deferred<Response>();
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/sessions/1/timeline-jobs/job-1')) return statusResponse.promise;
      throw new Error(`Unexpected request: ${url}`);
    });
    const pollingService = TestBed.inject(JobPollingService);
    let pollRun: Promise<void> | undefined;
    vi.spyOn(pollingService, 'run').mockImplementation(async (options) => {
      pollRun = options.pollStep().then(() => undefined);
      await pollRun;
    });

    component.session = session(1);
    (component as unknown as { timelineLoadGeneration: number }).timelineLoadGeneration = 1;
    const attachToTimelineJob = (component as unknown as {
      attachToTimelineJob: (
        jobId: string,
        pollIntervalSeconds: number,
        sessionId: number,
        loadGeneration: number,
      ) => void;
    }).attachToTimelineJob.bind(component);
    attachToTimelineJob('job-1', 1, 1, 1);
    expect(pollRun).toBeDefined();

    component.session = session(2);
    (component as unknown as { timelineLoadGeneration: number }).timelineLoadGeneration = 2;
    component.generationStatus.set('New session is active.');
    component.generationRunning.set(false);
    statusResponse.resolve(jsonResponse({
      status: 'completed',
      progress: 100,
      result: { timeline_id: 11 },
      error: null,
    }));
    await pollRun;

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(component.generationStatus()).toBe('New session is active.');
    expect(component.generationRunning()).toBe(false);
  });

  it('shows Stop, requests cancellation, and waits for the cancelled job state', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      if (init?.method === 'DELETE') {
        return jsonResponse({ job_id: 'job-1', success: true, message: 'Cancellation requested' });
      }
      if (String(input).includes('/sessions/1/timeline-jobs/job-1')) {
        return jsonResponse({
          job_id: 'job-1',
          job_type: 'session_timeline',
          status: 'cancelled',
          progress: 30,
          result: null,
          error: null,
        });
      }
      if (String(input).includes('/sessions/1/timelines')) return jsonResponse({ items: [] });
      return jsonResponse({
        items: [],
      });
    });
    const { pollStep } = attachActiveJob();
    const stopButton = fixture.nativeElement.querySelector(
      'button[aria-label="Stop timeline generation"]',
    ) as HTMLButtonElement | null;

    expect(stopButton).not.toBeNull();
    stopButton?.click();
    stopButton?.click();
    await fixture.whenStable();
    fixture.detectChanges();

    expect(fetchSpy.mock.calls.filter(([, init]) => init?.method === 'DELETE')).toHaveLength(1);
    expect(component.generationCancellationState()).toBe('requested');
    expect(fixture.nativeElement.textContent).toContain('Stopping…');
    expect(fixture.nativeElement.querySelector('button[aria-label="Stop timeline generation"]')).toBeNull();

    await pollStep();
    fixture.detectChanges();

    expect(component.generationRunning()).toBe(false);
    expect(component.generationJobId()).toBeNull();
    expect(component.generationStatus()).toBe('Timeline generation cancelled.');
    expect(component.generationError()).toBeNull();
  });

  it('reconciles a cancellation race when the job completed before DELETE arrived', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      if (init?.method === 'DELETE') {
        return new Response(JSON.stringify({ detail: 'Job not found.' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (String(input).includes('/sessions/1/timelines')) {
        return jsonResponse({ items: [preview(1, 11)] });
      }
      return jsonResponse({
        job_id: 'job-1',
        job_type: 'session_timeline',
        status: 'completed',
        progress: 100,
        result: { progress_message: 'Timeline saved.' },
        error: null,
      });
    });
    attachActiveJob();
    const stopButton = fixture.nativeElement.querySelector(
      'button[aria-label="Stop timeline generation"]',
    ) as HTMLButtonElement | null;

    stopButton?.click();
    await fixture.whenStable();
    fixture.detectChanges();

    expect(fetchSpy.mock.calls.filter(([, init]) => init?.method === 'DELETE')).toHaveLength(1);
    expect(component.generationJobId()).toBeNull();
    expect(component.generationStatus()).toBe('Timeline generated and saved.');
    expect(component.generationError()).toBeNull();
  });

  it('keeps polling and re-enables Stop after a cancellation request fails', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      if (init?.method === 'DELETE') {
        return new Response(JSON.stringify({ detail: 'Service unavailable.' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (String(input).includes('/sessions/1/timeline-jobs/job-1')) {
        return jsonResponse({
          job_id: 'job-1',
          job_type: 'session_timeline',
          status: 'running',
          progress: 30,
          result: null,
          error: null,
        });
      }
      if (String(input).includes('/sessions/1/timelines')) return jsonResponse({ items: [] });
      return jsonResponse({
        job_id: 'job-1',
        job_type: 'session_timeline',
        status: 'running',
        progress: 30,
        result: null,
        error: null,
      });
    });
    const { pollStep } = attachActiveJob();
    const stopButton = fixture.nativeElement.querySelector(
      'button[aria-label="Stop timeline generation"]',
    ) as HTMLButtonElement | null;

    stopButton?.click();
    await fixture.whenStable();
    fixture.detectChanges();

    expect(component.generationRunning()).toBe(true);
    expect(component.generationJobId()).toBe('job-1');
    expect(component.generationCancellationState()).toBe('idle');
    expect(component.generationError()).toBeTruthy();
    expect(fixture.nativeElement.querySelector('button[aria-label="Stop timeline generation"]')).not.toBeNull();

    await pollStep();
    expect(component.generationRunning()).toBe(true);
  });
});
