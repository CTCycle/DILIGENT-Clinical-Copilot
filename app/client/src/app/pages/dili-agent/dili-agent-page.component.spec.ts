import { ComponentFixture, TestBed } from '@angular/core/testing';
import { vi } from 'vitest';

import { DiliAgentPageComponent } from './dili-agent-page.component';

describe('DiliAgentPageComponent', () => {
  let fixture: ComponentFixture<DiliAgentPageComponent>;
  let component: DiliAgentPageComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [DiliAgentPageComponent] }).compileComponents();
    fixture = TestBed.createComponent(DiliAgentPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  it('canStartSession false when visit date missing', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, visitDate: '', clinicalInput: 'word '.repeat(60) },
    });
    expect(component.canStartSession()).toBeFalsy();
  });

  it('canStartSession false when input too short', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, visitDate: '2025-01-01', clinicalInput: 'word '.repeat(59) },
    });
    expect(component.canStartSession()).toBeFalsy();
  });

  it('canStartSession true when all preconditions are met', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, visitDate: '2025-01-01', clinicalInput: 'word '.repeat(60) },
      settings: { ...component.vm.settings, provider: 'openai' },
    });
    expect(component.canStartSession()).toBeTruthy();
  });

  it('allows an active run to be stopped even while the start click debounce is still active', () => {
    const stopSessionSpy = vi.spyOn(component, 'stopSession').mockResolvedValue();
    component.stateService.updateDiliAgent({
      isRunning: true,
      jobId: 'job-123',
    });
    (component as unknown as { runControlDebounced: boolean }).runControlDebounced = true;

    component.runOrStop();

    expect(stopSessionSpy).toHaveBeenCalled();
  });

  it('tracks acknowledgement for preflight review warnings', () => {
    component.preflightReviewMessages.set([
      'Review extracted timeline section before relying on generated output.',
    ]);

    expect(component.preflightReviewNoticeVisible()).toBeTruthy();
    expect(component.preflightReviewAcknowledged()).toBeFalsy();

    component.acknowledgePreflightReview();

    expect(component.preflightReviewAcknowledged()).toBeTruthy();
  });

  it('stops the spinner and displays a failed clinical job error', () => {
    component.stateService.updateDiliAgent({
      isRunning: true,
      jobStatus: 'running',
      jobId: 'job-123',
    });

    (component as unknown as {
      onJobStatusUpdate: (status: {
        job_id: string;
        job_type: string;
        status: 'failed';
        progress: number;
        result: null;
        error: string;
      }) => void;
    }).onJobStatusUpdate({
      job_id: 'job-123',
      job_type: 'clinical',
      status: 'failed',
      progress: 23,
      result: null,
      error: 'Local model could not be loaded due to insufficient memory.',
    });

    expect(component.showSpinner).toBeFalsy();
    expect(component.vm.isRunning).toBeFalsy();
    expect(component.vm.jobStatus).toBe('failed');
    expect(component.vm.message).toContain('Local model could not be loaded');
  });

  it('marks polling failure as failed and stops the spinner', () => {
    component.stateService.updateDiliAgent({
      isRunning: true,
      jobStatus: 'running',
      jobId: 'job-123',
    });

    (component as unknown as { onPollingError: (message: string) => void }).onPollingError(
      'Polling failed after repeated attempts.',
    );

    expect(component.showSpinner).toBeFalsy();
    expect(component.vm.isRunning).toBeFalsy();
    expect(component.vm.jobStatus).toBe('failed');
    expect(component.vm.message).toContain('Polling failed');
  });

  it('keeps stop action and explanatory spinner text during long extraction', () => {
    component.stateService.updateDiliAgent({
      isRunning: true,
      jobStatus: 'running',
      jobId: 'job-123',
      jobStage: 'therapy_extraction',
      jobStageMessage: 'Parsing therapy section',
      jobProgress: 23,
    });

    expect(component.showSpinner).toBeTruthy();
    expect(component.runActionLabel).toBe('Stop analysis');
    expect(component.runActionDisabled).toBeFalsy();
    expect(component.spinnerStatusLabel).toContain(
      'Local model extraction can take several minutes.',
    );
  });

  it('recovers stale polling by fetching the latest job status snapshot', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          job_id: 'job-123',
          job_type: 'clinical',
          status: 'completed',
          progress: 100,
          result: {
            report: 'Recovered final report',
            progress_message: 'Step 15/15: Auditing artifacts and saving session results...',
          },
          error: null,
          version: 42,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    );

    component.stateService.updateDiliAgent({
      isRunning: true,
      jobStatus: 'running',
      jobId: 'job-123',
      jobStage: 'therapy_extraction',
      jobStageMessage: 'Parsing therapy section',
      jobProgress: 23,
    });
    (component as unknown as { pollIntervalMs: number }).pollIntervalMs = 1000;
    (component as unknown as { lastPollResponseTimestamp: number }).lastPollResponseTimestamp =
      Date.now() - 20_000;

    await (component as unknown as { recoverPollingIfStale: () => Promise<void> }).recoverPollingIfStale();

    expect(fetchSpy).toHaveBeenCalledOnce();
    expect(component.vm.isRunning).toBeFalsy();
    expect(component.vm.jobStatus).toBe('completed');
    expect(component.vm.jobProgress).toBe(100);
    expect(component.vm.message).toContain('Recovered final report');
    expect(component.vm.jobStage).toBe('completed');
    expect(component.vm.jobStageMessage).toBe('Clinical analysis completed.');
    expect(component.vm.exportUrl).toBeTruthy();
  });
});
