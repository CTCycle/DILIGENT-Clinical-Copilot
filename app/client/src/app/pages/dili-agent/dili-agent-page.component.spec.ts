import { ComponentFixture, TestBed } from '@angular/core/testing';
import { vi } from 'vitest';

import { ClinicalRequestPayload } from '../../core/models/types';
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

  it('renders disabled run reason as visible associated feedback', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, visitDate: '2025-01-01', clinicalInput: 'word '.repeat(59) },
    });

    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    const runButton = root.querySelector<HTMLButtonElement>('.stitch-dili-actions .btn-primary');
    const reason = root.querySelector<HTMLElement>('#run-disabled-reason');

    expect(runButton?.disabled).toBe(true);
    expect(runButton?.getAttribute('aria-describedby')).toBe('run-disabled-reason');
    expect(reason?.textContent).toContain('Clinical input needs at least 60 words.');
  });

  it('canStartSession true when all preconditions are met', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, visitDate: '2025-01-01', clinicalInput: 'word '.repeat(60) },
      settings: { ...component.vm.settings, provider: 'openai' },
    });
    expect(component.canStartSession()).toBeTruthy();
  });

  it('renders and persists the RAG evidence toggle', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, useRag: false },
    });
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    const toggle = root.querySelector<HTMLInputElement>('#rag-enabled');
    expect(toggle).not.toBeNull();
    expect(root.textContent).toContain('Use RAG evidence');
    toggle!.checked = true;
    toggle!.dispatchEvent(new Event('change'));
    fixture.detectChanges();

    expect(component.vm.form.useRag).toBe(true);
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

  it('keeps stop action and explanatory spinner text during long extraction', () => {
    component.stateService.updateDiliAgent({
      isRunning: true,
      jobStatus: 'running',
      jobId: 'job-123',
      jobStage: 'therapy_extraction',
      jobStageMessage: 'Parsing therapy section',
      jobProgress: 23,
      jobStartedAtMs: Date.now() - 70_000,
      jobLastProgressAtMs: Date.now() - 5_000,
    });

    expect(component.showSpinner).toBeTruthy();
    expect(component.runActionLabel).toBe('Stop analysis');
    expect(component.runActionDisabled).toBeFalsy();
    expect(component.spinnerStatusLabel).toContain(
      'Local model extraction can take several minutes.',
    );
  });

  it('renders the Ollama RAG readiness dialog', () => {
    component.ragReadinessDialog.set({
      requested: true,
      available: false,
      backend: 'ollama',
      model: 'nomic-embed-text:latest',
      reason_code: 'rag_ollama_unavailable',
      message: 'Start Ollama and retry.',
    });

    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('RAG requires Ollama');
    expect(fixture.nativeElement.textContent).toContain('Run without RAG');
    expect(fixture.nativeElement.textContent).toContain('nomic-embed-text:latest');
  });

  it('disables RAG only for the pending run', async () => {
    const internal = component as unknown as {
      executeRunSession: (payload?: ClinicalRequestPayload) => Promise<void>;
      pendingRagPayload: ClinicalRequestPayload | null;
    };
    const executeSpy = vi
      .spyOn(internal, 'executeRunSession')
      .mockResolvedValue();
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, useRag: true },
    });
    internal.pendingRagPayload = {
      name: null,
      visit_date: null,
      clinical_input: 'clinical input',
      selected_model_providers: ['openai'],
      use_rag: true,
    };

    await component.runWithoutRag();

    expect(executeSpy).toHaveBeenCalledWith(
      expect.objectContaining({ use_rag: false }),
    );
    expect(component.vm.form.useRag).toBe(true);
  });
});
