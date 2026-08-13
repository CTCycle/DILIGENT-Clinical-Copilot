import { ComponentFixture, TestBed } from '@angular/core/testing';
import { vi } from 'vitest';

import {
  ClinicalInputPreflightIssue,
  ClinicalInputPreflightResponse,
  ClinicalRequestPayload,
} from '../../core/models/types';
import { DiliJobTrackerService } from '../../core/services/dili-job-tracker.service';
import { DiliAgentPageComponent } from './dili-agent-page.component';

function issue(
  severity: 'blocking' | 'non_blocking',
  code: string,
  field: string,
): ClinicalInputPreflightIssue {
  return {
    severity,
    code,
    field,
    message: `${code} message`,
    title: `${code} title`,
    description: `${code} description`,
    affected_section: field,
    consequence: `${code} consequence`,
    continuation_allowed: severity === 'non_blocking',
  };
}

function result(
  blocking: ClinicalInputPreflightIssue[] = [],
  warnings: ClinicalInputPreflightIssue[] = [],
): ClinicalInputPreflightResponse {
  return {
    ready: blocking.length === 0,
    blocking_issues: blocking,
    non_blocking_issues: warnings,
    runtime_settings: {},
    extraction_quality: {},
    deterministic_diagnostics: {},
    rag_readiness: null,
  };
}

describe('DiliAgentPageComponent', () => {
  let fixture: ComponentFixture<DiliAgentPageComponent>;
  let component: DiliAgentPageComponent;
  let tracker: DiliJobTrackerService;

  beforeEach(async () => {
    localStorage.clear();
    await TestBed.configureTestingModule({
      imports: [DiliAgentPageComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(DiliAgentPageComponent);
    component = fixture.componentInstance;
    tracker = TestBed.inject(DiliJobTrackerService);
    fixture.detectChanges();
  });

  it('keeps the run action available so the complete pre-flight can aggregate missing inputs', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, visitDate: '', clinicalInput: '' },
    });

    fixture.detectChanges();

    const runButton = fixture.nativeElement.querySelector('#run-analysis-button') as HTMLButtonElement | null;
    expect(runButton?.disabled).toBe(false);
  });

  it('renders all blocking issues and does not render a continuation action', () => {
    component.preflightDialog.set(
      result([
        issue('blocking', 'clinical_input_missing', 'clinical_input'),
        issue('blocking', 'visit_date_missing', 'visit_date'),
      ]),
    );

    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.textContent).toContain('Cannot start analysis');
    expect(root.querySelectorAll('.preflight-issue')).toHaveLength(2);
    expect(root.textContent).toContain('Return to input');
    expect(root.textContent).not.toContain('Continue with limitations');
  });

  it('renders multiple warnings with explicit return and continuation actions', () => {
    component.preflightDialog.set(
      result([], [
        issue('non_blocking', 'missing_timing', 'drugs'),
        issue('non_blocking', 'sparse_context', 'anamnesis'),
      ]),
    );

    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.textContent).toContain('Review before continuing');
    expect(root.querySelectorAll('.preflight-issue')).toHaveLength(2);
    expect(root.textContent).toContain('Return to input');
    expect(root.textContent).toContain('Continue with limitations');
  });

  it('renders a single blocking issue with only the return action', () => {
    component.preflightDialog.set(
      result([issue('blocking', 'visit_date_missing', 'visit_date')]),
    );

    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelectorAll('.preflight-issue')).toHaveLength(1);
    expect(root.querySelector('.modal-close')).toBeNull();
    expect(root.textContent).toContain('Return to input');
    expect(root.textContent).not.toContain('Continue with limitations');
  });

  it('does not dismiss a blocking pre-flight with Escape', () => {
    component.preflightDialog.set(
      result([issue('blocking', 'visit_date_missing', 'visit_date')]),
    );
    fixture.detectChanges();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    fixture.detectChanges();

    expect(component.preflightDialog()).not.toBeNull();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain('Cannot start analysis');
  });

  it('allows Escape to dismiss a non-blocking pre-flight', () => {
    component.preflightDialog.set(
      result([], [issue('non_blocking', 'missing_timing', 'drugs')]),
    );
    fixture.detectChanges();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    fixture.detectChanges();

    expect(component.preflightDialog()).toBeNull();
  });

  it('renders a single non-blocking warning with both explicit actions', () => {
    component.preflightDialog.set(
      result([], [issue('non_blocking', 'missing_timing', 'drugs')]),
    );

    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelectorAll('.preflight-issue')).toHaveLength(1);
    expect(root.textContent).toContain('Return to input');
    expect(root.textContent).toContain('Continue with limitations');
  });

  it('renders mixed blocking and warning issues but prevents continuation', () => {
    component.preflightDialog.set(
      result(
        [issue('blocking', 'clinical_input_missing', 'clinical_input')],
        [issue('non_blocking', 'missing_timing', 'drugs')],
      ),
    );

    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelectorAll('.preflight-issue')).toHaveLength(2);
    expect(root.textContent).toContain('1 blocking');
    expect(root.textContent).toContain('1 warning');
    expect(root.textContent).not.toContain('Continue with limitations');
  });

  it('reopens with the latest validation state after the form changes', () => {
    component.preflightDialog.set(
      result([issue('blocking', 'visit_date_missing', 'visit_date')]),
    );
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain(
      'visit_date_missing title',
    );

    component.returnToInputFromPreflight();
    component.handleFormChange('visitDate', '2026-07-17');
    component.preflightDialog.set(
      result([], [issue('non_blocking', 'missing_timing', 'drugs')]),
    );
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.textContent).not.toContain('visit_date_missing title');
    expect(root.textContent).toContain('missing_timing title');
    expect(root.textContent).toContain('Continue with limitations');
  });

  it('returns to the input panel without clearing entered data', () => {
    component.stateService.updateDiliAgent({
      form: {
        ...component.vm.form,
        patientName: 'Persisted Patient',
        visitDate: '2026-07-17',
        clinicalInput: 'Persisted clinical text',
      },
    });
    component.preflightDialog.set(
      result([issue('blocking', 'clinical_input_missing', 'clinical_input')]),
    );

    component.returnToInputFromPreflight();

    expect(component.preflightDialog()).toBeNull();
    expect(component.vm.form.patientName).toBe('Persisted Patient');
    expect(component.vm.form.visitDate).toBe('2026-07-17');
    expect(component.vm.form.clinicalInput).toBe('Persisted clinical text');
  });

  it('continues once and disables RAG only for the accepted pending run', async () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, useRag: true },
    });
    const payload: ClinicalRequestPayload = {
      name: 'Patient',
      visit_date: { day: 17, month: 7, year: 2026 },
      clinical_input: 'Clinical input',
      selected_model_providers: ['ollama'],
      use_rag: true,
    };
    const internal = component as unknown as {
      pendingPreflightPayload: ClinicalRequestPayload | null;
    };
    internal.pendingPreflightPayload = payload;
    component.preflightDialog.set(
      result([], [issue('non_blocking', 'rag_ollama_unavailable', 'use_rag')]),
    );
    const startSpy = vi.spyOn(tracker, 'startSession').mockResolvedValue();

    await Promise.all([
      component.continueAfterPreflight(),
      component.continueAfterPreflight(),
    ]);

    expect(startSpy).toHaveBeenCalledTimes(1);
    expect(startSpy).toHaveBeenCalledWith(
      expect.objectContaining({ use_rag: false }),
      null,
    );
    expect(component.vm.form.useRag).toBe(true);
  });

  it('renders a bounded issue-list scroll region for long validation results', () => {
    component.preflightDialog.set(
      result(
        [],
        Array.from({ length: 18 }, (_, index) =>
          issue('non_blocking', `warning_${index}`, 'clinical_input'),
        ),
      ),
    );

    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    const list = root.querySelector<HTMLElement>('[data-testid="preflight-issue-list"]');
    expect(list).not.toBeNull();
    expect(root.querySelectorAll('.preflight-issue')).toHaveLength(18);
    expect(list?.classList.contains('preflight-issue-list')).toBe(true);
  });

  it('renders and persists the RAG evidence toggle', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, useRag: false },
    });
    fixture.detectChanges();

    const toggle = fixture.nativeElement.querySelector('#rag-enabled') as HTMLInputElement | null;
    expect(toggle).not.toBeNull();
    toggle!.checked = true;
    toggle!.dispatchEvent(new Event('change'));
    fixture.detectChanges();

    expect(component.vm.form.useRag).toBe(true);
  });

  it('allows an active run to be stopped while the start click debounce is active', () => {
    const stopSessionSpy = vi.spyOn(component, 'stopSession').mockResolvedValue();
    component.stateService.updateDiliAgent({
      isRunning: true,
      jobId: 'job-123',
    });
    (component as unknown as { runControlDebounced: boolean }).runControlDebounced = true;

    component.runOrStop();

    expect(stopSessionSpy).toHaveBeenCalled();
  });

  it('keeps the stop action and explanatory spinner text during long extraction', () => {
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
});
