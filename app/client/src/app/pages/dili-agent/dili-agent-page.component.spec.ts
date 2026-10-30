// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { vi } from 'vitest';

import { DiliJobTrackerService } from '../../core/services/dili-job-tracker.service';
import { ClinicalDraftService } from '../../core/services/clinical-draft.service';
import { DiliAgentPageComponent } from './dili-agent-page.component';

describe('DiliAgentPageComponent', () => {
  let fixture: ComponentFixture<DiliAgentPageComponent>;
  let component: DiliAgentPageComponent;
  let tracker: DiliJobTrackerService;
  let draftService: ClinicalDraftService;

  beforeEach(async () => {
    localStorage.clear();
    draftService = {
      loading: signal(false),
      pendingRecovery: signal(false),
      autosaveEnabled: signal(false),
      saveStatus: signal('idle'),
      lastSavedAt: signal(null),
      hasPersistedDraft: signal(false),
      hasUnsavedChanges: signal(false),
      errorMessage: signal(null),
      loadOnOpen: vi.fn().mockResolvedValue(true),
      getRestorableForm: vi.fn().mockReturnValue(null),
      resume: vi.fn(),
      saveAndEnable: vi.fn().mockResolvedValue(true),
      scheduleSave: vi.fn(),
      retrySave: vi.fn().mockResolvedValue(true),
      discard: vi.fn().mockResolvedValue(true),
      reloadDraft: vi.fn().mockResolvedValue(true),
    } as unknown as ClinicalDraftService;
    await TestBed.configureTestingModule({
      imports: [DiliAgentPageComponent],
      providers: [{ provide: ClinicalDraftService, useValue: draftService }],
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

  it('opens the setup wizard instead of starting a run, and has no RAG checkbox', () => {
    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelector('#rag-enabled')).toBeNull();
    expect(root.querySelector('app-analysis-wizard')).not.toBeNull();

    component.runOrStop();

    expect(component.wizardOpen()).toBe(true);
  });

  it('enables the report link only when a report exists or a job is running', () => {
    const link = () => fixture.nativeElement.querySelector('.stitch-dili-report-link') as HTMLButtonElement;
    fixture.detectChanges();
    expect(link().disabled).toBe(true);

    component.stateService.updateDiliAgent({ isRunning: true, jobStatus: 'running', jobId: 'job-1' });
    fixture.detectChanges();
    expect(link().disabled).toBe(false);

    component.stateService.updateDiliAgent({
      isRunning: false,
      jobStatus: 'completed',
      message: '# Report',
    });
    fixture.detectChanges();
    expect(link().disabled).toBe(false);
  });

  it('shows the progress panel only once a job exists', () => {
    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelector('app-analysis-progress')).toBeNull();

    component.stateService.updateDiliAgent({
      isRunning: true,
      jobStatus: 'running',
      jobId: 'job-123',
      jobStage: 'therapy_extraction',
      jobProgress: 23,
    });
    fixture.detectChanges();

    expect(root.querySelector('app-analysis-progress')).not.toBeNull();
    expect(component.runActionLabel).toBe('Stop analysis');
    expect(component.runActionDisabled).toBeFalsy();
  });

  it('allows an active run to be stopped', () => {
    const stopSessionSpy = vi.spyOn(component, 'stopSession').mockResolvedValue();
    component.stateService.updateDiliAgent({
      isRunning: true,
      jobId: 'job-123',
    });

    component.runOrStop();

    expect(stopSessionSpy).toHaveBeenCalled();
    expect(component.wizardOpen()).toBe(false);
  });


  it('keeps the initial recovery read separate from the in-memory form and active job', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, patientName: 'Unsaved local name' },
      jobId: 'job-still-running',
      jobStatus: 'running',
      isRunning: true,
    });

    expect(draftService.loadOnOpen).toHaveBeenCalledOnce();
    expect(component.vm.form.patientName).toBe('Unsaved local name');
    expect(component.vm.jobId).toBe('job-still-running');
  });

  it('schedules form changes only after the clinician opts in or resumes', () => {
    component.handleFormChange('clinicalInput', 'unsaved text');
    expect(draftService.scheduleSave).not.toHaveBeenCalled();

    draftService.autosaveEnabled.set(true);
    component.handleFormChange('clinicalInput', 'opted-in text');
    expect(draftService.scheduleSave).toHaveBeenCalledOnce();
    expect(draftService.scheduleSave).toHaveBeenCalledWith(component.vm.form);
  });

  it('restores only the approved fields, leaves image out, and starts no assessment', () => {
    vi.mocked(draftService.getRestorableForm).mockReturnValue({
      patient_name: 'Recovered synthetic patient',
      visit_date: '2026-10-23',
      clinical_input: 'Exact\nclinical text  ',
    });
    const startSpy = vi.spyOn(tracker, 'startSession');

    component.resumeDraft();

    expect(component.vm.form).toEqual({
      patientName: 'Recovered synthetic patient',
      visitDate: '2026-10-23',
      patientImageDataUrl: null,
      clinicalInput: 'Exact\nclinical text  ',
    });
    expect(draftService.resume).toHaveBeenCalledOnce();
    expect(startSpy).not.toHaveBeenCalled();
  });

  it('requires confirmation before replacing different current input with a saved draft', () => {
    vi.mocked(draftService.getRestorableForm).mockReturnValue({
      patient_name: 'Saved patient',
      visit_date: null,
      clinical_input: 'Saved text',
    });
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, patientName: 'Current patient' },
    });

    component.resumeDraft();

    expect(component.draftConfirmation()).toBe('resume');
    expect(component.vm.form.patientName).toBe('Current patient');
    expect(draftService.resume).not.toHaveBeenCalled();
  });

  it('does not resume over an active job', () => {
    vi.mocked(draftService.getRestorableForm).mockReturnValue({
      patient_name: 'Saved patient',
      visit_date: null,
      clinical_input: 'Saved text',
    });
    component.stateService.updateDiliAgent({
      jobId: 'active-job',
      jobStatus: 'running',
      isRunning: true,
    });

    component.resumeDraft();

    expect(draftService.resume).not.toHaveBeenCalled();
    expect(component.vm.jobId).toBe('active-job');
  });

  it('clears a saved draft only after conditional discard succeeds', async () => {
    draftService.hasPersistedDraft.set(true);
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, patientName: 'Keep this until deletion succeeds' },
    });
    vi.mocked(draftService.discard).mockResolvedValue(false);

    component.clearAll();
    expect(component.draftConfirmation()).toBe('clear');
    await component.confirmDraftAction();

    expect(component.vm.form.patientName).toBe('Keep this until deletion succeeds');
    expect(draftService.discard).toHaveBeenCalledOnce();
  });
});
