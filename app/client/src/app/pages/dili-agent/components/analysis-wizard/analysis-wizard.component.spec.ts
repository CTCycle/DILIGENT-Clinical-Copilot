// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  ClinicalInputPreflightIssue,
  ClinicalInputPreflightResponse,
  ClinicalRequestPayload,
  ModelConfigStateResponse,
} from '../../../../core/models/types';
import { DiliJobTrackerService } from '../../../../core/services/dili-job-tracker.service';
import { AppStateService } from '../../../../core/state/app-state.service';
import { ModelConfigStateService } from '../../../../core/state/model-config-state.service';
import { AnalysisWizardComponent, LIVERTOX_WARNING } from './analysis-wizard.component';

function issue(severity: 'blocking' | 'non_blocking', code: string, field: string): ClinicalInputPreflightIssue {
  return {
    severity,
    code,
    field,
    message: code,
    title: `${code} title`,
    description: `${code} description`,
    affected_section: field,
    consequence: `${code} consequence`,
    continuation_allowed: severity === 'non_blocking',
  };
}

function preflight(blocking: ClinicalInputPreflightIssue[] = [], warnings: ClinicalInputPreflightIssue[] = []) {
  return {
    ready: blocking.length === 0,
    blocking_issues: blocking,
    non_blocking_issues: warnings,
    runtime_settings: {},
    extraction_quality: {},
    deterministic_diagnostics: {},
    rag_readiness: null,
  } satisfies ClinicalInputPreflightResponse;
}

function modelConfig(): ModelConfigStateResponse {
  return {
    local_models: [
      { name: 'qwen:9b', available_in_ollama: true },
      { name: 'other:7b', available_in_ollama: true },
    ],
    cloud_providers: [],
    use_cloud_services: false,
    llm_provider: '',
    cloud_model: null,
    clinical_model: 'qwen:9b',
    text_extraction_model: 'qwen:9b',
    revision_model: 'qwen:9b',
    timeline_model: 'qwen:9b',
    reasoning_level: 'off',
  } as unknown as ModelConfigStateResponse;
}

describe('AnalysisWizardComponent', () => {
  let fixture: ComponentFixture<AnalysisWizardComponent>;
  let component: AnalysisWizardComponent;
  let appState: AppStateService;
  let tracker: DiliJobTrackerService;
  let payloads: ClinicalRequestPayload[];
  let responses: ClinicalInputPreflightResponse[];

  const root = () => fixture.nativeElement as HTMLElement;
  const button = (label: string) =>
    Array.from(root().querySelectorAll('button')).find((item) => item.textContent?.trim() === label) as
      | HTMLButtonElement
      | undefined;

  const startButton = () => root().querySelector('.wizard-footer .btn-primary') as HTMLButtonElement | null;

  async function settle(): Promise<void> {
    for (let i = 0; i < 5; i += 1) {
      fixture.detectChanges();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  beforeEach(async () => {
    localStorage.clear();
    payloads = [];
    responses = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
      payloads.push(JSON.parse(String(init?.body)) as ClinicalRequestPayload);
      return new Response(JSON.stringify(responses.shift() ?? preflight()), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });
    await TestBed.configureTestingModule({ imports: [AnalysisWizardComponent] }).compileComponents();
    TestBed.inject(ModelConfigStateService).setFromApiState(modelConfig());
    appState = TestBed.inject(AppStateService);
    tracker = TestBed.inject(DiliJobTrackerService);
    appState.updateDiliAgent({
      form: { patientName: 'P', visitDate: '2026-10-01', patientImageDataUrl: null, clinicalInput: 'text' },
    });
    fixture = TestBed.createComponent(AnalysisWizardComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('isOpen', true);
    fixture.detectChanges();
    await settle();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('runs preflight on open with LiverTox selected by default', () => {
    expect(payloads).toHaveLength(1);
    expect(payloads[0].evidence_sources).toEqual(['livertox']);
    expect(payloads[0].model_overrides).toBeNull();
    expect('use_rag' in payloads[0]).toBe(false);
  });

  it('returns to preflight when reopened after the clinical input changed', async () => {
    component.goTo(1);
    fixture.componentRef.setInput('isOpen', false);
    fixture.detectChanges();
    responses.push(preflight([issue('blocking', 'visit_date_missing', 'visit_date')]));
    appState.updateDiliAgent({
      form: { patientName: 'P', visitDate: '', patientImageDataUrl: null, clinicalInput: 'text' },
    });

    fixture.componentRef.setInput('isOpen', true);
    await settle();

    expect(component.step()).toBe('preflight');
    expect(payloads).toHaveLength(2);
    expect(component.blockingCount()).toBe(1);
  });

  it('labels LiverTox as recommended and warns when it is unticked, on both steps', async () => {
    component.goTo(1);
    fixture.detectChanges();
    expect(root().textContent).toContain('Recommended');
    expect(root().textContent).not.toContain(LIVERTOX_WARNING);

    const livertox = root().querySelector('#evidence-livertox') as HTMLInputElement;
    livertox.checked = false;
    livertox.dispatchEvent(new Event('change'));
    fixture.detectChanges();

    expect(root().textContent).toContain(LIVERTOX_WARNING);
    expect(root().textContent).toContain('LLM knowledge only');
    expect(root().querySelector('[role="status"]')).not.toBeNull();

    component.goTo(3);
    fixture.detectChanges();
    expect(root().textContent).toContain(LIVERTOX_WARNING);
  });

  it('blocks Next and later steps while preflight has blocking issues', async () => {
    responses.push(preflight([issue('blocking', 'visit_date_missing', 'visit_date')]));
    component.reset();
    await component.runPreflight();
    fixture.detectChanges();

    expect(button('Next')?.disabled).toBe(true);
    expect(component.canVisit(1)).toBe(false);
    expect(root().textContent).toContain('1 blocking');
  });

  it('invalidates the check when sources change and re-runs it on Review before Start enables', async () => {
    expect(component.checkCurrent()).toBe(true);

    component.toggleSource('dilirank', true);
    expect(component.checkCurrent()).toBe(false);

    component.goTo(3);
    fixture.detectChanges();
    expect(startButton()?.disabled).toBe(true);
    await settle();

    expect(payloads.at(-1)?.evidence_sources).toEqual(['livertox', 'dilirank']);
    expect(component.checkCurrent()).toBe(true);
    expect(startButton()?.disabled).toBe(false);
  });

  it('sends only changed roles as overrides and restores defaults', () => {
    component.setModel('clinical_model', 'other:7b');
    expect(component.modelOverrides()).toEqual({ clinical_model: 'other:7b' });

    component.setModel('clinical_model', 'qwen:9b');
    expect(component.modelOverrides()).toBeNull();

    component.setModel('text_extraction_model', 'other:7b');
    component.restoreModelDefaults();
    expect(component.modelOverrides()).toBeNull();
  });

  it('starts once even when Start is clicked twice', async () => {
    const startSpy = vi.spyOn(tracker, 'startSession').mockImplementation(async () => {
      appState.updateDiliAgent({ isRunning: true });
    });
    const started = vi.fn();
    component.started.subscribe(started);
    component.goTo(3);
    fixture.detectChanges();

    await Promise.all([component.start(), component.start()]);

    expect(startSpy).toHaveBeenCalledTimes(1);
    expect(started).toHaveBeenCalledTimes(1);
    expect(component.sources()).toEqual(['livertox']);
  });

  it('re-enables Start and shows the error when the job could not be started', async () => {
    vi.spyOn(tracker, 'startSession').mockImplementation(async () => {
      appState.updateDiliAgent({ isRunning: false, message: '[ERROR] backend down' });
    });
    component.goTo(3);

    await component.start();
    fixture.detectChanges();

    expect(component.starting()).toBe(false);
    expect(root().textContent).toContain('[ERROR] backend down');
  });
});
