// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { BehaviorSubject } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AppStateService } from '../../core/state/app-state.service';
import { DiliReportPageComponent } from './dili-report-page.component';

const SESSION = {
  session_id: 7,
  patient_name: 'Stored Patient',
  visit_date: '2026-09-01',
  text_extraction_model: 'qwen:9b',
  clinical_model: 'other:7b',
  metadata: {
    evidence_sources: ['livertox', 'dilirank'],
    model_overrides: { clinical_model: 'other:7b' },
  },
  report: '# Stored report',
  result_payload: {},
};

describe('DiliReportPageComponent', () => {
  let fixture: ComponentFixture<DiliReportPageComponent>;
  let appState: AppStateService;
  const params = new BehaviorSubject(convertToParamMap({}));
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  const text = () => (fixture.nativeElement as HTMLElement).textContent ?? '';

  async function create(sessionId: string | null): Promise<void> {
    params.next(convertToParamMap(sessionId ? { sessionId } : {}));
    fixture = TestBed.createComponent(DiliReportPageComponent);
    for (let i = 0; i < 5; i += 1) {
      fixture.detectChanges();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  beforeEach(async () => {
    localStorage.clear();
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(
      async () =>
        new Response(JSON.stringify(SESSION), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
    );
    await TestBed.configureTestingModule({
      imports: [DiliReportPageComponent],
      providers: [
        provideRouter([]),
        {
          provide: ActivatedRoute,
          useValue: {
            paramMap: params,
            get snapshot() {
              return { paramMap: params.value };
            },
          },
        },
      ],
    }).compileComponents();
    appState = TestBed.inject(AppStateService);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows the empty state when there is no job or session', async () => {
    await create(null);

    expect(text()).toContain('No report yet');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('shows live progress while a job runs', async () => {
    appState.updateDiliAgent({ isRunning: true, jobStatus: 'running', jobStage: 'drugs.resolving', jobProgress: 30 });
    await create(null);

    expect(fixture.nativeElement.querySelector('app-analysis-progress')).not.toBeNull();
    expect(text()).toContain('Normalizing drug names');
  });

  it('shows the finished workspace report and loads session metadata for the header', async () => {
    appState.updateDiliAgent({ jobStatus: 'completed', message: '# Live report', sessionId: 7 });
    await create(null);

    expect(fixture.nativeElement.querySelector('app-clinical-report-viewer')).not.toBeNull();
    expect(text()).toContain('Live report');
    expect(text()).toContain('LiverTox, FDA DILIrank 2.0');
    expect(text()).toContain('Override');
  });

  it('re-fetches the report from the persisted session id after a reload', async () => {
    appState.updateDiliAgent({ sessionId: 7 });
    await create(null);

    expect(String(fetchSpy.mock.calls[0][0])).toContain('/inspection/sessions/7');
    expect(text()).toContain('Stored report');
  });

  it('shows the failure message for a failed run', async () => {
    appState.updateDiliAgent({ jobStatus: 'failed', message: '[ERROR] extraction failed' });
    await create(null);

    expect(text()).toContain('Analysis failed');
    expect(text()).toContain('[ERROR] extraction failed');
  });

  it('opens a stored session from the route', async () => {
    await create('7');

    expect(text()).toContain('Stored Patient');
    expect(text()).toContain('Stored report');
  });
});
