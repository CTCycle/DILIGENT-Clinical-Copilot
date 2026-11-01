// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { BehaviorSubject } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ClinicalSessionState } from '../../core/models/types';
import { FetchCall, jsonResponse, sessionState } from '../../core/testing/clinical-session-fixtures';
import { DiliReportPageComponent } from './dili-report-page.component';

const DETAIL = {
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

const INPUT = {
  session_id: 7,
  status: 'completed',
  patient_name: 'Stored Patient',
  visit_date: '2026-09-01',
  clinical_input: 'Original clinical text',
  patient_image_base64: null,
  updated_at: '2026-09-01T08:00:00Z',
};

describe('DiliReportPageComponent', () => {
  let fixture: ComponentFixture<DiliReportPageComponent>;
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let current: ClinicalSessionState;
  let cancelResponse: ClinicalSessionState | Response;
  const params = new BehaviorSubject(convertToParamMap({ sessionId: '7' }));

  const root = () => fixture.nativeElement as HTMLElement;
  const text = () => root().textContent ?? '';
  const buttonByText = (label: string) =>
    [...root().querySelectorAll('button')].find((button) => button.textContent?.includes(label));

  async function settle(): Promise<void> {
    for (let i = 0; i < 5; i += 1) {
      fixture.detectChanges();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  async function create(state: Partial<ClinicalSessionState>): Promise<void> {
    current = sessionState({ session_id: 7, patient_name: 'Stored Patient', ...state });
    fixture = TestBed.createComponent(DiliReportPageComponent);
    await settle();
  }

  beforeEach(async () => {
    cancelResponse = sessionState();
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith('/cancel')) {
        return cancelResponse instanceof Response ? cancelResponse : jsonResponse(cancelResponse);
      }
      if (url.endsWith('/status')) return jsonResponse(current);
      if (url.endsWith('/input')) return jsonResponse(INPUT);
      if (url.endsWith('/inspection/sessions/7')) return jsonResponse(DETAIL);
      throw new Error(`Unexpected request ${init?.method} ${url}`);
    });
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
  });

  afterEach(() => {
    fixture?.destroy();
    vi.restoreAllMocks();
  });

  it('shows the queue position while a session waits for capacity', async () => {
    await create({ status: 'queued', queue_position: 2 });

    expect(text()).toContain('Waiting for capacity');
    expect(text()).toContain('position 2');
    expect(root().querySelector('app-analysis-progress')).toBeNull();
    expect(buttonByText('Cancel analysis')).toBeDefined();
  });

  it('renders the stored report, metadata, and report actions once completed', async () => {
    await create({ status: 'completed', has_report: true });

    expect(root().querySelector('app-clinical-report-viewer')).not.toBeNull();
    expect(text()).toContain('Stored report');
    expect(text()).toContain('LiverTox, FDA DILIrank 2.0');
    expect(text()).toContain('Override');
    expect(root().querySelector('[aria-label="Copy report"]')).not.toBeNull();
    expect(buttonByText('Cancel analysis')).toBeUndefined();
  });

  it('confirms inline before cancelling and then shows the cooperative stop state', async () => {
    await create({ status: 'running', progress: 30, progress_stage: 'drugs.resolving' });
    expect(root().querySelector('app-analysis-progress')).not.toBeNull();
    expect(text()).toContain('Normalizing drug names');

    buttonByText('Cancel analysis')!.click();
    await settle();
    expect(text()).toContain('Cancel this analysis?');
    expect((fetchSpy.mock.calls as FetchCall[]).some(([url]) => String(url).endsWith('/cancel'))).toBe(false);

    cancelResponse = sessionState({
      session_id: 7,
      status: 'running',
      stop_requested: true,
      version: 5,
    });
    buttonByText('Yes, cancel')!.click();
    await settle();

    expect(text()).toContain('Stopping after the current step');
    expect(buttonByText('Cancel analysis')).toBeUndefined();
    expect(text()).not.toContain('aborted');
  });

  it('keeps the running view and shows the error when cancellation is refused', async () => {
    await create({ status: 'running' });
    cancelResponse = jsonResponse({ detail: 'Session is not cancellable.' }, 409);

    buttonByText('Cancel analysis')!.click();
    await settle();
    buttonByText('Yes, cancel')!.click();
    await settle();

    expect(text()).toContain('Another operation is already running');
    expect(buttonByText('Cancel analysis')).toBeDefined();
  });

  it('explains an interrupted run and still shows the partial report', async () => {
    await create({
      status: 'failed',
      has_report: true,
      error: { code: 'interrupted', message: 'Application restarted during analysis.' },
    });

    expect(text()).toContain('Application restarted during analysis.');
    expect(text()).toContain('Partial report');
    expect(text()).toContain('Stored report');
  });

  it('offers editing for a draft and exposes the original clinical input', async () => {
    await create({ status: 'draft' });

    const editLink = [...root().querySelectorAll('a')].find((anchor) => anchor.textContent?.includes('Edit draft'));
    expect(editLink?.getAttribute('href')).toBe('/?session=7');
    expect(text()).toContain('This session is a draft');
    expect(root().querySelector('details')?.textContent).toContain('Original clinical text');
  });

  it('falls back to the session number when there is no patient name', async () => {
    await create({ status: 'queued', patient_name: null });

    expect(root().querySelector('h1')?.textContent).toContain('Session #7');
  });

  it('shows an error when the session cannot be loaded', async () => {
    fetchSpy.mockImplementation(async () => jsonResponse({ detail: 'Session not found.' }, 404));
    fixture = TestBed.createComponent(DiliReportPageComponent);
    await settle();

    expect(text()).toContain('Session unavailable');
  });
});
