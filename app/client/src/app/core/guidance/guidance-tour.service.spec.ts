import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { DILI_ASSESSMENT_TOUR } from './guidance-content';
import { GuidanceStateService } from './guidance-state.service';
import { GuidanceTourService } from './guidance-tour.service';

describe('GuidanceTourService', () => {
  let service: GuidanceTourService;
  let router: { url: string; navigateByUrl: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    localStorage.clear();
    router = { url: '/', navigateByUrl: vi.fn().mockResolvedValue(true) };
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [
        GuidanceTourService,
        { provide: Router, useValue: router },
      ],
    });
    service = TestBed.inject(GuidanceTourService);
  });

  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    TestBed.resetTestingModule();
  });

  it('keeps the DILI walkthrough targets in workflow order', () => {
    expect(DILI_ASSESSMENT_TOUR.version).toBe(2);
    expect(DILI_ASSESSMENT_TOUR.steps.map((step) => step.target)).toEqual([
      '[data-guidance-target="dili-clinical-input"]',
      '[data-guidance-target="dili-patient-details"]',
      '[data-guidance-target="dili-rag-toggle"]',
      '[data-guidance-target="dili-review-run"]',
    ]);
  });

  it('supports start, next, back, skip, completion, and manual restart state', () => {
    const state = TestBed.inject(GuidanceStateService);
    service.start(DILI_ASSESSMENT_TOUR);
    expect(service.activeTour()?.stepIndex).toBe(0);
    expect(state.status('dili-assessment-tour')?.status).toBe('seen');

    service.next();
    expect(service.activeTour()?.stepIndex).toBe(1);
    service.back();
    expect(service.activeTour()?.stepIndex).toBe(0);
    service.skip();
    expect(service.activeTour()).toBeNull();
    expect(state.status('dili-assessment-tour')?.status).toBe('skipped');

    service.restart(DILI_ASSESSMENT_TOUR);
    expect(service.activeTour()?.stepIndex).toBe(0);
    expect(state.status('dili-assessment-tour')?.restartCount).toBe(1);
    service.next();
    service.next();
    service.next();
    service.next();
    expect(service.activeTour()).toBeNull();
    expect(state.status('dili-assessment-tour')?.status).toBe('completed');
  });
});
