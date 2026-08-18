import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GuidanceStateService, GUIDANCE_STORAGE_KEY } from './guidance-state.service';

describe('GuidanceStateService', () => {
  let service: GuidanceStateService;

  beforeEach(() => {
    localStorage.clear();
    TestBed.resetTestingModule();
    service = TestBed.inject(GuidanceStateService);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
    TestBed.resetTestingModule();
  });

  it('persists versioned status transitions and keeps a manual restart count', () => {
    expect(service.shouldShow('dili-first-assessment', 1)).toBe(true);

    service.markSeen('dili-first-assessment', 1);
    expect(service.status('dili-first-assessment')?.status).toBe('seen');
    expect(service.shouldShow('dili-first-assessment', 1)).toBe(false);
    expect(service.shouldShow('dili-first-assessment', 2)).toBe(true);

    service.dismiss('dili-first-assessment', 1);
    expect(service.status('dili-first-assessment')?.status).toBe('dismissed');
    service.restart('dili-first-assessment', 1);
    expect(service.status('dili-first-assessment')).toMatchObject({
      status: 'restarted',
      restartCount: 1,
    });
    service.complete('dili-first-assessment', 1);
    expect(service.status('dili-first-assessment')).toMatchObject({
      status: 'completed',
      restartCount: 1,
    });

    const persisted = JSON.parse(localStorage.getItem(GUIDANCE_STORAGE_KEY) ?? '{}');
    expect(persisted.schemaVersion).toBe(1);
    expect(persisted.entries['dili-first-assessment'].status).toBe('completed');
  });

  it('ignores malformed or incompatible persisted payloads', () => {
    localStorage.setItem(GUIDANCE_STORAGE_KEY, JSON.stringify({ schemaVersion: 99, entries: { anything: {} } }));
    TestBed.resetTestingModule();
    const freshService = TestBed.inject(GuidanceStateService);

    expect(freshService.state()).toEqual({ schemaVersion: 1, entries: {} });
    expect(freshService.shouldShow('dili-assessment-tour', 1)).toBe(true);
  });

  it('keeps the in-memory state usable when localStorage writes fail', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('storage unavailable');
    });

    service.markSeen('timeline-review-controls', 1);

    expect(service.status('timeline-review-controls')?.status).toBe('seen');
    expect(service.shouldShow('timeline-review-controls', 1)).toBe(false);
  });
});
