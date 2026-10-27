// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SystemStatusResponse } from '../services/system-status-api';
import { SYSTEM_STATUS_POLL_INTERVAL_MS, SystemStatusStateService } from './system-status-state.service';

const STATUS: SystemStatusResponse = {
  providers: { active: 1, total: 6 },
  ollama: 'reachable',
  cpu_percent: 12,
  ram: { used_bytes: 8, total_bytes: 16 },
  vram: null,
};

function setHidden(hidden: boolean): void {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
  document.dispatchEvent(new Event('visibilitychange'));
}

describe('SystemStatusStateService', () => {
  let service: SystemStatusStateService;
  const fetchSpy = vi.fn();

  beforeEach(() => {
    vi.useFakeTimers();
    setHidden(false);
    fetchSpy.mockReset().mockImplementation(async () =>
      new Response(JSON.stringify(STATUS), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchSpy);
    service = TestBed.inject(SystemStatusStateService);
  });

  afterEach(() => {
    service.stop();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('reports a failed poll as disconnected and recovers on the next success', async () => {
    fetchSpy.mockRejectedValueOnce(new Error('down'));
    service.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(service.connection()).toBe('disconnected');
    expect(service.status()).toBeNull();

    await vi.advanceTimersByTimeAsync(SYSTEM_STATUS_POLL_INTERVAL_MS);
    expect(service.connection()).toBe('connected');
    expect(service.status()).toEqual(STATUS);
  });

  it('stops polling while the tab is hidden and refreshes when it becomes visible', async () => {
    service.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    setHidden(true);
    await vi.advanceTimersByTimeAsync(SYSTEM_STATUS_POLL_INTERVAL_MS * 3);
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    setHidden(false);
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });
});
