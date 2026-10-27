// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable, signal } from '@angular/core';

import { SystemStatusResponse, fetchSystemStatus } from '../services/system-status-api';

export const SYSTEM_STATUS_POLL_INTERVAL_MS = 5000;

export type BackendConnection = 'loading' | 'connected' | 'disconnected';

/** Single centralized poller for the global status bar; idle while the tab is hidden. */
@Injectable({ providedIn: 'root' })
export class SystemStatusStateService {
  private readonly connectionState = signal<BackendConnection>('loading');
  private readonly statusState = signal<SystemStatusResponse | null>(null);
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private inFlight = false;

  readonly connection = this.connectionState.asReadonly();
  /** Latest successful reading; null while loading or when the backend is unreachable. */
  readonly status = this.statusState.asReadonly();

  private readonly onVisibilityChange = (): void => {
    if (!this.running) {
      return;
    }
    if (document.hidden) {
      this.clearTimer();
    } else {
      void this.refresh();
    }
  };

  start(): void {
    if (this.running) {
      return;
    }
    this.running = true;
    document.addEventListener('visibilitychange', this.onVisibilityChange);
    if (!document.hidden) {
      void this.refresh();
    }
  }

  stop(): void {
    this.running = false;
    document.removeEventListener('visibilitychange', this.onVisibilityChange);
    this.clearTimer();
  }

  private async refresh(): Promise<void> {
    if (this.inFlight) {
      return;
    }
    this.clearTimer();
    this.inFlight = true;
    try {
      this.statusState.set(await fetchSystemStatus());
      this.connectionState.set('connected');
    } catch {
      this.statusState.set(null);
      this.connectionState.set('disconnected');
    } finally {
      this.inFlight = false;
    }
    if (this.running && !document.hidden) {
      this.timer = setTimeout(() => void this.refresh(), SYSTEM_STATUS_POLL_INTERVAL_MS);
    }
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}
