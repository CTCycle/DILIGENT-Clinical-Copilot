// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { DestroyRef, Injectable, computed, inject, signal } from '@angular/core';
import { NavigationEnd, Router } from '@angular/router';
import { filter } from 'rxjs';

import { JobSummary, cancelJob, fetchJobs } from './jobs-api';

const ACTIVE_POLL_MS = 2000;
const IDLE_POLL_MS = 10000;
const MAX_BACKOFF_MS = 30000;

/**
 * Single backend-owned view of every background job. Polls GET /api/jobs: fast
 * while jobs run or the monitor is open, slowly otherwise, and not at all while
 * the tab is hidden. Page-level trackers keep owning their own results.
 */
@Injectable({ providedIn: 'root' })
export class JobMonitorService {
  readonly jobs = signal<JobSummary[]>([]);
  readonly unavailable = signal(false);
  readonly activeJobs = computed(() =>
    this.jobs().filter((job) => job.status === 'pending' || job.status === 'running'),
  );
  readonly recentJobs = computed(() =>
    this.jobs().filter((job) => job.status !== 'pending' && job.status !== 'running'),
  );

  private readonly router = inject(Router);
  private timer: ReturnType<typeof setTimeout> | null = null;
  private inFlight = false;
  private failures = 0;
  private started = false;
  private panelOpen = false;

  constructor() {
    const destroyRef = inject(DestroyRef);
    const onVisibility = (): void => {
      if (!document.hidden) this.refresh();
    };
    document.addEventListener('visibilitychange', onVisibility);
    const navigation = this.router.events
      .pipe(filter((event) => event instanceof NavigationEnd))
      .subscribe(() => this.refresh());
    destroyRef.onDestroy(() => {
      document.removeEventListener('visibilitychange', onVisibility);
      navigation.unsubscribe();
      this.stopTimer();
    });
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    this.refresh();
  }

  setPanelOpen(open: boolean): void {
    this.panelOpen = open;
    if (open) this.refresh();
  }

  async cancel(jobId: string): Promise<void> {
    await cancelJob(jobId);
    this.refresh();
  }

  refresh(): void {
    if (!this.started || this.inFlight) return;
    this.stopTimer();
    void this.poll();
  }

  private async poll(): Promise<void> {
    this.inFlight = true;
    try {
      const response = await fetchJobs();
      this.jobs.set(response.jobs);
      this.unavailable.set(false);
      this.failures = 0;
    } catch {
      this.failures += 1;
      this.unavailable.set(true);
    } finally {
      this.inFlight = false;
      this.scheduleNext();
    }
  }

  private scheduleNext(): void {
    this.stopTimer();
    if (document.hidden) return;
    const base = this.activeJobs().length > 0 || this.panelOpen ? ACTIVE_POLL_MS : IDLE_POLL_MS;
    const delay = this.failures > 0
      ? Math.min(MAX_BACKOFF_MS, base * 2 ** this.failures)
      : base;
    this.timer = setTimeout(() => this.refresh(), delay);
  }

  private stopTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}
