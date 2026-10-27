// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { computed, signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';

import { JobMonitorService } from '../../core/services/job-monitor.service';
import { JobSummary } from '../../core/services/jobs-api';
import { JobsMonitorComponent } from './jobs-monitor.component';

function job(overrides: Partial<JobSummary>): JobSummary {
  return {
    job_id: 'job-1',
    job_type: 'rag_update',
    status: 'running',
    progress: null,
    phase: null,
    message: null,
    error: null,
    started_at: Date.now() / 1000 - 30,
    finished_at: null,
    stop_requested: false,
    cancellable: true,
    ...overrides,
  };
}

describe('JobsMonitorComponent', () => {
  const jobs = signal<JobSummary[]>([]);
  const monitor = {
    jobs,
    unavailable: signal(false),
    activeJobs: computed(() => jobs().filter((item) => item.status === 'running' || item.status === 'pending')),
    recentJobs: computed(() => jobs().filter((item) => item.status !== 'running' && item.status !== 'pending')),
    start: vi.fn(),
    setPanelOpen: vi.fn(),
    cancel: vi.fn().mockResolvedValue(undefined),
  };

  function render(items: JobSummary[]): HTMLElement {
    jobs.set(items);
    const fixture = TestBed.createComponent(JobsMonitorComponent);
    fixture.componentInstance.open.set(true);
    fixture.detectChanges();
    return fixture.nativeElement as HTMLElement;
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [{ provide: JobMonitorService, useValue: monitor }],
    });
  });

  it('shows an empty state when there are no jobs', () => {
    const element = render([]);

    expect(element.querySelector('.jobs-monitor-badge')).toBeNull();
    expect(element.textContent).toContain('No background jobs yet');
  });

  it('uses an indeterminate bar unless progress was measured', () => {
    const element = render([
      job({ job_id: 'a', progress: null }),
      job({ job_id: 'b', progress: 42, job_type: 'ollama_pull' }),
    ]);

    expect(element.querySelector('.jobs-monitor-badge')?.textContent).toBe('2');
    expect(element.querySelectorAll('.jobs-monitor-bar.is-indeterminate')).toHaveLength(1);
    expect(element.querySelector('[aria-valuenow="42"]')).not.toBeNull();
  });

  it('shows failure details and cancel only where it is allowed', () => {
    const element = render([
      job({ job_id: 'run', cancellable: true }),
      job({ job_id: 'bad', status: 'failed', error: 'Download timed out.', cancellable: false }),
    ]);

    expect(element.textContent).toContain('Download timed out.');
    expect(element.querySelectorAll('.jobs-monitor-cancel')).toHaveLength(1);
  });
});
