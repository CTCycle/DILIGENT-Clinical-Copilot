// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { DecimalPipe, NgTemplateOutlet } from '@angular/common';
import { Component, ElementRef, HostListener, OnDestroy, effect, inject, signal } from '@angular/core';

import { JobMonitorService } from '../../core/services/job-monitor.service';
import { JobStatus, JobSummary } from '../../core/services/jobs-api';
import { NotificationService } from '../../core/services/notification.service';

const JOB_TYPE_INFO: Record<string, { name: string; kind: string }> = {
  clinical: { name: 'Clinical analysis', kind: 'Analysis' },
  ollama_pull: { name: 'Model download', kind: 'Download' },
  rxnav_update: { name: 'RxNav catalog update', kind: 'Knowledge update' },
  livertox_update: { name: 'LiverTox update', kind: 'Knowledge update' },
  terminology_update: { name: 'Terminology update', kind: 'Knowledge update' },
  dilirank_update: { name: 'DILIrank update', kind: 'Knowledge update' },
  structured_sources_update: { name: 'Structured sources update', kind: 'Knowledge update' },
  rag_update: { name: 'RAG embeddings update', kind: 'Indexing' },
  session_timeline: { name: 'Patient timeline', kind: 'Timeline' },
  session_revision: { name: 'Report revision', kind: 'Revision' },
};

const STATUS_LABEL: Record<JobStatus, string> = {
  pending: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

@Component({
  selector: 'app-jobs-monitor',
  standalone: true,
  imports: [DecimalPipe, NgTemplateOutlet],
  templateUrl: './jobs-monitor.component.html',
  styleUrl: './jobs-monitor.component.scss',
})
export class JobsMonitorComponent implements OnDestroy {
  readonly monitor = inject(JobMonitorService);
  private readonly notifications = inject(NotificationService);
  private readonly host = inject(ElementRef<HTMLElement>);

  readonly open = signal(false);
  readonly now = signal(Date.now());
  readonly cancelling = signal<ReadonlySet<string>>(new Set());
  private ticker: ReturnType<typeof setInterval> | null = null;

  constructor() {
    this.monitor.start();
    effect(() => {
      const shouldTick = this.open() && this.monitor.activeJobs().length > 0;
      if (shouldTick && this.ticker === null) {
        this.ticker = setInterval(() => this.now.set(Date.now()), 1000);
      } else if (!shouldTick) {
        this.clearTicker();
      }
    });
  }

  ngOnDestroy(): void {
    this.clearTicker();
  }

  toggle(): void {
    const next = !this.open();
    this.open.set(next);
    this.monitor.setPanelOpen(next);
    if (next) this.now.set(Date.now());
  }

  badge(): string {
    const count = this.monitor.activeJobs().length;
    return count > 9 ? '9+' : String(count);
  }

  name(job: JobSummary): string {
    return JOB_TYPE_INFO[job.job_type]?.name ?? job.job_type;
  }

  kind(job: JobSummary): string {
    return JOB_TYPE_INFO[job.job_type]?.kind ?? 'Job';
  }

  statusLabel(job: JobSummary): string {
    return job.stop_requested && job.status === 'running' ? 'Stopping' : STATUS_LABEL[job.status];
  }

  detail(job: JobSummary): string {
    if (job.status === 'failed') return job.error ?? 'The job failed.';
    return job.message ?? job.phase ?? '';
  }

  isActive(job: JobSummary): boolean {
    return job.status === 'pending' || job.status === 'running';
  }

  timing(job: JobSummary): string {
    const startMs = job.started_at * 1000;
    if (this.isActive(job)) {
      return `Elapsed ${this.duration(Math.max(0, this.now() - startMs))}`;
    }
    const finishedMs = (job.finished_at ?? job.started_at) * 1000;
    const time = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit' }).format(finishedMs);
    return `${this.duration(Math.max(0, finishedMs - startMs))} · finished ${time}`;
  }

  async cancel(job: JobSummary): Promise<void> {
    this.cancelling.update((ids) => new Set(ids).add(job.job_id));
    try {
      await this.monitor.cancel(job.job_id);
    } catch {
      this.notifications.notify(`Could not cancel ${this.name(job).toLowerCase()}.`, 'error');
    } finally {
      this.cancelling.update((ids) => {
        const next = new Set(ids);
        next.delete(job.job_id);
        return next;
      });
    }
  }

  private duration(ms: number): string {
    const total = Math.floor(ms / 1000);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  }

  private clearTicker(): void {
    if (this.ticker !== null) {
      clearInterval(this.ticker);
      this.ticker = null;
    }
  }

  @HostListener('document:click', ['$event'])
  closeOnOutsideClick(event: MouseEvent): void {
    const target = event.target;
    if (this.open() && target instanceof Node && !this.host.nativeElement.contains(target)) {
      this.open.set(false);
      this.monitor.setPanelOpen(false);
    }
  }

  @HostListener('document:keydown.escape')
  closeOnEscape(): void {
    if (!this.open()) return;
    this.open.set(false);
    this.monitor.setPanelOpen(false);
  }
}
