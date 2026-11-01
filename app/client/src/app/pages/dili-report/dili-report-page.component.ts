// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, DestroyRef, computed, effect, inject, signal, untracked } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { LucideCheck, LucideCopy, LucideDownload } from '@lucide/angular';
import { map } from 'rxjs';

import { AnalysisProgressComponent } from '../../components/analysis-progress/analysis-progress.component';
import { ClinicalReportViewerComponent } from '../../components/clinical-report-viewer/clinical-report-viewer.component';
import { REPORT_EXPORT_FILENAME } from '../../core/constants';
import { describeEvidenceSources } from '../../core/evidence-sources';
import { ClinicalSessionDetail } from '../../core/models/inspection-types';
import { ClinicalSessionInput, ClinicalSessionState } from '../../core/models/types';
import {
  cancelClinicalSession,
  fetchClinicalSessionInput,
  fetchClinicalSessionStatus,
} from '../../core/services/clinical-session-lifecycle-api';
import { ClinicalSessionPoller } from '../../core/services/clinical-session-poller';
import { fetchClinicalSessionDetail } from '../../core/services/clinical-sessions-api';
import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';
import { createDownloadUrl, formatUnknownError, isRecord } from '../../core/utils';
import { isActiveSessionStatus, sessionStatusLabel, sessionStatusTone } from '../../core/utils/session-status';
import { previewReport } from '../clinical-sessions/clinical-session-preview';

type CancelStep = 'idle' | 'confirming' | 'requesting';

function parseSessionId(value: string | null): number | null {
  if (value === null) {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

@Component({
  selector: 'app-dili-report-page',
  imports: [RouterLink, LucideCheck, LucideCopy, LucideDownload, AnalysisProgressComponent, ClinicalReportViewerComponent],
  templateUrl: './dili-report-page.component.html',
  styleUrl: './dili-report-page.component.scss',
})
export class DiliReportPageComponent {
  private readonly route = inject(ActivatedRoute);
  private readonly poller = inject(ClinicalSessionPoller);
  private readonly markdownRenderer = inject(MarkdownRendererService);

  readonly sessionId = toSignal(
    this.route.paramMap.pipe(map((params) => parseSessionId(params.get('sessionId')))),
    { initialValue: parseSessionId(this.route.snapshot.paramMap.get('sessionId')) },
  );
  private loadGeneration = 0;
  private stopPolling: (() => void) | null = null;

  readonly state = signal<ClinicalSessionState | null>(null);
  readonly detail = signal<ClinicalSessionDetail | null>(null);
  readonly input = signal<ClinicalSessionInput | null>(null);
  readonly loadError = signal<string | null>(null);
  readonly detailError = signal<string | null>(null);
  readonly cancelStep = signal<CancelStep>('idle');
  readonly cancelError = signal<string | null>(null);
  readonly copied = signal(false);
  private copiedTimer: ReturnType<typeof setTimeout> | null = null;

  readonly status = computed(() => this.state()?.status ?? null);
  readonly statusLabel = computed(() => {
    const status = this.status();
    return status ? sessionStatusLabel(status) : '';
  });
  readonly statusTone = computed(() => {
    const status = this.status();
    return status ? sessionStatusTone(status) : 'neutral';
  });
  readonly isActive = computed(() => {
    const status = this.status();
    return status !== null && isActiveSessionStatus(status);
  });
  readonly stopRequested = computed(() => this.isActive() && Boolean(this.state()?.stop_requested));
  readonly title = computed(() => {
    const name = (this.state()?.patient_name ?? this.detail()?.patient_name)?.trim();
    return name || `Session #${this.sessionId()}`;
  });
  readonly visitDate = computed(() => this.state()?.visit_date ?? this.detail()?.visit_date ?? null);
  readonly reportMarkdown = computed(() => {
    const detail = this.detail();
    return detail && this.state()?.has_report ? previewReport(detail) : '';
  });
  readonly progressStartedAtMs = computed(() => {
    const started = this.state()?.started_at;
    return started ? Date.parse(started) : null;
  });
  readonly progressUpdatedAtMs = computed(() => {
    const updated = this.state()?.updated_at;
    return updated ? Date.parse(updated) : null;
  });
  readonly isInterrupted = computed(() => this.state()?.error?.code === 'interrupted');
  readonly evidenceSources = computed(() => {
    const sources = this.detail()?.metadata['evidence_sources'];
    return Array.isArray(sources)
      ? describeEvidenceSources(sources.filter((item): item is string => typeof item === 'string'))
      : null;
  });
  readonly models = computed(() => {
    const detail = this.detail();
    if (!detail) return [];
    const overrides = detail.metadata['model_overrides'];
    const overridden = isRecord(overrides) ? overrides : {};
    return [
      { role: 'Text extraction', name: detail.text_extraction_model, overridden: Boolean(overridden['text_extraction_model']) },
      { role: 'Clinical', name: detail.clinical_model, overridden: Boolean(overridden['clinical_model']) },
    ].filter((model) => model.name);
  });

  constructor() {
    effect(() => {
      const sessionId = this.sessionId();
      untracked(() => this.open(sessionId));
    });
    inject(DestroyRef).onDestroy(() => {
      this.loadGeneration += 1;
      this.stopPolling?.();
    });
  }

  requestCancel(): void {
    this.cancelError.set(null);
    this.cancelStep.set('confirming');
  }

  dismissCancel(): void {
    this.cancelStep.set('idle');
  }

  async confirmCancel(): Promise<void> {
    const sessionId = this.sessionId();
    if (sessionId === null || this.cancelStep() === 'requesting') return;
    const generation = this.loadGeneration;
    this.cancelStep.set('requesting');
    this.cancelError.set(null);
    try {
      const next = await cancelClinicalSession(sessionId);
      this.applyState(next, generation);
    } catch (error) {
      if (generation !== this.loadGeneration) return;
      this.cancelError.set(formatUnknownError(error, 'Cancellation could not be requested.'));
      // A 409 means the session already finished; pick up its real state.
      try {
        this.applyState(await fetchClinicalSessionStatus(sessionId), generation);
      } catch {
        // Keep the cancellation error; polling continues.
      }
    } finally {
      if (generation === this.loadGeneration) this.cancelStep.set('idle');
    }
  }

  async copyReport(): Promise<void> {
    const rendered = this.markdownRenderer.render(this.reportMarkdown());
    if (!rendered.text || !navigator.clipboard) return;
    const clipboardItemCtor = (globalThis as { ClipboardItem?: typeof ClipboardItem }).ClipboardItem;
    if (clipboardItemCtor) {
      await navigator.clipboard.write([
        new clipboardItemCtor({
          'text/html': new Blob([rendered.html], { type: 'text/html' }),
          'text/plain': new Blob([rendered.text], { type: 'text/plain' }),
        }),
      ]);
    } else {
      await navigator.clipboard.writeText(rendered.text);
    }
    this.copied.set(true);
    if (this.copiedTimer) clearTimeout(this.copiedTimer);
    this.copiedTimer = setTimeout(() => this.copied.set(false), 2000);
  }

  downloadReport(): void {
    const markdown = this.reportMarkdown();
    if (!markdown) return;
    const url = createDownloadUrl(markdown, REPORT_EXPORT_FILENAME);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = REPORT_EXPORT_FILENAME;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  private open(sessionId: number | null): void {
    const generation = ++this.loadGeneration;
    this.stopPolling?.();
    this.stopPolling = null;
    this.state.set(null);
    this.detail.set(null);
    this.input.set(null);
    this.loadError.set(null);
    this.detailError.set(null);
    this.cancelStep.set('idle');
    this.cancelError.set(null);
    if (sessionId === null) {
      this.loadError.set('Invalid session id.');
      return;
    }
    this.stopPolling = this.poller.watch(
      sessionId,
      (next) => this.applyState(next, generation),
      (error) => {
        if (generation === this.loadGeneration && this.state() === null) {
          this.loadError.set(formatUnknownError(error, 'Failed to load the session.'));
        }
      },
    );
    void this.loadInput(sessionId, generation);
  }

  private applyState(next: ClinicalSessionState, generation: number): void {
    if (generation !== this.loadGeneration) return;
    const previous = this.state();
    if (previous && next.version < previous.version) return;
    this.state.set(next);
    this.loadError.set(null);
    const settled = !isActiveSessionStatus(next.status);
    const changed = !previous || previous.status !== next.status || previous.has_report !== next.has_report;
    if (settled && changed && next.status !== 'draft') {
      void this.loadDetail(next.session_id, generation);
    }
  }

  private async loadDetail(sessionId: number, generation: number): Promise<void> {
    this.detailError.set(null);
    try {
      const detail = await fetchClinicalSessionDetail(sessionId);
      if (generation === this.loadGeneration) this.detail.set(detail);
    } catch (error) {
      if (generation === this.loadGeneration) {
        this.detailError.set(formatUnknownError(error, 'Failed to load the report.'));
      }
    }
  }

  private async loadInput(sessionId: number, generation: number): Promise<void> {
    try {
      const input = await fetchClinicalSessionInput(sessionId);
      if (generation === this.loadGeneration) this.input.set(input);
    } catch {
      // The input section is supplementary; the report remains usable without it.
    }
  }
}
