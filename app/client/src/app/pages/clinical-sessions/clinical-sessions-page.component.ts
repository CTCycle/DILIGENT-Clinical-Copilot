import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import {
  fetchClinicalSessionDetail,
  fetchClinicalSessionRevisionJobStatus,
  fetchInspectionSessions,
  generateInspectionSessionTimeline,
  startClinicalSessionRevisionJob,
  updateClinicalSession,
} from '../../core/services/inspection-api';
import {
  ClinicalSessionDetail,
  InspectionSessionItem,
  InspectionSessionStatus,
  JobStatus,
} from '../../core/models/types';
import { formatErrorMessage, formatUnknownError } from '../../core/utils';

@Component({
  selector: 'app-clinical-sessions-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './clinical-sessions-page.component.html',
  styleUrl: './clinical-sessions-page.component.scss',
})
export class ClinicalSessionsPageComponent implements OnInit, OnDestroy {
  @ViewChild('sessionTextEditor') private sessionTextEditor?: ElementRef<HTMLTextAreaElement>;

  private readonly router = inject(Router);
  private pollCancelled = false;

  readonly sessions = signal<InspectionSessionItem[]>([]);
  readonly selected = signal<ClinicalSessionDetail | null>(null);
  readonly loading = signal(false);
  readonly detailLoading = signal(false);
  readonly listError = signal<string | null>(null);
  readonly detailError = signal<string | null>(null);
  readonly query = signal('');
  readonly editorText = signal('');
  readonly metadataText = signal('{\n  "documents": [],\n  "images": []\n}');
  readonly revisionSelection = signal('');
  readonly revisionInstruction = signal('');
  readonly revisionModelOverrides = signal('{\n  "clinical_model": null,\n  "text_extraction_model": null,\n  "cloud_model": null\n}');
  readonly activeSection = signal<'preview' | 'editor' | 'metadata' | 'revision' | 'timeline'>('preview');
  readonly saveStatus = signal('');
  readonly revisionStatus = signal('');
  readonly revisionProgress = signal(0);
  readonly revisionJobStatus = signal<JobStatus | null>(null);

  ngOnInit(): void {
    void this.loadSessions();
  }

  ngOnDestroy(): void {
    this.stopPoller();
  }

  async loadSessions(): Promise<void> {
    this.loading.set(true);
    this.listError.set(null);
    try {
      const payload = await fetchInspectionSessions({
        search: this.query() || undefined,
        offset: 0,
        limit: 100,
      });
      this.sessions.set(payload.items);
      if (!this.selected() && payload.items[0]) {
        await this.openSession(payload.items[0].session_id);
      }
    } catch (error) {
      this.listError.set(formatUnknownError(error, 'Failed to load clinical sessions.'));
    } finally {
      this.loading.set(false);
    }
  }

  async openSession(sessionId: number): Promise<void> {
    this.detailLoading.set(true);
    this.detailError.set(null);
    try {
      const detail = await fetchClinicalSessionDetail(sessionId);
      this.selected.set(detail);
      this.editorText.set(detail.session_text || '');
      this.metadataText.set(JSON.stringify(detail.metadata || {}, null, 2));
      this.revisionSelection.set('');
      this.revisionInstruction.set('');
      this.activeSection.set('preview');
    } catch (error) {
      this.detailError.set(formatUnknownError(error, 'Failed to open session.'));
    } finally {
      this.detailLoading.set(false);
    }
  }

  updateQuery(value: string): void {
    this.query.set(value);
  }

  updateEditorText(value: string): void {
    this.editorText.set(value);
  }

  updateMetadataText(value: string): void {
    this.metadataText.set(value);
  }

  updateRevisionSelection(value: string): void {
    this.revisionSelection.set(value);
  }

  updateRevisionInstruction(value: string): void {
    this.revisionInstruction.set(value);
  }

  updateRevisionModelOverrides(value: string): void {
    this.revisionModelOverrides.set(value);
  }

  setSection(section: 'preview' | 'editor' | 'metadata' | 'revision' | 'timeline'): void {
    this.activeSection.set(section);
  }

  useEditorSelectionForRevision(): void {
    const element = this.sessionTextEditor?.nativeElement;
    if (!element) {
      this.activeSection.set('revision');
      return;
    }
    const selectedText = element.value.slice(element.selectionStart, element.selectionEnd).trim();
    if (selectedText) {
      this.revisionSelection.set(selectedText);
    }
    this.activeSection.set('revision');
  }

  async saveSession(): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    let metadata: Record<string, unknown>;
    try {
      metadata = JSON.parse(this.metadataText()) as Record<string, unknown>;
    } catch {
      this.saveStatus.set('[ERROR] Metadata must be valid JSON.');
      return;
    }
    this.saveStatus.set('Saving...');
    try {
      const updated = await updateClinicalSession(detail.session_id, {
        session_text: this.editorText(),
        metadata,
      });
      this.selected.set(updated);
      this.saveStatus.set('Saved.');
    } catch (error) {
      this.saveStatus.set(formatUnknownError(error, 'Failed to save session.'));
    }
  }

  async startRevision(): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    let overrides: Record<string, unknown>;
    let metadata: Record<string, unknown>;
    try {
      overrides = JSON.parse(this.revisionModelOverrides()) as Record<string, unknown>;
    } catch {
      this.revisionStatus.set('[ERROR] Model overrides must be valid JSON.');
      return;
    }
    try {
      metadata = JSON.parse(this.metadataText()) as Record<string, unknown>;
    } catch {
      this.revisionStatus.set('[ERROR] Metadata must be valid JSON before revision.');
      return;
    }
    this.stopPoller();
    this.revisionStatus.set('Starting revision...');
    this.revisionProgress.set(0);
    try {
      const started = await startClinicalSessionRevisionJob(detail.session_id, {
        selected_text: this.revisionSelection().trim() || null,
        revision_instruction: this.revisionInstruction().trim() || null,
        model_overrides: overrides,
        metadata: { ...metadata, revision_note: 'Manual revision mode' },
      });
      this.revisionJobStatus.set(started.status);
      void this.pollRevisionJob(started.job_id, Math.max(1000, started.poll_interval * 1000));
    } catch (error) {
      this.revisionStatus.set(formatUnknownError(error, 'Failed to start revision.'));
    }
  }

  private async pollRevisionJob(jobId: string, intervalMs: number): Promise<void> {
    this.pollCancelled = false;
    while (!this.pollCancelled) {
      try {
        const status = await fetchClinicalSessionRevisionJobStatus(jobId);
          this.revisionJobStatus.set(status.status);
          this.revisionProgress.set(status.progress);
          this.revisionStatus.set(
            typeof status.result?.progress_message === 'string'
              ? status.result.progress_message
              : `Revision ${status.status}`,
          );
          if (status.status === 'completed') {
            this.stopPoller();
            await this.loadSessions();
            const revisedSessionId = this.revisedSessionId(status.result);
            if (revisedSessionId !== null) {
              await this.openSession(revisedSessionId);
              this.activeSection.set('preview');
            }
            return;
          }
          if (status.status === 'failed') {
            this.stopPoller();
            this.revisionStatus.set(formatErrorMessage(status.error || 'Revision failed.'));
            return;
          }
      } catch (error) {
        this.stopPoller();
        this.revisionStatus.set(formatUnknownError(error, 'Revision polling failed.'));
        return;
      }
      await new Promise((resolve) => globalThis.setTimeout(resolve, intervalMs));
    }
  }

  async createTimeline(): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    this.saveStatus.set('Creating timeline...');
    try {
      await generateInspectionSessionTimeline(detail.session_id, { force_regenerate: true });
      await this.router.navigate(['/sessions', detail.session_id, 'timetable']);
    } catch (error) {
      this.saveStatus.set(formatUnknownError(error, 'Failed to create timeline.'));
    }
  }

  statusLabel(value: InspectionSessionStatus): string {
    return value === 'failed' ? 'Failed' : 'Successful';
  }

  previewReport(detail: ClinicalSessionDetail): string {
    const report = detail.report || detail.result_payload?.['report'];
    return typeof report === 'string' && report.trim() ? report.trim() : 'No AI report preview is available for this session.';
  }

  previewDetectedDrugs(detail: ClinicalSessionDetail): string[] {
    const detected = detail.result_payload?.['detected_drugs'];
    return Array.isArray(detected)
      ? detected.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [];
  }

  previewRevisionAudit(detail: ClinicalSessionDetail): Record<string, unknown> | null {
    const audit = detail.result_payload?.['revision_audit'];
    return audit && typeof audit === 'object' && !Array.isArray(audit)
      ? audit as Record<string, unknown>
      : null;
  }

  metadataEntries(key: 'documents' | 'images'): string[] {
    try {
      const parsed = JSON.parse(this.metadataText()) as Record<string, unknown>;
      const values = parsed[key];
      if (!Array.isArray(values)) return [];
      return values
        .map((item) => {
          if (typeof item === 'string') return item.trim();
          if (!item || typeof item !== 'object') return '';
          const record = item as Record<string, unknown>;
          const label = record['title'] || record['file_name'] || record['name'] || record['path'] || record['source'];
          return typeof label === 'string' ? label.trim() : JSON.stringify(record);
        })
        .filter((item) => item.length > 0);
    } catch {
      return [];
    }
  }

  private revisedSessionId(result: Record<string, unknown> | null): number | null {
    const value = result?.['session_id'];
    return typeof value === 'number' && Number.isInteger(value) ? value : null;
  }

  private stopPoller(): void {
    this.pollCancelled = true;
  }
}
