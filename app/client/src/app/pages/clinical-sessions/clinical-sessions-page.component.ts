import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import {
  fetchClinicalSessionDetail,
  fetchClinicalSessionRevisionJobStatus,
  fetchInspectionLiverToxCatalog,
  fetchInspectionRxNavCatalog,
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

type DetectedDrugEvidence = {
  name: string;
  liverTox: boolean;
  rxNav: boolean;
  inAnamnesis: boolean;
  inTherapy: boolean;
};

@Component({
  selector: 'app-clinical-sessions-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './clinical-sessions-page.component.html',
  styleUrl: './clinical-sessions-page.component.scss',
})
export class ClinicalSessionsPageComponent implements OnInit, OnDestroy {
  @ViewChild('sessionTextEditor') private sessionTextEditor?: ElementRef<HTMLDivElement>;

  private readonly router = inject(Router);
  private pollCancelled = false;

  readonly sessions = signal<InspectionSessionItem[]>([]);
  readonly statusFilter = signal<'all' | InspectionSessionStatus>('all');
  readonly dateFilterMode = signal<'any' | 'after' | 'before' | 'exact'>('any');
  readonly dateFilter = signal('');
  readonly filteredSessions = computed(() => {
    const status = this.statusFilter();
    const dateMode = this.dateFilterMode();
    const dateFilter = this.dateFilter();
    const filtered = this.sessions().filter((session) => {
      if (status !== 'all' && session.status !== status) return false;
      if (dateMode === 'any' || !dateFilter) return true;
      const sessionDate = this.dateKey(session.session_timestamp);
      if (!sessionDate) return false;
      if (dateMode === 'after') return sessionDate > dateFilter;
      if (dateMode === 'before') return sessionDate < dateFilter;
      return sessionDate === dateFilter;
    });
    return [...filtered].sort((left, right) => {
      const leftTime = Date.parse(left.session_timestamp || '') || 0;
      const rightTime = Date.parse(right.session_timestamp || '') || 0;
      return rightTime - leftTime;
    });
  });
  readonly selected = signal<ClinicalSessionDetail | null>(null);
  readonly loading = signal(false);
  readonly detailLoading = signal(false);
  readonly listError = signal<string | null>(null);
  readonly detailError = signal<string | null>(null);
  readonly query = signal('');
  readonly editorText = signal('');
  readonly editorFontSize = signal(16);
  readonly metadataText = signal('{\n  "documents": [],\n  "images": []\n}');
  readonly revisionSelection = signal('');
  readonly revisionInstruction = signal('');
  readonly revisionModelOverrides = signal('{\n  "clinical_model": null,\n  "text_extraction_model": null,\n  "cloud_model": null\n}');
  readonly activeSection = signal<'preview' | 'editor' | 'metadata' | 'revision' | 'timeline'>('preview');
  readonly saveStatus = signal('');
  readonly revisionStatus = signal('');
  readonly revisionProgress = signal(0);
  readonly revisionJobStatus = signal<JobStatus | null>(null);
  readonly detectedDrugEvidence = signal<DetectedDrugEvidence[]>([]);
  readonly detectedDiseases = signal<string[]>([]);
  readonly labSummary = signal<Array<{ label: string; value: string }>>([]);
  readonly hepatotoxicityPattern = signal<string>('N/A');

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
      this.editorText.set(this.toEditorHtml(this.previewReport(detail)));
      this.metadataText.set(JSON.stringify(detail.metadata || {}, null, 2));
      this.revisionSelection.set('');
      this.revisionInstruction.set('');
      this.activeSection.set('preview');
      this.detectedDiseases.set(this.previewDetectedDiseases(detail));
      this.labSummary.set(this.previewLaboratorySummary(detail));
      this.hepatotoxicityPattern.set(this.previewHepatotoxicityPattern(detail));
      void this.loadDetectedDrugEvidence(detail);
    } catch (error) {
      this.detailError.set(formatUnknownError(error, 'Failed to open session.'));
    } finally {
      this.detailLoading.set(false);
    }
  }

  updateQuery(value: string): void {
    this.query.set(value);
  }

  updateStatusFilter(value: 'all' | InspectionSessionStatus): void {
    this.statusFilter.set(value);
  }

  updateDateFilterMode(value: 'any' | 'after' | 'before' | 'exact'): void {
    this.dateFilterMode.set(value);
    if (value === 'any') {
      this.dateFilter.set('');
    }
  }

  updateDateFilter(value: string): void {
    this.dateFilter.set(value);
  }

  updateEditorText(value: string): void {
    this.editorText.set(value);
  }

  setEditorFontSize(delta: number): void {
    const next = Math.min(22, Math.max(12, this.editorFontSize() + delta));
    this.editorFontSize.set(next);
  }

  runEditorCommand(command: string, value?: string): void {
    const element = this.sessionTextEditor?.nativeElement;
    if (!element) return;
    element.focus();
    document.execCommand(command, false, value);
    this.editorText.set(element.innerHTML);
  }

  onEditorInput(event: Event): void {
    const target = event.target as HTMLDivElement | null;
    if (!target) return;
    this.editorText.set(target.innerHTML);
  }

  onEditorKeydown(event: KeyboardEvent): void {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      void this.saveSession();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
      event.preventDefault();
      this.runEditorCommand('redo');
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'z') {
      event.preventDefault();
      this.runEditorCommand('redo');
    }
  }

  insertLink(): void {
    const url = globalThis.prompt('Enter URL');
    if (!url) return;
    this.runEditorCommand('createLink', url);
  }

  removeSelection(): void {
    const element = this.sessionTextEditor?.nativeElement;
    if (!element) return;
    element.focus();
    document.execCommand('delete');
    this.editorText.set(element.innerHTML);
  }

  clearFormatting(): void {
    this.runEditorCommand('removeFormat');
    this.runEditorCommand('unlink');
  }

  private toEditorHtml(text: string): string {
    const escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    return escaped.replace(/\r?\n/g, '<br>');
  }

  private editorValueToPersistedText(): string {
    const element = this.sessionTextEditor?.nativeElement;
    if (!element) return this.editorText();
    return element.innerHTML.trim();
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
    const persistedEditorValue = this.editorValueToPersistedText();
    try {
      const updated = await updateClinicalSession(detail.session_id, {
        session_text: persistedEditorValue,
        metadata,
      });
      this.selected.set({
        ...updated,
        report: persistedEditorValue,
        result_payload: {
          ...updated.result_payload,
          report: persistedEditorValue,
        },
      });
      this.editorText.set(persistedEditorValue);
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

  private async loadDetectedDrugEvidence(detail: ClinicalSessionDetail): Promise<void> {
    const drugs = this.previewDetectedDrugs(detail);
    const sections = this.sectionTextMap(detail);
    this.detectedDrugEvidence.set(drugs.map((name) => ({
      name,
      liverTox: false,
      rxNav: false,
      inAnamnesis: this.textContainsDrug(sections.anamnesis, name),
      inTherapy: this.textContainsDrug(sections.therapy, name),
    })));
    if (!drugs.length) return;

    const rows = await Promise.all(drugs.map(async (name): Promise<DetectedDrugEvidence> => {
      const [rxNav, liverTox] = await Promise.all([
        this.catalogHasDrug('rxnav', name),
        this.catalogHasDrug('livertox', name),
      ]);
      return {
        name,
        rxNav,
        liverTox,
        inAnamnesis: this.textContainsDrug(sections.anamnesis, name),
        inTherapy: this.textContainsDrug(sections.therapy, name),
      };
    }));
    if (this.selected()?.session_id === detail.session_id) {
      this.detectedDrugEvidence.set(rows);
    }
  }

  private async catalogHasDrug(source: 'rxnav' | 'livertox', name: string): Promise<boolean> {
    const normalized = this.normalizeDrugName(name);
    const search = normalized || name;
    try {
      const payload = source === 'rxnav'
        ? await fetchInspectionRxNavCatalog({ search, offset: 0, limit: 5 })
        : await fetchInspectionLiverToxCatalog({ search, offset: 0, limit: 5 });
      return payload.items.some((item) => this.normalizeDrugName(item.drug_name) === normalized);
    } catch {
      return false;
    }
  }

  private normalizeDrugName(value: string): string {
    return value.toLowerCase().replace(/\([^)]*\)/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
  }

  private sectionTextMap(detail: ClinicalSessionDetail): { anamnesis: string; therapy: string } {
    const sections = detail.sections || {};
    const anamnesis = typeof sections['anamnesis'] === 'string' ? sections['anamnesis'] : '';
    const therapy = typeof sections['therapy'] === 'string' ? sections['therapy'] : '';
    return { anamnesis, therapy };
  }

  private textContainsDrug(text: string, drug: string): boolean {
    if (!text.trim() || !drug.trim()) return false;
    const normalizedText = this.normalizeDrugName(text);
    const normalizedDrug = this.normalizeDrugName(drug);
    if (!normalizedDrug) return false;
    if (normalizedText.includes(normalizedDrug)) return true;
    const firstToken = normalizedDrug.split(' ')[0] || '';
    return firstToken.length > 3 && normalizedText.includes(firstToken);
  }

  private previewDetectedDiseases(detail: ClinicalSessionDetail): string[] {
    const fromPayload = detail.result_payload?.['detected_diseases'];
    if (Array.isArray(fromPayload)) {
      const diseases = fromPayload
        .filter((item): item is string => typeof item === 'string')
        .map((item) => item.trim())
        .filter((item) => item.length > 0);
      if (diseases.length) return diseases;
    }
    const report = this.previewReport(detail);
    const lines = report.split(/\r?\n/);
    const diseaseLine = lines.find((line) => /detected diseases?/i.test(line));
    if (!diseaseLine) return [];
    return diseaseLine
      .split(':')
      .slice(1)
      .join(':')
      .split(',')
      .map((item) => item.replace(/[*-]/g, '').trim())
      .filter((item) => item.length > 0);
  }

  private previewLaboratorySummary(detail: ClinicalSessionDetail): Array<{ label: string; value: string }> {
    const payload = detail.result_payload || {};
    const flatPayload = this.flattenPayload(payload);
    const fromPayload = this.collectLabValues(flatPayload);
    if (fromPayload.length) return fromPayload;

    const report = this.previewReport(detail).replace(/[*_`]/g, '');
    const regexMap: Array<{ label: string; regex: RegExp }> = [
      { label: 'ALT', regex: /\bALT\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z/%µμ\.]*\/?[A-Za-z]*)/i },
      { label: 'AST', regex: /\bAST\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z/%µμ\.]*\/?[A-Za-z]*)/i },
      { label: 'ALP', regex: /\bALP\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z/%µμ\.]*\/?[A-Za-z]*)/i },
      { label: 'Bilirubin', regex: /\b(?:total\s+)?bilirubin\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z/%µμ\.]*\/?[A-Za-z]*)/i },
      { label: 'INR', regex: /\bINR\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)/i },
      { label: 'R-score', regex: /\bR-?score\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)/i },
    ];
    return regexMap
      .map(({ label, regex }) => {
        const match = report.match(regex);
        return match?.[1] ? { label, value: match[1].trim() } : null;
      })
      .filter((item): item is { label: string; value: string } => item !== null);
  }

  private previewHepatotoxicityPattern(detail: ClinicalSessionDetail): string {
    const payload = detail.result_payload || {};
    const flatPayload = this.flattenPayload(payload);
    const fromPayload = [
      flatPayload['hepatotoxicity_pattern'],
      flatPayload['pattern_classification'],
      flatPayload['classification'],
      flatPayload['hepatotoxicity.classification'],
    ].find((value) => typeof value === 'string' && value.trim().length > 0);
    if (typeof fromPayload === 'string') return fromPayload.trim();

    const report = this.previewReport(detail).replace(/[*_`]/g, '');
    const patternMatch = report.match(/\b(?:hepatotoxicity pattern|classification)\b\s*[:=]\s*([A-Za-z -]+)/i);
    return patternMatch?.[1]?.trim() || 'N/A';
  }

  private flattenPayload(
    value: unknown,
    prefix = '',
    acc: Record<string, unknown> = {},
  ): Record<string, unknown> {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return acc;
    const record = value as Record<string, unknown>;
    for (const [key, nested] of Object.entries(record)) {
      const fullKey = prefix ? `${prefix}.${key}` : key;
      acc[fullKey.toLowerCase()] = nested;
      if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
        this.flattenPayload(nested, fullKey, acc);
      }
    }
    return acc;
  }

  private collectLabValues(flatPayload: Record<string, unknown>): Array<{ label: string; value: string }> {
    const keys: Array<{ label: string; includes: string[] }> = [
      { label: 'ALT', includes: ['alt'] },
      { label: 'AST', includes: ['ast'] },
      { label: 'ALP', includes: ['alp', 'alkaline_phosphatase'] },
      { label: 'Bilirubin', includes: ['bilirubin', 'tbil'] },
      { label: 'INR', includes: ['inr'] },
      { label: 'R-score', includes: ['r_score', 'rscore', 'r-score'] },
    ];

    return keys
      .map(({ label, includes }) => {
        const payloadEntry = Object.entries(flatPayload).find(([key, val]) =>
          includes.some((needle) => key.includes(needle)) &&
          val !== null &&
          val !== undefined &&
          String(val).trim().length > 0,
        );
        if (!payloadEntry) return null;
        return { label, value: String(payloadEntry[1]).trim() };
      })
      .filter((item): item is { label: string; value: string } => item !== null);
  }

  private dateKey(value: string | null): string {
    if (!value) return '';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return '';
    return parsed.toISOString().slice(0, 10);
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
