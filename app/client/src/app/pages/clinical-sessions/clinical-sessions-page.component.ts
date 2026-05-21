import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import {
  LucideBookOpen,
  LucideBraces,
  LucideFileText,
  LucideImage,
} from '@lucide/angular';

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
import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';
import { formatErrorMessage, formatUnknownError } from '../../core/utils';

type DetectedDrugEvidence = {
  name: string;
  liverTox: boolean;
  rxNav: boolean;
  inAnamnesis: boolean;
  inTherapy: boolean;
};

type LabTimelineRow = {
  marker: string;
  value: string;
  unit: string;
  upperLimit: string;
  timing: string;
  source: string;
  evidence: string;
};

type DrugEvidenceDraft = DetectedDrugEvidence & {
  hasPersistedMatch: boolean;
};

@Component({
  selector: 'app-clinical-sessions-page',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideBookOpen, LucideBraces, LucideFileText, LucideImage],
  templateUrl: './clinical-sessions-page.component.html',
  styleUrl: './clinical-sessions-page.component.scss',
})
export class ClinicalSessionsPageComponent implements OnInit, OnDestroy {
  @ViewChild('sessionTextEditor') private sessionTextEditor?: ElementRef<HTMLDivElement>;

  private readonly router = inject(Router);
  private readonly markdownRenderer = inject(MarkdownRendererService);
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
  readonly editorViewMode = signal<'source' | 'rendered'>('source');
  readonly editorFontSize = signal(16);
  readonly metadataText = signal('{\n  "documents": [],\n  "images": []\n}');
  readonly revisionSelection = signal('');
  readonly revisionInstruction = signal('');
  readonly revisionModelProvider = signal<'local' | 'cloud'>('local');
  readonly revisionClinicalModel = signal('');
  readonly revisionTextParsingModel = signal('');
  readonly revisionRagSearch = signal(false);
  readonly activeSection = signal<'preview' | 'editor' | 'metadata' | 'revision' | 'timeline'>('preview');
  readonly saveStatus = signal('');
  readonly revisionStatus = signal('');
  readonly revisionProgress = signal(0);
  readonly revisionJobStatus = signal<JobStatus | null>(null);
  readonly detectedDrugEvidence = signal<DetectedDrugEvidence[]>([]);
  readonly detectedDiseases = signal<string[]>([]);
  readonly labSummary = signal<Array<{ label: string; value: string }>>([]);
  readonly labTimeline = signal<LabTimelineRow[]>([]);
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
      this.editorViewMode.set('source');
      this.metadataText.set(JSON.stringify(this.normalizeMetadata(detail.metadata || {}), null, 2));
      this.revisionSelection.set('');
      this.revisionInstruction.set('');
      this.revisionModelProvider.set(detail.result_payload?.['cloud_model'] ? 'cloud' : 'local');
      this.revisionClinicalModel.set(detail.clinical_model || '');
      this.revisionTextParsingModel.set(detail.text_extraction_model || '');
      this.revisionRagSearch.set(Boolean(detail.metadata?.['use_rag']));
      this.activeSection.set('preview');
      this.detectedDiseases.set(this.previewDetectedDiseases(detail));
      this.labSummary.set(this.previewLaboratorySummary(detail));
      this.labTimeline.set(this.previewLabTimeline(detail));
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

  setEditorViewMode(mode: 'source' | 'rendered'): void {
    if (this.editorViewMode() === mode) return;
    this.editorViewMode.set(mode);
    const detail = this.selected();
    const sourceText = detail ? this.previewReport(detail) : this.editorText();
    this.editorText.set(mode === 'rendered' ? this.markdownRenderer.render(sourceText).html : this.toEditorHtml(sourceText));
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

  setSection(section: 'preview' | 'editor' | 'metadata' | 'revision' | 'timeline'): void {
    this.activeSection.set(section);
  }

  updateRevisionModelProvider(value: 'local' | 'cloud'): void {
    this.revisionModelProvider.set(value);
  }

  updateRevisionClinicalModel(value: string): void {
    this.revisionClinicalModel.set(value);
  }

  updateRevisionTextParsingModel(value: string): void {
    this.revisionTextParsingModel.set(value);
  }

  updateRevisionRagSearch(value: boolean): void {
    this.revisionRagSearch.set(value);
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
    let metadata: Record<string, unknown>;
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
        model_overrides: this.revisionModelOverrides(),
        metadata: {
          ...metadata,
          use_rag: this.revisionRagSearch(),
          revision_note: 'Manual revision mode',
        },
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

  previewReportHtml(detail: ClinicalSessionDetail): string {
    return this.markdownRenderer.render(this.previewReport(detail)).html;
  }

  previewDetectedDrugs(detail: ClinicalSessionDetail): string[] {
    const detected = detail.result_payload?.['detected_drugs'];
    return Array.isArray(detected)
      ? detected.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [];
  }

  private async loadDetectedDrugEvidence(detail: ClinicalSessionDetail): Promise<void> {
    const rows = this.buildPersistedDrugEvidence(detail);
    if (this.selected()?.session_id === detail.session_id) {
      this.detectedDrugEvidence.set(rows.map(({ hasPersistedMatch: _hasPersistedMatch, ...row }) => row));
    }
    if (!rows.length) return;

    const needsFallback = rows.filter((row) => !row.hasPersistedMatch || !row.rxNav);
    if (!needsFallback.length) return;

    const fallbackByName = new Map<string, Partial<DetectedDrugEvidence>>();
    await Promise.all(needsFallback.map(async (row) => {
      const [rxNav, liverTox] = await Promise.all([
        row.rxNav ? Promise.resolve(true) : this.catalogHasDrug('rxnav', row.name),
        row.liverTox || row.hasPersistedMatch ? Promise.resolve(row.liverTox) : this.catalogHasDrug('livertox', row.name),
      ]);
      fallbackByName.set(row.name, { rxNav, liverTox });
    }));
    if (this.selected()?.session_id === detail.session_id) {
      this.detectedDrugEvidence.set(rows.map(({ hasPersistedMatch: _hasPersistedMatch, ...row }) => ({
        ...row,
        rxNav: fallbackByName.get(row.name)?.rxNav ?? row.rxNav,
        liverTox: fallbackByName.get(row.name)?.liverTox ?? row.liverTox,
      })));
    }
  }

  private buildPersistedDrugEvidence(detail: ClinicalSessionDetail): DrugEvidenceDraft[] {
    const rows = new Map<string, DrugEvidenceDraft>();
    const sections = this.sectionTextMap(detail);
    const ensureRow = (name: string): DrugEvidenceDraft => {
      const normalized = this.normalizeDrugName(name);
      const key = normalized || name.trim().toLowerCase();
      const existing = rows.get(key);
      if (existing) return existing;
      const next: DrugEvidenceDraft = {
        name,
        liverTox: false,
        rxNav: false,
        inAnamnesis: this.textContainsDrug(sections.anamnesis, name),
        inTherapy: this.textContainsDrug(sections.therapy, name),
        hasPersistedMatch: false,
      };
      rows.set(key, next);
      return next;
    };

    for (const name of this.previewDetectedDrugs(detail)) {
      ensureRow(name);
    }

    const structuredCase = this.recordValue(detail.result_payload?.['structured_case']);
    const therapyDrugs = this.arrayValue(structuredCase?.['therapy_drugs']);
    const anamnesisDrugs = [
      ...this.arrayValue(structuredCase?.['anamnesis_drugs']),
      ...this.arrayValue(detail.result_payload?.['anamnesis_drugs']),
    ];
    for (const item of therapyDrugs) {
      const name = this.drugNameFromUnknown(item);
      if (!name) continue;
      ensureRow(name).inTherapy = true;
    }
    for (const item of anamnesisDrugs) {
      const name = this.drugNameFromUnknown(item);
      if (!name) continue;
      ensureRow(name).inAnamnesis = true;
    }

    for (const item of this.arrayValue(detail.result_payload?.['matched_drugs'])) {
      const record = this.recordValue(item);
      if (!record) continue;
      const name = this.stringValue(record['raw_drug_name'])
        || this.stringValue(record['drug_name'])
        || this.stringValue(record['matched_drug_name']);
      if (!name) continue;
      const row = ensureRow(name);
      const matchedName = this.stringValue(record['matched_drug_name']);
      row.hasPersistedMatch = true;
      row.name = row.name || matchedName || name;
      row.liverTox = row.liverTox || this.hasLiverToxEvidence(record);
      row.rxNav = row.rxNav || this.hasRxNavEvidence(record);
      row.inTherapy = row.inTherapy || this.originsContain(record, 'therapy') || this.rawMentionsContain(record, sections.therapy);
      row.inAnamnesis = row.inAnamnesis || this.originsContain(record, 'anamnesis') || this.rawMentionsContain(record, sections.anamnesis);
    }

    return [...rows.values()];
  }

  private hasLiverToxEvidence(record: Record<string, unknown>): boolean {
    if (this.recordValue(record['matched_livertox_row'])) return true;
    if (this.stringValue(record['nbk_id'])) return true;
    const status = this.stringValue(record['match_status'])?.toLowerCase();
    if (status === 'matched_with_excerpt' || status === 'matched_no_excerpt' || status === 'matched') return true;
    return record['missing_livertox'] === false;
  }

  private hasRxNavEvidence(record: Record<string, unknown>): boolean {
    if (this.stringValue(record['rxnorm_rxcui'])) return true;
    if (this.stringValue(record['rxcui'])) return true;
    const sources = this.arrayValue(record['sources']);
    return sources.some((source) => this.stringValue(source)?.toLowerCase() === 'rxnav');
  }

  private originsContain(record: Record<string, unknown>, origin: 'therapy' | 'anamnesis'): boolean {
    return this.arrayValue(record['origins']).some((value) => this.stringValue(value)?.toLowerCase().includes(origin));
  }

  private rawMentionsContain(record: Record<string, unknown>, text: string): boolean {
    return this.arrayValue(record['raw_mentions']).some((value) => {
      const mention = this.stringValue(value);
      return mention ? this.textContainsDrug(text, mention) : false;
    });
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
    const fromAnamnesis = detail.result_payload?.['anamnesis_diseases'];
    const structuredCase = this.recordValue(detail.result_payload?.['structured_case']);
    const structuredDiseases = structuredCase?.['anamnesis_diseases'];
    const direct = this.collectDiseaseNames(fromPayload);
    if (direct.length) return direct;
    const anamnesis = this.collectDiseaseNames(fromAnamnesis);
    if (anamnesis.length) return anamnesis;
    const structured = this.collectDiseaseNames(structuredDiseases);
    if (structured.length) return structured;
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

  private collectDiseaseNames(value: unknown): string[] {
    if (!Array.isArray(value)) return [];
    const names = value
      .map((item) => {
        if (typeof item === 'string') return item.trim();
        const record = this.recordValue(item);
        if (!record) return '';
        return this.stringValue(record['name']) || this.stringValue(record['disease_name']) || '';
      })
      .filter((name) => name.length > 0);
    return [...new Set(names)];
  }

  private previewLabTimeline(detail: ClinicalSessionDetail): LabTimelineRow[] {
    return this.arrayValue(detail.result_payload?.['lab_timeline'])
      .map((item) => this.recordValue(item))
      .filter((item): item is Record<string, unknown> => item !== null)
      .map((item) => {
        const value = this.stringValue(item['value']) || this.stringValue(item['value_text']) || 'N/A';
        const unit = this.stringValue(item['unit']) || '';
        const upperLimit = this.stringValue(item['upper_limit_normal']) || this.stringValue(item['upper_limit_text']) || 'N/A';
        const timing = this.stringValue(item['sample_date']) || this.stringValue(item['relative_time']) || 'Unknown';
        return {
          marker: this.stringValue(item['marker_name']) || 'Lab',
          value,
          unit,
          upperLimit,
          timing,
          source: this.stringValue(item['source']) || 'N/A',
          evidence: this.stringValue(item['evidence']) || '',
        };
      });
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

  private drugNameFromUnknown(value: unknown): string | null {
    if (typeof value === 'string') return value.trim() || null;
    const record = this.recordValue(value);
    if (!record) return null;
    return this.stringValue(record['name'])
      || this.stringValue(record['drug_name'])
      || this.stringValue(record['raw_drug_name'])
      || this.stringValue(record['matched_drug_name']);
  }

  private stringValue(value: unknown): string | null {
    if (typeof value === 'string') return value.trim() || null;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    return null;
  }

  private recordValue(value: unknown): Record<string, unknown> | null {
    return value && typeof value === 'object' && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  }

  private arrayValue(value: unknown): unknown[] {
    return Array.isArray(value) ? value : [];
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

  onMetadataFilesSelected(kind: 'documents' | 'images', event: Event): void {
    const input = event.target as HTMLInputElement | null;
    const files = Array.from(input?.files || []);
    if (!files.length) return;
    const additions = files.map((file) => ({
      file_name: file.name,
      file_size: file.size,
      file_type: file.type || 'application/octet-stream',
      category: kind === 'images' ? 'image' : 'document',
      last_modified: new Date(file.lastModified).toISOString(),
    }));
    const metadata = this.readMetadataDraft();
    const current = Array.isArray(metadata[kind]) ? metadata[kind] as unknown[] : [];
    metadata[kind] = [...current, ...additions];
    this.metadataText.set(JSON.stringify(this.normalizeMetadata(metadata), null, 2));
    if (input) input.value = '';
  }

  private normalizeMetadata(metadata: Record<string, unknown>): Record<string, unknown> {
    return {
      documents: Array.isArray(metadata['documents']) ? metadata['documents'] : [],
      images: Array.isArray(metadata['images']) ? metadata['images'] : [],
      manual_metadata: metadata['manual_metadata'] && typeof metadata['manual_metadata'] === 'object'
        ? metadata['manual_metadata']
        : {},
      ...metadata,
    };
  }

  private readMetadataDraft(): Record<string, unknown> {
    try {
      return this.normalizeMetadata(JSON.parse(this.metadataText()) as Record<string, unknown>);
    } catch {
      return this.normalizeMetadata({});
    }
  }

  private revisionModelOverrides(): Record<string, unknown> {
    return {
      provider: this.revisionModelProvider(),
      clinical_model: this.revisionClinicalModel().trim() || null,
      text_extraction_model: this.revisionTextParsingModel().trim() || null,
      use_rag: this.revisionRagSearch(),
    };
  }

  private revisedSessionId(result: Record<string, unknown> | null): number | null {
    const value = result?.['session_id'];
    return typeof value === 'number' && Number.isInteger(value) ? value : null;
  }

  private stopPoller(): void {
    this.pollCancelled = true;
  }
}
