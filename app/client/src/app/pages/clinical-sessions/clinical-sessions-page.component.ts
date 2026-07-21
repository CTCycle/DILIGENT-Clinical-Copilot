import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  LucideFileText,
  LucideFlaskConical,
  LucideHeartPulse,
  LucideImage,
  LucidePill,
  LucideTrash2,
} from '@lucide/angular';

import {
  cancelSessionRevisionJob,
  deleteInspectionSession,
  fetchClinicalSessionDetail,
  fetchRevisionArtifacts,
  fetchRevisionPipelineSteps,
  fetchInspectionLiverToxCatalog,
  fetchInspectionRxNavCatalog,
  fetchInspectionSessions,
  manualEditClinicalSessionReport,
  startSessionRevisionJob,
  fetchSessionRevisionJobStatus,
  updateClinicalSession,
  updateRevisionClinicalReview,
} from '../../core/services/inspection-api';
import { fetchModelConfigState } from '../../core/services/model-config-api';
import {
  ClinicalSessionDetail,
  InspectionSessionItem,
  InspectionSessionStatus,
  CloudProvider,
  ModelConfigStateResponse,
  RevisionArtifact,
  RevisionPipelineStep,
} from '../../core/models/types';
import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';
import { formatErrorMessage, formatUnknownError } from '../../core/utils';
import {
  DEFAULT_CLINICAL_SESSION_METADATA_TEXT,
  ClinicalSessionMetadataKey,
  normalizeClinicalSessionMetadata,
  readMetadataEntries,
} from './clinical-session-metadata';
import { ClinicalSessionEditorToolbarComponent } from './components/clinical-session-editor-toolbar.component';
import { ClinicalSessionTimelineWorkspaceComponent } from './components/clinical-session-timeline-workspace.component';
import { applyMarkdownCommand } from './markdown-editor';
import {
  ClinicalSessionDateFilterMode,
  ClinicalSessionSection,
  EditorCommandEvent,
  EditorCommandName,
  EditorViewMode,
} from './clinical-sessions.types';

type DetectedDrugEvidence = {
  name: string;
  liverTox: boolean;
  rxNav: boolean;
  inAnamnesis: boolean;
  inTherapy: boolean;
  temporalReference: string;
  extractionFallback: boolean;
  bibliographyLabel: string;
  bibliographyFallback: boolean;
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
  imports: [
    CommonModule,
    FormsModule,
    ClinicalSessionEditorToolbarComponent,
    ClinicalSessionTimelineWorkspaceComponent,
    LucideFileText,
    LucideFlaskConical,
    LucideHeartPulse,
    LucideImage,
    LucidePill,
    LucideTrash2,
  ],
  templateUrl: './clinical-sessions-page.component.html',
  styleUrl: './clinical-sessions-page.component.scss',
})
export class ClinicalSessionsPageComponent implements OnInit, OnDestroy {
  private readonly markdownRenderer = inject(MarkdownRendererService);
  private pollCancelled = false;

  readonly sessions = signal<InspectionSessionItem[]>([]);
  readonly statusFilter = signal<'all' | InspectionSessionStatus>('all');
  readonly dateFilterMode = signal<ClinicalSessionDateFilterMode>('any');
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
  readonly editorViewMode = signal<EditorViewMode>('source');
  readonly editorFontSize = signal(16);
  readonly renderedEditorHtml = computed(() => this.markdownRenderer.render(this.editorText()).html);
  private editorUndoStack: string[] = [];
  private editorRedoStack: string[] = [];
  readonly manualEditReviewerNote = signal('');
  readonly manualEditEditedBy = signal('');
  readonly metadataText = signal(DEFAULT_CLINICAL_SESSION_METADATA_TEXT);
  readonly metadataSaveStatus = signal('');
  readonly activeSection = signal<ClinicalSessionSection>('preview');
  readonly saveStatus = signal('');
  readonly deletingSessionId = signal<number | null>(null);
  readonly detectedDrugEvidence = signal<DetectedDrugEvidence[]>([]);
  readonly detectedDiseases = signal<string[]>([]);
  readonly labSummary = signal<Array<{ label: string; value: string }>>([]);
  readonly labTimeline = signal<LabTimelineRow[]>([]);
  readonly hepatotoxicityPattern = signal<string>('N/A');
  readonly revisionInstruction = signal('');
  readonly revisionStatus = signal('');
  readonly revisionRunning = signal(false);
  readonly revisionJobId = signal<string | null>(null);
  readonly revisionVersionId = signal<number | null>(null);
  readonly revisionSteps = signal<RevisionPipelineStep[]>([]);
  readonly revisionArtifacts = signal<RevisionArtifact[]>([]);
  readonly revisionModelLoading = signal(false);
  readonly revisionModelError = signal<string | null>(null);
  readonly revisionModelConfig = signal<ModelConfigStateResponse | null>(null);
  readonly revisionModelUseCloud = signal(false);
  readonly revisionModelProvider = signal<CloudProvider>('openai');
  readonly revisionModelName = signal('');
  private revisionPollCancelled = false;

  ngOnInit(): void {
    void this.loadSessions();
    void this.loadRevisionModelConfig();
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
      await this.ensureSelectedSessionVisible();
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
      this.editorText.set(this.previewOfficialReport(detail));
      this.editorUndoStack = [];
      this.editorRedoStack = [];
      this.editorViewMode.set('source');
      this.manualEditReviewerNote.set('');
      this.manualEditEditedBy.set(this.defaultReviewerLabel(detail));
      this.metadataText.set(JSON.stringify(normalizeClinicalSessionMetadata(detail.metadata || {}), null, 2));
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

  async deleteSession(session: InspectionSessionItem, event: Event): Promise<void> {
    event.stopPropagation();
    if (this.deletingSessionId() !== null) return;
    const patientLabel = session.patient_name || `Session ${session.session_id}`;
    if (!globalThis.confirm(`Delete ${patientLabel}? This cannot be undone.`)) return;
    this.deletingSessionId.set(session.session_id);
    this.listError.set(null);
    try {
      await deleteInspectionSession(session.session_id);
      this.sessions.update((items) => items.filter((item) => item.session_id !== session.session_id));
      if (this.selected()?.session_id === session.session_id) {
        this.clearSelectedSession();
        await this.ensureSelectedSessionVisible();
      }
    } catch (error) {
      this.listError.set(formatUnknownError(error, 'Failed to delete clinical session.'));
    } finally {
      this.deletingSessionId.set(null);
    }
  }

  updateQuery(value: string): void {
    this.query.set(value);
  }

  updateStatusFilter(value: 'all' | InspectionSessionStatus): void {
    this.statusFilter.set(value);
    void this.ensureSelectedSessionVisible();
  }

  updateDateFilterMode(value: ClinicalSessionDateFilterMode): void {
    this.dateFilterMode.set(value);
    if (value === 'any') {
      this.dateFilter.set('');
    }
    void this.ensureSelectedSessionVisible();
  }

  updateDateFilter(value: string): void {
    this.dateFilter.set(value);
    void this.ensureSelectedSessionVisible();
  }

  private async ensureSelectedSessionVisible(): Promise<void> {
    const visibleSessions = this.filteredSessions();
    const selectedId = this.selected()?.session_id;
    if (selectedId && visibleSessions.some((session) => session.session_id === selectedId)) {
      return;
    }
    const nextSession = visibleSessions[0];
    if (nextSession) {
      await this.openSession(nextSession.session_id);
      return;
    }
    this.clearSelectedSession();
  }

  private clearSelectedSession(): void {
    this.selected.set(null);
    this.editorText.set('');
    this.manualEditReviewerNote.set('');
    this.manualEditEditedBy.set('');
    this.metadataText.set(DEFAULT_CLINICAL_SESSION_METADATA_TEXT);
    this.detectedDrugEvidence.set([]);
    this.detectedDiseases.set([]);
    this.labSummary.set([]);
    this.labTimeline.set([]);
    this.hepatotoxicityPattern.set('N/A');
    this.detailError.set(null);
  }

  updateEditorText(value: string): void {
    if (value === this.editorText()) return;
    this.editorUndoStack.push(this.editorText());
    this.editorRedoStack = [];
    this.editorText.set(value);
  }

  setEditorViewMode(mode: EditorViewMode): void {
    if (this.editorViewMode() === mode) return;
    this.editorViewMode.set(mode);
  }

  setEditorFontSize(delta: number): void {
    const next = Math.min(22, Math.max(12, this.editorFontSize() + delta));
    this.editorFontSize.set(next);
  }

  handleEditorCommand(event: EditorCommandEvent): void {
    this.runEditorCommand(event.command, event.value);
  }

  runEditorCommand(command: EditorCommandName, value?: string): void {
    if (command === 'undo') {
      const previous = this.editorUndoStack.pop();
      if (previous !== undefined) {
        this.editorRedoStack.push(this.editorText());
        this.editorText.set(previous);
      }
      return;
    }
    if (command === 'redo') {
      const next = this.editorRedoStack.pop();
      if (next !== undefined) {
        this.editorUndoStack.push(this.editorText());
        this.editorText.set(next);
      }
      return;
    }
    const element = document.querySelector<HTMLTextAreaElement>('.clinical-session-editor-source');
    if (!element) return;
    const edit = applyMarkdownCommand(this.editorText(), element.selectionStart, element.selectionEnd, command, value);
    if (edit.text === this.editorText()) return;
    this.editorUndoStack.push(this.editorText());
    this.editorRedoStack = [];
    this.editorText.set(edit.text);
    requestAnimationFrame(() => {
      element.focus();
      element.setSelectionRange(edit.selectionStart, edit.selectionEnd);
    });
  }

  onEditorInput(event: Event): void {
    const target = event.target as HTMLTextAreaElement | null;
    if (!target) return;
    this.updateEditorText(target.value);
  }

  onEditorKeydown(event: KeyboardEvent): void {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      void this.saveManualReportEdit();
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

  private defaultReviewerLabel(detail: ClinicalSessionDetail): string {
    const reviewer = this.stringValue(detail.metadata?.['reviewer']);
    if (reviewer) return reviewer;
    const manualMetadata = this.recordValue(detail.metadata?.['manual_metadata']);
    return this.stringValue(manualMetadata?.['reviewer']) || '';
  }

  updateMetadataText(value: string): void {
    this.metadataText.set(value);
  }

  updateRevisionInstruction(value: string): void {
    this.revisionInstruction.set(value);
  }

  async loadRevisionModelConfig(): Promise<void> {
    this.revisionModelLoading.set(true);
    this.revisionModelError.set(null);
    try {
      const payload = await fetchModelConfigState(true);
      this.revisionModelConfig.set(payload);
      this.revisionModelUseCloud.set(payload.use_cloud_services);
      const configuredProvider = payload.cloud_providers.find((provider) => provider.id === payload.llm_provider)?.id;
      this.revisionModelProvider.set(configuredProvider || payload.cloud_providers[0]?.id || 'openai');
      this.revisionModelName.set(this.resolveInitialRevisionModel(payload));
    } catch (error) {
      this.revisionModelError.set(formatUnknownError(error, 'Unable to load revision model options.'));
    } finally {
      this.revisionModelLoading.set(false);
    }
  }

  revisionLocalModels(): ModelConfigStateResponse['local_models'] {
    return (this.revisionModelConfig()?.local_models || []).filter((model) => model.available_in_ollama);
  }

  revisionCloudProviders(): ModelConfigStateResponse['cloud_providers'] {
    return this.revisionModelConfig()?.cloud_providers || [];
  }

  revisionCloudModels(): ModelConfigStateResponse['cloud_providers'][number]['models'] {
    return this.revisionCloudProviders().find((provider) => provider.id === this.revisionModelProvider())?.models || [];
  }

  setRevisionModelRuntime(mode: 'local' | 'cloud'): void {
    const useCloud = mode === 'cloud';
    this.revisionModelUseCloud.set(useCloud);
    const options = useCloud
      ? this.revisionCloudModels().map((model) => model.id)
      : this.revisionLocalModels().map((model) => model.name);
    if (!options.includes(this.revisionModelName())) {
      this.revisionModelName.set(options[0] || '');
    }
  }

  setRevisionModelProvider(value: string): void {
    const provider = this.revisionCloudProviders().find((candidate) => candidate.id === value);
    if (!provider) return;
    this.revisionModelProvider.set(provider.id);
    const options = provider.models.map((model) => model.id);
    if (!options.includes(this.revisionModelName())) {
      this.revisionModelName.set(options[0] || '');
    }
  }

  updateRevisionModelName(value: string): void {
    this.revisionModelName.set(value);
  }

  private resolveInitialRevisionModel(payload: ModelConfigStateResponse): string {
    if (payload.use_cloud_services) {
      const provider = payload.cloud_providers.find((candidate) => candidate.id === payload.llm_provider);
      return provider?.models.some((model) => model.id === payload.cloud_model)
        ? payload.cloud_model || ''
        : provider?.models[0]?.id || '';
    }
    const localModels = payload.local_models.filter((model) => model.available_in_ollama);
    return localModels.some((model) => model.name === payload.clinical_model)
      ? payload.clinical_model || ''
      : localModels[0]?.name || payload.clinical_model || '';
  }

  async startRevision(): Promise<void> {
    const detail = this.selected();
    if (!detail || this.revisionRunning()) return;
    this.revisionRunning.set(true);
    this.revisionStatus.set('Starting revision agent...');
    this.revisionPollCancelled = false;
    try {
      const started = await startSessionRevisionJob(detail.session_id, {
        revision_instruction: this.revisionInstruction().trim() || null,
        model_overrides: this.revisionModelUseCloud()
          ? {
              use_cloud_models: true,
              cloud_provider: this.revisionModelProvider(),
              cloud_model: this.revisionModelName().trim() || null,
            }
          : {
              use_cloud_models: false,
              clinical_model: this.revisionModelName().trim() || null,
            },
      });
      this.revisionJobId.set(started.job_id);
      await this.pollRevision(detail.session_id, started.job_id);
    } catch (error) {
      this.revisionStatus.set(formatUnknownError(error, 'Failed to start revision agent.'));
      this.revisionRunning.set(false);
    }
  }

  async cancelRevision(): Promise<void> {
    const jobId = this.revisionJobId();
    if (!jobId) return;
    await cancelSessionRevisionJob(jobId);
    this.revisionPollCancelled = true;
    this.revisionRunning.set(false);
    this.revisionStatus.set('Cancellation requested.');
  }

  async recordRevisionReview(status: 'approved' | 'rejected'): Promise<void> {
    const detail = this.selected();
    const versionId = this.revisionVersionId();
    if (!detail || versionId === null) return;
    try {
      await updateRevisionClinicalReview(detail.session_id, versionId, {
        clinical_review_status: status,
        reviewed_by: this.manualEditEditedBy().trim() || null,
        reviewer_note: this.revisionInstruction().trim() || null,
      });
      this.revisionStatus.set(`Draft ${status} by human reviewer.`);
    } catch (error) {
      this.revisionStatus.set(formatUnknownError(error, 'Failed to record clinical review.'));
    }
  }
  private async pollRevision(sessionId: number, jobId: string): Promise<void> {
    while (!this.revisionPollCancelled) {
      const status = await fetchSessionRevisionJobStatus(jobId);
      const result = status.result;
      this.revisionStatus.set(status.status === 'running' ? 'Revision agent is working...' : status.status);
      if (typeof result?.revision_version_id === 'number') this.revisionVersionId.set(result.revision_version_id);
      if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
        this.revisionRunning.set(false);
        const pipelineRunId = typeof result?.pipeline_run_id === 'string' ? result.pipeline_run_id : null;
        const versionId = this.revisionVersionId();
        if (pipelineRunId) this.revisionSteps.set((await fetchRevisionPipelineSteps(pipelineRunId)).items);
        if (versionId) this.revisionArtifacts.set((await fetchRevisionArtifacts(sessionId, versionId)).items);
        return;
      }
      await new Promise<void>((resolve) => globalThis.setTimeout(resolve, 1000));
    }
  }
  updateManualEditReviewerNote(value: string): void {
    this.manualEditReviewerNote.set(value);
  }

  updateManualEditEditedBy(value: string): void {
    this.manualEditEditedBy.set(value);
  }

  setSection(section: ClinicalSessionSection): void {
    this.activeSection.set(section);
  }

  async saveManualReportEdit(): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    this.saveStatus.set('Saving manual report edit...');
    const persistedEditorValue = this.editorText();
    try {
      const response = await manualEditClinicalSessionReport(detail.session_id, {
        report_text: persistedEditorValue,
        edited_fields: ['report_text'],
        reviewer_note: this.manualEditReviewerNote().trim() || null,
        edited_by: this.manualEditEditedBy().trim() || null,
        metadata: {},
      });
      this.selected.set(response.session);
      this.editorText.set(persistedEditorValue);
      this.manualEditReviewerNote.set('');
      this.saveStatus.set('Manual report edit saved.');
    } catch (error) {
      this.saveStatus.set(formatUnknownError(error, 'Failed to save manual report edit.'));
    }
  }

  async saveMetadata(): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    let metadata: Record<string, unknown>;
    try {
      metadata = JSON.parse(this.metadataText()) as Record<string, unknown>;
    } catch {
      this.metadataSaveStatus.set('[ERROR] Metadata must be valid JSON.');
      return;
    }
    this.metadataSaveStatus.set('Saving metadata...');
    try {
      const updated = await updateClinicalSession(detail.session_id, { metadata });
      this.selected.set(updated);
      this.metadataSaveStatus.set('Metadata saved.');
    } catch (error) {
      this.metadataSaveStatus.set(formatUnknownError(error, 'Failed to save metadata.'));
    }
  }

  statusLabel(value: InspectionSessionStatus): string {
    return value === 'failed' ? 'Failed' : 'Successful';
  }

  previewReport(detail: ClinicalSessionDetail): string {
    const report = detail.official_report_text || detail.report || detail.result_payload?.['report'];
    return typeof report === 'string' && report.trim() ? report.trim() : 'No AI report preview is available for this session.';
  }

  previewOfficialReport(detail: ClinicalSessionDetail): string {
    return this.previewReport(detail);
  }

  previewReportHtml(detail: ClinicalSessionDetail): string {
    return this.markdownRenderer.render(this.previewOfficialReport(detail)).html;
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
        bibliographyLabel: this.resolveDrugBibliographyLabel(row, fallbackByName.get(row.name)),
        bibliographyFallback: row.bibliographyFallback
          || (!row.liverTox && Boolean(fallbackByName.get(row.name)?.liverTox))
          || (!row.rxNav && Boolean(fallbackByName.get(row.name)?.rxNav)),
      })));
    }
  }

  private buildPersistedDrugEvidence(detail: ClinicalSessionDetail): DrugEvidenceDraft[] {
    const rows = new Map<string, DrugEvidenceDraft>();
    const sections = this.sectionTextMap(detail);
    const ensureRow = (name: string, options: { fallback?: boolean } = {}): DrugEvidenceDraft => {
      const normalized = this.normalizeDrugName(name);
      const key = normalized || name.trim().toLowerCase();
      const existing = rows.get(key);
      if (existing) {
        existing.extractionFallback = existing.extractionFallback && Boolean(options.fallback);
        return existing;
      }
      const next: DrugEvidenceDraft = {
        name,
        liverTox: false,
        rxNav: false,
        inAnamnesis: this.textContainsDrug(sections.anamnesis, name),
        inTherapy: this.textContainsDrug(sections.therapy, name),
        temporalReference: this.drugTemporalReference(name, detail),
        extractionFallback: Boolean(options.fallback),
        bibliographyLabel: 'No backend match',
        bibliographyFallback: false,
        hasPersistedMatch: false,
      };
      rows.set(key, next);
      return next;
    };

    for (const name of this.previewDetectedDrugs(detail)) {
      for (const candidate of this.expandDrugCandidates(name, detail)) {
        ensureRow(candidate, { fallback: candidate !== name });
      }
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
      for (const candidate of this.expandDrugCandidates(name, detail)) {
        ensureRow(candidate, { fallback: candidate !== name }).inTherapy = true;
      }
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
      const expandedNames = this.expandDrugCandidates(name, detail);
      for (const expandedName of expandedNames) {
        ensureRow(expandedName, { fallback: expandedName !== name }).hasPersistedMatch = true;
      }
      if (this.looksLikeSentenceFragment(name) && expandedNames.length) continue;
      const row = ensureRow(name);
      const matchedName = this.stringValue(record['matched_drug_name']);
      row.hasPersistedMatch = true;
      row.name = row.name || matchedName || name;
      row.liverTox = row.liverTox || this.hasLiverToxEvidence(record);
      row.rxNav = row.rxNav || this.hasRxNavEvidence(record);
      row.bibliographyLabel = this.resolveDrugBibliographyLabel(row);
      row.inTherapy = row.inTherapy || this.originsContain(record, 'therapy') || this.rawMentionsContain(record, sections.therapy);
      row.inAnamnesis = row.inAnamnesis || this.originsContain(record, 'anamnesis') || this.rawMentionsContain(record, sections.anamnesis);
      row.temporalReference = this.drugTemporalReference(row.name, detail);
    }

    for (const row of rows.values()) {
      if (!row.liverTox && this.hasLiverToxReportEvidence(detail, row.name)) {
        row.liverTox = true;
        row.bibliographyLabel = this.resolveDrugBibliographyLabel(row);
      }
    }

    return [...rows.values()];
  }

  private resolveDrugBibliographyLabel(
    row: Pick<DetectedDrugEvidence, 'liverTox' | 'rxNav'>,
    fallback?: Partial<Pick<DetectedDrugEvidence, 'liverTox' | 'rxNav'>>,
  ): string {
    const backendLabels = [
      row.liverTox ? 'LiverTox' : null,
      row.rxNav ? 'RxNav' : null,
    ].filter((label): label is string => Boolean(label));
    if (backendLabels.length) return backendLabels.join(' + ');
    const fallbackLabels = [
      fallback?.liverTox ? 'LiverTox catalog fallback' : null,
      fallback?.rxNav ? 'RxNav catalog fallback' : null,
    ].filter((label): label is string => Boolean(label));
    return fallbackLabels.length ? fallbackLabels.join(' + ') : 'No backend match';
  }

  private expandDrugCandidates(name: string, detail: ClinicalSessionDetail): string[] {
    if (!this.looksLikeSentenceFragment(name)) return [name];
    const sections = this.sectionTextMap(detail);
    const source = `${sections.anamnesis}\n${sections.therapy}\n${detail.session_text}`;
    const candidates = [
      ...this.extractDrugsAfterStarting(source),
      ...this.extractCurrentMedicationList(source),
    ];
    return candidates.length ? [...new Set(candidates)] : [name];
  }

  private looksLikeSentenceFragment(value: string): boolean {
    const normalized = value.trim();
    return normalized.split(/\s+/).length > 4 || /patient|suspected|injury|symptoms/i.test(normalized);
  }

  private extractDrugsAfterStarting(source: string): string[] {
    const match = source.match(/\bafter starting\s+([^.\n]+?)(?:\.| symptoms| labs|$)/i);
    return match?.[1] ? this.splitMedicationList(match[1]) : [];
  }

  private extractCurrentMedicationList(source: string): string[] {
    const match = source.match(/\bcurrent medications are\s+([^.\n]+)/i);
    return match?.[1] ? this.splitMedicationList(match[1]) : [];
  }

  private splitMedicationList(value: string): string[] {
    const normalized = value
      .replace(/\bamoxicillin\s+clavulanate\b/gi, 'amoxicillin clavulanate,')
      .replace(/\batorvastatin\b/gi, 'atorvastatin,')
      .replace(/\bramipril\b/gi, 'ramipril,');
    return normalized
      .replace(/\band\b/gi, ',')
      .split(',')
      .map((item) => item.trim().replace(/[.;:]$/g, ''))
      .filter((item) => item.length > 2);
  }

  private drugTemporalReference(name: string, detail: ClinicalSessionDetail): string {
    const structuredCase = this.recordValue(detail.result_payload?.['structured_case']);
    const therapyDrugs = this.arrayValue(structuredCase?.['therapy_drugs']);
    for (const item of therapyDrugs) {
      const record = this.recordValue(item);
      if (!record) continue;
      const drugName = this.drugNameFromUnknown(record);
      if (!drugName || this.normalizeDrugName(drugName) !== this.normalizeDrugName(name)) continue;
      const startDate = this.stringValue(record['therapy_start_date']);
      if (startDate && !this.looksLikeSentenceFragment(startDate)) return startDate;
      const temporal = this.stringValue(record['temporal_classification']);
      if (temporal) return temporal.replace(/_/g, ' ');
    }
    const sections = this.sectionTextMap(detail);
    const source = `${sections.anamnesis}\n${sections.therapy}`;
    if (this.textContainsDrug(source, name) && /\bafter starting\b/i.test(source)) return 'after starting';
    if (this.textContainsDrug(source, name) && /\bcurrent medications?\b/i.test(source)) return 'current medication';
    return 'time not specified';
  }

  private hasLiverToxEvidence(record: Record<string, unknown>): boolean {
    if (this.recordValue(record['matched_livertox_row'])) return true;
    if (this.stringValue(record['nbk_id'])) return true;
    const status = this.stringValue(record['match_status'])?.toLowerCase();
    if (status === 'matched_with_excerpt' || status === 'matched_no_excerpt' || status === 'matched') return true;
    if (record['missing_livertox'] === false) return true;
    const candidates = this.arrayValue(record['livertox_candidates']);
    return candidates.some((candidate) => {
      const item = this.recordValue(candidate);
      return item?.['has_excerpt'] === true;
    });
  }

  private hasLiverToxReportEvidence(detail: ClinicalSessionDetail, drugName: string): boolean {
    const report = typeof detail.sections?.['final_report'] === 'string'
      ? detail.sections['final_report']
      : '';
    const normalizedReport = report.toLowerCase();
    const reportIndex = normalizedReport.indexOf(drugName.trim().toLowerCase());
    if (reportIndex < 0) return false;
    return /livertox/.test(normalizedReport.slice(reportIndex, reportIndex + 2400));
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
    const therapy = typeof sections['therapy'] === 'string'
      ? sections['therapy']
      : typeof sections['drugs'] === 'string'
        ? sections['drugs']
        : '';
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
    const fromSourceText = this.collectDiseasesFromSourceText(detail);
    if (fromSourceText.length) return fromSourceText;
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

  private collectDiseasesFromSourceText(detail: ClinicalSessionDetail): string[] {
    const sectionExtraction = this.recordValue(detail.result_payload?.['section_extraction']);
    const candidates = [
      detail.sections?.['anamnesis'],
      detail.session_text,
      this.stringValue(sectionExtraction?.['anamnesis']),
    ];
    const source = candidates
      .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
      .join('\n');
    if (!source.trim()) return [];
    const diseasePhrases: string[] = [];
    const historyMatch = source.match(/\b(?:past history|medical history|history)\s+includes?\s+([^.\n]+)/i);
    if (historyMatch?.[1]) {
      diseasePhrases.push(...this.splitDiseasePhrase(historyMatch[1]));
    }
    const suspectedMatch = source.match(/\b(?:suspected|concern is)\s+([^.\n]*?(?:liver injury|hepatitis|cholestasis|hepatocellular pattern)[^.\n]*)/i);
    if (suspectedMatch?.[1]) {
      diseasePhrases.push(suspectedMatch[1].trim());
    }
    return [...new Set(diseasePhrases.map((item) => item.trim()).filter((item) => item.length > 0))];
  }

  private splitDiseasePhrase(value: string): string[] {
    return value
      .replace(/\band\b/gi, ',')
      .split(',')
      .map((item) => item.trim().replace(/[.;:]$/g, ''))
      .filter((item) => item.length > 0);
  }

  diseaseTemporalLabel(disease: string): string {
    const normalized = disease.toLowerCase();
    if (normalized.includes('liver injury') || normalized.includes('hepatocellular')) return 'current concern';
    return 'past history';
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

  metadataEntries(key: ClinicalSessionMetadataKey): string[] {
    return readMetadataEntries(this.metadataText(), key);
  }

  onMetadataFilesSelected(kind: ClinicalSessionMetadataKey, event: Event): void {
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
    const current: unknown[] = Array.isArray(metadata[kind]) ? metadata[kind] : [];
    metadata[kind] = [...current, ...additions];
    this.metadataText.set(JSON.stringify(normalizeClinicalSessionMetadata(metadata), null, 2));
    if (input) input.value = '';
  }

  private readMetadataDraft(): Record<string, unknown> {
    try {
      return normalizeClinicalSessionMetadata(
        JSON.parse(this.metadataText()) as Record<string, unknown>,
      );
    } catch {
      return normalizeClinicalSessionMetadata({});
    }
  }

  private stopPoller(): void {
    this.pollCancelled = true;
  }
}


