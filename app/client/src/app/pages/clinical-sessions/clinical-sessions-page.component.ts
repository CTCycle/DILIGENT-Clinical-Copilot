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

import { ModalShellComponent } from '../../components/modal-shell/modal-shell.component';
import { HelpPopoverComponent } from '../../core/guidance/help-popover.component';

import {
  deleteInspectionSession,
  fetchClinicalSessionDetail,
  fetchInspectionSessions,
  manualEditClinicalSessionReport,
  updateClinicalSession,
} from '../../core/services/clinical-sessions-api';
import {
  fetchInspectionLiverToxCatalog,
  fetchInspectionRxNavCatalog,
} from '../../core/services/knowledge-catalog-api';
import {
  cancelSessionRevisionJob,
  fetchRevisionArtifacts,
  fetchRevisionPipelineSteps,
  fetchSessionRevisionJobStatus,
  startSessionRevisionJob,
  updateRevisionClinicalReview,
} from '../../core/services/session-revision-api';
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
import { JobPollingService } from '../../core/services/job-polling.service';
import { formatErrorMessage, formatUnknownError, isRecord } from '../../core/utils';
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
import {
  buildPersistedDrugEvidence,
  DetectedDrugEvidence,
  diseaseTemporalLabel as previewDiseaseTemporalLabel,
  LabTimelineRow,
  normalizeDrugName,
  previewDetectedDiseases as extractDetectedDiseases,
  previewDetectedDrugs as extractDetectedDrugs,
  previewHepatotoxicityPattern as extractHepatotoxicityPattern,
  previewLaboratorySummary as extractLaboratorySummary,
  previewLabTimeline as extractLabTimeline,
  previewReport as extractReport,
  resolveDrugBibliographyLabel,
} from './clinical-session-preview';

@Component({
  selector: 'app-clinical-sessions-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ModalShellComponent,
    HelpPopoverComponent,
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
  private readonly jobPolling = inject(JobPollingService);

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
  readonly linkDialogOpen = signal(false);
  readonly linkUrl = signal('');
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
  readonly revisionReviewAvailable = computed(() => (
    this.revisionVersionId() !== null
    && !this.revisionRunning()
    && this.revisionStatus().trim().toLowerCase() === 'completed'
  ));
  readonly revisionModelLoading = signal(false);
  readonly revisionModelError = signal<string | null>(null);
  readonly revisionModelConfig = signal<ModelConfigStateResponse | null>(null);
  readonly revisionModelProvider = signal<'ollama' | CloudProvider>('ollama');
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
    this.linkUrl.set('https://');
    this.linkDialogOpen.set(true);
  }

  updateLinkUrl(value: string): void {
    this.linkUrl.set(value);
  }

  closeLinkDialog(): void {
    this.linkDialogOpen.set(false);
    this.linkUrl.set('');
  }

  confirmLink(): void {
    const url = this.linkUrl().trim();
    if (!url) return;
    this.runEditorCommand('createLink', url);
    this.closeLinkDialog();
  }

  private defaultReviewerLabel(detail: ClinicalSessionDetail): string {
    const reviewerValue = detail.metadata?.['reviewer'];
    const reviewer = typeof reviewerValue === 'string' ? reviewerValue.trim() : '';
    if (reviewer) return reviewer;
    const manualMetadata = isRecord(detail.metadata?.['manual_metadata'])
      ? detail.metadata['manual_metadata']
      : null;
    const manualReviewer = manualMetadata?.['reviewer'];
    return typeof manualReviewer === 'string' ? manualReviewer.trim() : '';
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
      const payload = await fetchModelConfigState();
      this.revisionModelConfig.set(payload);
      const configuredProvider = payload.cloud_providers.find((provider) => provider.id === payload.llm_provider)?.id;
      this.revisionModelProvider.set(payload.use_cloud_services ? configuredProvider || payload.cloud_providers[0]?.id || 'ollama' : 'ollama');
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

  revisionProviderOptions(): Array<{ id: 'ollama' | CloudProvider; display_name: string }> {
    return [{ id: 'ollama', display_name: 'Ollama' }, ...this.revisionCloudProviders()];
  }

  revisionCloudModels(): ModelConfigStateResponse['cloud_providers'][number]['models'] {
    return this.revisionCloudProviders().find((provider) => provider.id === this.revisionModelProvider())?.models || [];
  }

  revisionModels(): Array<{ id: string; display_name?: string }> {
    return this.revisionModelProvider() === 'ollama'
      ? this.revisionLocalModels().map((model) => ({ id: model.name, display_name: model.name }))
      : this.revisionCloudModels();
  }

  setRevisionModelProvider(value: string): void {
    if (value !== 'ollama' && !this.revisionCloudProviders().some((candidate) => candidate.id === value)) return;
    this.revisionModelProvider.set(value as 'ollama' | CloudProvider);
    const options = this.revisionModels().map((model) => model.id);
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
    this.revisionVersionId.set(null);
    this.revisionSteps.set([]);
    this.revisionArtifacts.set([]);
    this.revisionRunning.set(true);
    this.revisionStatus.set('Starting revision agent...');
    this.revisionPollCancelled = false;
    try {
      const started = await startSessionRevisionJob(detail.session_id, {
        revision_instruction: this.revisionInstruction().trim() || null,
        model_overrides: this.revisionModelProvider() !== 'ollama'
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
        clinical_review_status: status === 'approved' ? 'approved_by_human' : 'rejected_by_human',
        reviewed_by: this.manualEditEditedBy().trim() || null,
        reviewer_note: this.revisionInstruction().trim() || null,
      });
      this.revisionStatus.set(`Draft ${status} by human reviewer.`);
    } catch (error) {
      this.revisionStatus.set(formatUnknownError(error, 'Failed to record clinical review.'));
    }
  }
  private async pollRevision(sessionId: number, jobId: string): Promise<void> {
    await this.jobPolling.run({
      intervalMs: 1000,
      isCancelled: () => this.revisionPollCancelled,
      pollStep: async () => {
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
          return false;
        }
        return true;
      },
    });
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
      const parsed: unknown = JSON.parse(this.metadataText());
      if (!isRecord(parsed)) {
        this.metadataSaveStatus.set('[ERROR] Metadata must be a JSON object.');
        return;
      }
      metadata = parsed;
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
    return extractReport(detail);
  }

  previewOfficialReport(detail: ClinicalSessionDetail): string {
    return this.previewReport(detail);
  }

  previewReportHtml(detail: ClinicalSessionDetail): string {
    return this.markdownRenderer.render(this.previewOfficialReport(detail)).html;
  }

  previewDetectedDrugs(detail: ClinicalSessionDetail): string[] {
    return extractDetectedDrugs(detail);
  }

  previewDetectedDiseases(detail: ClinicalSessionDetail): string[] {
    return extractDetectedDiseases(detail);
  }

  previewLaboratorySummary(detail: ClinicalSessionDetail): Array<{ label: string; value: string }> {
    return extractLaboratorySummary(detail);
  }

  previewLabTimeline(detail: ClinicalSessionDetail): LabTimelineRow[] {
    return extractLabTimeline(detail);
  }

  previewHepatotoxicityPattern(detail: ClinicalSessionDetail): string {
    return extractHepatotoxicityPattern(detail);
  }

  diseaseTemporalLabel(disease: string): string {
    return previewDiseaseTemporalLabel(disease);
  }

  private async loadDetectedDrugEvidence(detail: ClinicalSessionDetail): Promise<void> {
    const rows = buildPersistedDrugEvidence(detail);
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
        bibliographyLabel: resolveDrugBibliographyLabel(row, fallbackByName.get(row.name)),
        bibliographyFallback: row.bibliographyFallback
          || (!row.liverTox && Boolean(fallbackByName.get(row.name)?.liverTox))
          || (!row.rxNav && Boolean(fallbackByName.get(row.name)?.rxNav)),
      })));
    }
  }


  private async catalogHasDrug(source: 'rxnav' | 'livertox', name: string): Promise<boolean> {
    const normalized = normalizeDrugName(name);
    const search = normalized || name;
    try {
      const payload = source === 'rxnav'
        ? await fetchInspectionRxNavCatalog({ search, offset: 0, limit: 5 })
        : await fetchInspectionLiverToxCatalog({ search, offset: 0, limit: 5 });
      return payload.items.some((item) => normalizeDrugName(item.drug_name) === normalized);
    } catch {
      return false;
    }
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
      const parsed: unknown = JSON.parse(this.metadataText());
      return normalizeClinicalSessionMetadata(isRecord(parsed) ? parsed : {});
    } catch {
      return normalizeClinicalSessionMetadata({});
    }
  }

  private stopPoller(): void {
    this.revisionPollCancelled = true;
  }
}


