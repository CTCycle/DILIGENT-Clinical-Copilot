import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import {
  LucideBookOpen,
  LucideBraces,
  LucideFileText,
  LucideFlaskConical,
  LucideHeartPulse,
  LucideImage,
  LucidePill,
  LucideSave,
  LucideTrash2,
} from '@lucide/angular';

import { RevisionPipelineStatusComponent } from '../../components/revision-pipeline-status/revision-pipeline-status.component';
import { RevisionQaBadgeComponent } from '../../components/revision-qa-badge/revision-qa-badge.component';
import {
  deleteInspectionSession,
  fetchClinicalSessionDetail,
  fetchClinicalSessionRevisionArtifacts,
  fetchClinicalSessionRevisionEntities,
  fetchClinicalSessionRevisionReviews,
  fetchClinicalSessionRevisionJobStatus,
  fetchClinicalSessionRevisionPipelineRun,
  fetchClinicalSessionRevisionPipelineSteps,
  fetchClinicalSessionVersionComparison,
  fetchClinicalSessionVersions,
  fetchInspectionSessionTimelineList,
  fetchInspectionLiverToxCatalog,
  fetchInspectionRxNavCatalog,
  fetchInspectionSessions,
  generateInspectionSessionTimeline,
  manualEditClinicalSessionReport,
  retryClinicalSessionRevisionPipelineRun,
  startClinicalSessionRevisionJob,
  updateClinicalSession,
  updateClinicalSessionRevisionClinicalReview,
} from '../../core/services/inspection-api';
import { fetchModelConfigState } from '../../core/services/model-config-api';
import {
  ClinicalSessionDetail,
  CloudProvider,
  InspectionSessionItem,
  InspectionSessionStatus,
  InspectionSessionTimelinePreview,
  JobStatus,
  LocalModelCard,
  RevisionArtifact,
  RevisionClinicalReviewAction,
  RevisionClinicalReviewStatus,
  RevisionEntityDiff,
  RevisionEntity,
  RevisionPipelineRun,
  RevisionPipelineStep,
  SessionVersionComparisonResponse,
  SessionVersionSummary,
} from '../../core/models/types';
import { resolveCloudChoices } from '../../core/model-config';
import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';
import { formatErrorMessage, formatUnknownError } from '../../core/utils';

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

type RevisionProvider = 'ollama' | CloudProvider;

@Component({
  selector: 'app-clinical-sessions-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    RevisionPipelineStatusComponent,
    RevisionQaBadgeComponent,
    LucideBookOpen,
    LucideBraces,
    LucideFileText,
    LucideFlaskConical,
    LucideHeartPulse,
    LucideImage,
    LucidePill,
    LucideSave,
    LucideTrash2,
  ],
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
  readonly manualEditReviewerNote = signal('');
  readonly manualEditEditedBy = signal('');
  readonly metadataText = signal('{\n  "documents": [],\n  "images": []\n}');
  readonly revisionSelection = signal('');
  readonly revisionInstruction = signal('');
  readonly revisionModelProvider = signal<RevisionProvider>('ollama');
  readonly revisionClinicalModel = signal('');
  readonly revisionTextParsingModel = signal('');
  readonly revisionRagSearch = signal(false);
  readonly revisionLocalModels = signal<LocalModelCard[]>([]);
  readonly revisionCloudChoices = signal(resolveCloudChoices(undefined));
  readonly revisionModelDefaults = signal({ clinicalModel: '', textExtractionModel: '' });
  readonly revisionAvailableModels = computed(() => {
    const provider = this.revisionModelProvider();
    if (provider === 'ollama') {
      return this.revisionLocalModels()
        .filter((model) => model.available_in_ollama)
        .map((model) => model.name);
    }
    return this.revisionCloudChoices()[provider] || [];
  });
  readonly revisionClinicalModelOptions = computed(() => this.revisionAvailableModels());
  readonly revisionTextParsingModelOptions = computed(() => this.revisionAvailableModels());
  readonly activeSection = signal<'preview' | 'editor' | 'metadata' | 'revision' | 'timeline'>('preview');
  readonly saveStatus = signal('');
  readonly deletingSessionId = signal<number | null>(null);
  readonly revisionStatus = signal('');
  readonly revisionProgress = signal(0);
  readonly revisionJobStatus = signal<JobStatus | null>(null);
  readonly sessionVersions = signal<SessionVersionSummary[]>([]);
  readonly revisionHistoryLoading = signal(false);
  readonly revisionHistoryError = signal<string | null>(null);
  readonly selectedRevisionVersionId = signal<number | null>(null);
  readonly selectedRevisionPipelineRun = signal<RevisionPipelineRun | null>(null);
  readonly selectedRevisionPipelineSteps = signal<RevisionPipelineStep[]>([]);
  readonly selectedRevisionArtifacts = signal<RevisionArtifact[]>([]);
  readonly selectedRevisionEntities = signal<RevisionEntity[]>([]);
  readonly selectedRevisionReviews = signal<RevisionClinicalReviewAction[]>([]);
  readonly revisionComparisonBaseVersionId = signal<number | null>(null);
  readonly revisionComparisonLoading = signal(false);
  readonly revisionComparisonError = signal<string | null>(null);
  readonly revisionComparison = signal<SessionVersionComparisonResponse | null>(null);
  readonly revisionReviewEditedBy = signal('');
  readonly revisionReviewNote = signal('');
  readonly revisionReviewSaving = signal(false);
  readonly retryRevisionLoading = signal(false);
  readonly selectedRevisionVersion = computed(() =>
    this.sessionVersions().find((version) => version.version_id === this.selectedRevisionVersionId()) || null,
  );
  readonly selectedRevisionQaArtifact = computed(() =>
    this.selectedRevisionArtifacts().find((artifact) => artifact.artifact_kind === 'llm_qa_output') || null,
  );
  readonly selectedRevisionReportComparisonArtifact = computed(() =>
    this.selectedRevisionArtifacts().find((artifact) => artifact.artifact_kind === 'report_comparison') || null,
  );
  readonly selectedRevisionConsultationExecutionArtifact = computed(() =>
    this.selectedRevisionPipelineArtifacts().find((artifact) => artifact.artifact_key === 'revision_consultation_execution') || null,
  );
  readonly selectedRevisionFinalizationExecutionArtifact = computed(() =>
    this.selectedRevisionPipelineArtifacts().find((artifact) => artifact.artifact_key === 'revision_finalization_execution') || null,
  );
  readonly selectedRevisionStructuredCaseArtifacts = computed(() =>
    this.selectedRevisionArtifacts().filter((artifact) => artifact.artifact_kind === 'structured_case_entity'),
  );
  readonly selectedRevisionDrugEntities = computed(() =>
    this.selectedRevisionEntities().filter((entity) => entity.entity_type === 'drug'),
  );
  readonly selectedRevisionDiseaseEntities = computed(() =>
    this.selectedRevisionEntities().filter((entity) => entity.entity_type === 'disease'),
  );
  readonly selectedRevisionLabEntities = computed(() =>
    this.selectedRevisionEntities().filter((entity) => entity.entity_type === 'lab_timeline_entry'),
  );
  readonly selectedRevisionLiverToxEntities = computed(() =>
    this.selectedRevisionEntities().filter((entity) => entity.entity_type === 'livertox_match'),
  );
  readonly selectedRevisionDiliAssessmentEntities = computed(() =>
    this.selectedRevisionEntities().filter((entity) => entity.entity_type === 'dili_assessment'),
  );
  readonly selectedRevisionPipelineArtifacts = computed(() =>
    this.selectedRevisionArtifacts().filter((artifact) => artifact.artifact_kind === 'pipeline_artifact'),
  );
  readonly selectedRevisionCanReview = computed(() => {
    const version = this.selectedRevisionVersion();
    return Boolean(version && version.revision_kind === 'llm_assisted_revision' && version.session_id !== null);
  });
  readonly revisionBusy = computed(() => {
    const status = this.revisionJobStatus();
    return status === 'pending' || status === 'running';
  });
  readonly selectedRevisionCanRetry = computed(() => {
    const version = this.selectedRevisionVersion();
    const run = this.selectedRevisionPipelineRun();
    return Boolean(
      version
      && run
      && version.version_status === 'draft_revision'
      && run.status === 'failed'
      && !this.revisionBusy()
      && !this.retryRevisionLoading(),
    );
  });
  readonly revisionComparisonBaseOptions = computed(() => {
    const selectedVersionId = this.selectedRevisionVersionId();
    return this.sessionVersions().filter((version) => version.version_id !== selectedVersionId);
  });
  readonly detectedDrugEvidence = signal<DetectedDrugEvidence[]>([]);
  readonly detectedDiseases = signal<string[]>([]);
  readonly labSummary = signal<Array<{ label: string; value: string }>>([]);
  readonly labTimeline = signal<LabTimelineRow[]>([]);
  readonly hepatotoxicityPattern = signal<string>('N/A');
  readonly timelinePreviews = signal<InspectionSessionTimelinePreview[]>([]);
  readonly timelineListLoading = signal(false);
  readonly timelineListError = signal<string | null>(null);
  readonly timelineModelName = signal('');
  readonly timelineModelSource = signal<'local' | 'cloud'>('local');

  ngOnInit(): void {
    void this.loadRevisionModelCatalog();
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
      this.editorText.set(this.toEditorHtml(this.previewOfficialReport(detail)));
      this.editorViewMode.set('source');
      this.manualEditReviewerNote.set('');
      this.manualEditEditedBy.set(this.defaultReviewerLabel(detail));
      this.revisionReviewEditedBy.set(this.defaultReviewerLabel(detail));
      this.revisionReviewNote.set('');
      this.metadataText.set(JSON.stringify(this.normalizeMetadata(detail.metadata || {}), null, 2));
      this.revisionSelection.set('');
      this.revisionInstruction.set('');
      this.revisionModelProvider.set(this.resolveRevisionProvider(detail));
      this.revisionClinicalModel.set(detail.clinical_model || '');
      this.revisionTextParsingModel.set(detail.text_extraction_model || '');
      this.revisionRagSearch.set(this.resolvePersistedRagPreference(detail));
      this.syncRevisionModelSelections();
      this.syncTimelineModelSelection(detail);
      this.activeSection.set('preview');
      this.detectedDiseases.set(this.previewDetectedDiseases(detail));
      this.labSummary.set(this.previewLaboratorySummary(detail));
      this.labTimeline.set(this.previewLabTimeline(detail));
      this.hepatotoxicityPattern.set(this.previewHepatotoxicityPattern(detail));
      this.resetRevisionHistoryState();
      this.resetTimelineHistoryState();
      void this.loadDetectedDrugEvidence(detail);
      void this.loadRevisionHistory(detail.session_id);
      void this.loadTimelineHistory(detail.session_id);
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

  updateDateFilterMode(value: 'any' | 'after' | 'before' | 'exact'): void {
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
    this.revisionReviewEditedBy.set('');
    this.revisionReviewNote.set('');
    this.metadataText.set('{\n  "documents": [],\n  "images": []\n}');
    this.revisionSelection.set('');
    this.revisionInstruction.set('');
    this.detectedDrugEvidence.set([]);
    this.detectedDiseases.set([]);
    this.labSummary.set([]);
    this.labTimeline.set([]);
    this.hepatotoxicityPattern.set('N/A');
    this.resetRevisionHistoryState();
    this.resetTimelineHistoryState();
    this.detailError.set(null);
  }

  updateEditorText(value: string): void {
    this.editorText.set(value);
  }

  setEditorViewMode(mode: 'source' | 'rendered'): void {
    if (this.editorViewMode() === mode) return;
    this.editorViewMode.set(mode);
    const detail = this.selected();
    const sourceText = detail ? this.previewOfficialReport(detail) : this.editorText();
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
    if (!element) return this.editorText().trim();
    return this.editorHtmlToCanonicalText(element.innerHTML);
  }

  private editorHtmlToCanonicalText(html: string): string {
    const parsed = new DOMParser().parseFromString(`<div>${html}</div>`, 'text/html');
    const container = parsed.body.firstElementChild;
    if (!container) return '';
    const chunks: string[] = [];

    const appendText = (value: string): void => {
      const normalized = value.replace(/\s+/g, ' ').trim();
      if (!normalized) return;
      const previous = chunks[chunks.length - 1] || '';
      if (previous && !previous.endsWith('\n') && !previous.endsWith(' ')) {
        chunks.push(' ');
      }
      chunks.push(normalized);
    };

    const walk = (node: Node, listDepth = 0): void => {
      if (node.nodeType === Node.TEXT_NODE) {
        appendText(node.textContent || '');
        return;
      }
      if (!(node instanceof HTMLElement)) return;
      const tag = node.tagName.toLowerCase();
      if (tag === 'br') {
        chunks.push('\n');
        return;
      }
      if (tag === 'li') {
        chunks.push(`${chunks.length ? '\n' : ''}${'  '.repeat(listDepth)}- `);
      } else if (['p', 'div', 'section', 'article', 'blockquote', 'pre'].includes(tag)) {
        if (chunks.length && !chunks[chunks.length - 1].endsWith('\n')) chunks.push('\n');
      } else if (/^h[1-6]$/.test(tag)) {
        if (chunks.length && !chunks[chunks.length - 1].endsWith('\n')) chunks.push('\n');
        chunks.push(`${'#'.repeat(Number(tag[1]))} `);
      }
      const nextDepth = tag === 'ul' || tag === 'ol' ? listDepth + 1 : listDepth;
      for (const child of Array.from(node.childNodes)) {
        walk(child, nextDepth);
      }
      if (['p', 'div', 'section', 'article', 'blockquote', 'pre', 'li'].includes(tag)) {
        if (!chunks[chunks.length - 1]?.endsWith('\n')) chunks.push('\n');
      }
    };

    for (const child of Array.from(container.childNodes)) {
      walk(child);
    }
    return chunks.join('').replace(/\n{3,}/g, '\n\n').trim();
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

  updateRevisionSelection(value: string): void {
    this.revisionSelection.set(value);
  }

  updateRevisionInstruction(value: string): void {
    this.revisionInstruction.set(value);
  }

  updateManualEditReviewerNote(value: string): void {
    this.manualEditReviewerNote.set(value);
  }

  updateManualEditEditedBy(value: string): void {
    this.manualEditEditedBy.set(value);
  }

  updateRevisionReviewNote(value: string): void {
    this.revisionReviewNote.set(value);
  }

  updateRevisionReviewEditedBy(value: string): void {
    this.revisionReviewEditedBy.set(value);
  }

  updateRevisionComparisonBaseVersion(value: string): void {
    const parsed = Number.parseInt(value, 10);
    this.revisionComparisonBaseVersionId.set(Number.isFinite(parsed) ? parsed : null);
    void this.loadRevisionComparison();
  }

  setSection(section: 'preview' | 'editor' | 'metadata' | 'revision' | 'timeline'): void {
    this.activeSection.set(section);
  }

  async selectRevisionVersion(versionId: number): Promise<void> {
    const version = this.sessionVersions().find((item) => item.version_id === versionId) || null;
    this.selectedRevisionVersionId.set(version?.version_id ?? null);
    this.revisionComparisonBaseVersionId.set(
      version ? this.resolveRevisionComparisonBaseVersionId(this.sessionVersions(), version) : null,
    );
    if (version) {
      await Promise.all([
        this.loadRevisionArtifacts(version.version_id),
        this.loadRevisionEntities(version.version_id),
        this.loadRevisionReviews(version.version_id),
      ]);
    } else {
      this.selectedRevisionArtifacts.set([]);
      this.selectedRevisionEntities.set([]);
      this.selectedRevisionReviews.set([]);
    }
    if (version?.pipeline_run_id) {
      await this.loadRevisionPipeline(version.pipeline_run_id);
    } else {
      this.selectedRevisionPipelineRun.set(null);
      this.selectedRevisionPipelineSteps.set([]);
    }
    await this.loadRevisionComparison();
  }

  async openVersionFromHistory(version: SessionVersionSummary): Promise<void> {
    if (version.session_id === null) {
      await this.selectRevisionVersion(version.version_id);
      return;
    }
    await this.openSession(version.session_id);
  }

  updateRevisionModelProvider(value: RevisionProvider): void {
    this.revisionModelProvider.set(value);
    this.syncRevisionModelSelections();
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

  async saveManualReportEdit(): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    if (this.revisionBusy()) {
      this.saveStatus.set('[ERROR] Manual report edits are disabled while a revision job is running.');
      return;
    }
    this.saveStatus.set('Saving manual report edit...');
    const persistedEditorValue = this.editorValueToPersistedText();
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
      this.saveStatus.set('[ERROR] Metadata must be valid JSON.');
      return;
    }
    this.saveStatus.set('Saving metadata...');
    try {
      const updated = await updateClinicalSession(detail.session_id, { metadata });
      this.selected.set(updated);
      this.saveStatus.set('Metadata saved.');
    } catch (error) {
      this.saveStatus.set(formatUnknownError(error, 'Failed to save metadata.'));
    }
  }

  private resetRevisionHistoryState(): void {
    this.sessionVersions.set([]);
    this.revisionHistoryLoading.set(false);
    this.revisionHistoryError.set(null);
    this.selectedRevisionVersionId.set(null);
    this.selectedRevisionPipelineRun.set(null);
    this.selectedRevisionPipelineSteps.set([]);
    this.selectedRevisionArtifacts.set([]);
    this.selectedRevisionEntities.set([]);
    this.selectedRevisionReviews.set([]);
    this.revisionComparisonBaseVersionId.set(null);
    this.revisionComparisonLoading.set(false);
    this.revisionComparisonError.set(null);
    this.revisionComparison.set(null);
    this.revisionReviewNote.set('');
    this.retryRevisionLoading.set(false);
    this.revisionReviewSaving.set(false);
  }

  private resetTimelineHistoryState(): void {
    this.timelinePreviews.set([]);
    this.timelineListLoading.set(false);
    this.timelineListError.set(null);
  }

  private async loadTimelineHistory(sessionId: number): Promise<void> {
    this.timelineListLoading.set(true);
    this.timelineListError.set(null);
    try {
      const payload = await fetchInspectionSessionTimelineList(sessionId);
      if (this.selected()?.session_id !== sessionId) return;
      this.timelinePreviews.set(payload.items || []);
    } catch (error) {
      if (this.selected()?.session_id === sessionId) {
        this.timelinePreviews.set([]);
        this.timelineListError.set(formatUnknownError(error, 'Failed to load timeline history.'));
      }
    } finally {
      if (this.selected()?.session_id === sessionId) {
        this.timelineListLoading.set(false);
      }
    }
  }

  private async loadRevisionHistory(
    sessionId: number,
    options: { preferredVersionId?: number | null; preferredPipelineRunId?: string | null } = {},
  ): Promise<void> {
    this.revisionHistoryLoading.set(true);
    this.revisionHistoryError.set(null);
    try {
      const payload = await fetchClinicalSessionVersions(sessionId);
      if (this.selected()?.session_id !== sessionId) return;
      const items = payload.items || [];
      this.sessionVersions.set(items);
      const resolvedVersion =
        this.resolveVersionSelection(items, {
          sessionId,
          preferredVersionId: options.preferredVersionId ?? this.selectedRevisionVersionId(),
        });
      this.selectedRevisionVersionId.set(resolvedVersion?.version_id ?? null);
      this.revisionComparisonBaseVersionId.set(
        resolvedVersion ? this.resolveRevisionComparisonBaseVersionId(items, resolvedVersion) : null,
      );
      if (resolvedVersion) {
        await Promise.all([
          this.loadRevisionArtifacts(resolvedVersion.version_id),
          this.loadRevisionEntities(resolvedVersion.version_id),
          this.loadRevisionReviews(resolvedVersion.version_id),
        ]);
      } else {
        this.selectedRevisionArtifacts.set([]);
        this.selectedRevisionEntities.set([]);
        this.selectedRevisionReviews.set([]);
      }
      const pipelineRunId = options.preferredPipelineRunId || resolvedVersion?.pipeline_run_id || null;
      if (pipelineRunId) {
        await this.loadRevisionPipeline(pipelineRunId);
      } else {
        this.selectedRevisionPipelineRun.set(null);
        this.selectedRevisionPipelineSteps.set([]);
      }
      await this.loadRevisionComparison();
    } catch (error) {
      this.revisionHistoryError.set(formatUnknownError(error, 'Failed to load revision history.'));
    } finally {
      this.revisionHistoryLoading.set(false);
    }
  }

  private resolveVersionSelection(
    versions: SessionVersionSummary[],
    options: { sessionId: number; preferredVersionId?: number | null },
  ): SessionVersionSummary | null {
    const preferredVersionId = options.preferredVersionId ?? null;
    if (preferredVersionId !== null) {
      const preferred = versions.find((item) => item.version_id === preferredVersionId) || null;
      if (preferred) return preferred;
    }
    const currentSessionVersion = versions.find((item) => item.session_id === options.sessionId) || null;
    if (currentSessionVersion) return currentSessionVersion;
    return versions[versions.length - 1] || null;
  }

  private resolveRevisionComparisonBaseVersionId(
    versions: SessionVersionSummary[],
    selectedVersion: SessionVersionSummary,
  ): number | null {
    const sourceVersionId = selectedVersion.source_version_id;
    if (sourceVersionId !== null && versions.some((version) => version.version_id === sourceVersionId)) {
      return sourceVersionId;
    }
    const previousVersion = [...versions]
      .filter((version) => version.version_id !== selectedVersion.version_id)
      .sort((left, right) => right.version_number - left.version_number)
      .find((version) => version.version_number < selectedVersion.version_number);
    return previousVersion?.version_id ?? null;
  }

  private async loadRevisionArtifacts(versionId: number): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    try {
      const payload = await fetchClinicalSessionRevisionArtifacts(detail.session_id, versionId);
      if (this.selected()?.session_id !== detail.session_id || this.selectedRevisionVersionId() !== versionId) return;
      this.selectedRevisionArtifacts.set(payload.items || []);
    } catch (error) {
      this.selectedRevisionArtifacts.set([]);
      this.revisionHistoryError.set(formatUnknownError(error, 'Failed to load revision artifacts.'));
    }
  }

  private async loadRevisionEntities(versionId: number): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    try {
      const payload = await fetchClinicalSessionRevisionEntities(detail.session_id, versionId);
      if (this.selected()?.session_id !== detail.session_id || this.selectedRevisionVersionId() !== versionId) return;
      this.selectedRevisionEntities.set(payload.items || []);
    } catch (error) {
      this.selectedRevisionEntities.set([]);
      this.revisionHistoryError.set(formatUnknownError(error, 'Failed to load revision entities.'));
    }
  }

  private async loadRevisionReviews(versionId: number): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    try {
      const payload = await fetchClinicalSessionRevisionReviews(detail.session_id, versionId);
      if (this.selected()?.session_id !== detail.session_id || this.selectedRevisionVersionId() !== versionId) return;
      this.selectedRevisionReviews.set(payload.items || []);
    } catch (error) {
      this.selectedRevisionReviews.set([]);
      this.revisionHistoryError.set(formatUnknownError(error, 'Failed to load revision review history.'));
    }
  }

  private async loadRevisionComparison(): Promise<void> {
    const detail = this.selected();
    const selectedVersion = this.selectedRevisionVersion();
    const leftVersionId = this.revisionComparisonBaseVersionId();
    if (!detail || !selectedVersion || leftVersionId === null) {
      this.revisionComparison.set(null);
      this.revisionComparisonError.set(null);
      return;
    }
    this.revisionComparisonLoading.set(true);
    this.revisionComparisonError.set(null);
    try {
      const payload = await fetchClinicalSessionVersionComparison(
        detail.session_id,
        leftVersionId,
        selectedVersion.version_id,
      );
      if (
        this.selected()?.session_id !== detail.session_id
        || this.selectedRevisionVersionId() !== selectedVersion.version_id
        || this.revisionComparisonBaseVersionId() !== leftVersionId
      ) {
        return;
      }
      this.revisionComparison.set(payload);
    } catch (error) {
      this.revisionComparison.set(null);
      this.revisionComparisonError.set(formatUnknownError(error, 'Failed to load revision comparison.'));
    } finally {
      this.revisionComparisonLoading.set(false);
    }
  }

  private async loadRevisionPipeline(pipelineRunId: string): Promise<void> {
    try {
      const [run, steps] = await Promise.all([
        fetchClinicalSessionRevisionPipelineRun(pipelineRunId),
        fetchClinicalSessionRevisionPipelineSteps(pipelineRunId),
      ]);
      if (this.selectedRevisionVersion()?.pipeline_run_id !== pipelineRunId) return;
      this.selectedRevisionPipelineRun.set(run);
      this.selectedRevisionPipelineSteps.set(steps.items || []);
    } catch (error) {
      this.selectedRevisionPipelineRun.set(null);
      this.selectedRevisionPipelineSteps.set([]);
      this.revisionHistoryError.set(formatUnknownError(error, 'Failed to load revision pipeline details.'));
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
    this.activeSection.set('revision');
    this.revisionHistoryError.set(null);
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

  async retrySelectedRevisionRun(): Promise<void> {
    const detail = this.selected();
    const run = this.selectedRevisionPipelineRun();
    const versionId = this.selectedRevisionVersionId();
    if (!detail || !run || versionId === null || !this.selectedRevisionCanRetry()) return;
    this.stopPoller();
    this.retryRevisionLoading.set(true);
    this.revisionStatus.set('Starting revision retry...');
    this.revisionProgress.set(0);
    this.revisionHistoryError.set(null);
    try {
      const started = await retryClinicalSessionRevisionPipelineRun(run.pipeline_run_id);
      this.revisionJobStatus.set(started.status);
      await this.loadRevisionHistory(detail.session_id, {
        preferredVersionId: versionId,
        preferredPipelineRunId: run.pipeline_run_id,
      });
      void this.pollRevisionJob(started.job_id, Math.max(1000, started.poll_interval * 1000));
    } catch (error) {
      this.revisionStatus.set(formatUnknownError(error, 'Failed to retry revision.'));
    } finally {
      this.retryRevisionLoading.set(false);
    }
  }

  async updateRevisionClinicalReview(status: RevisionClinicalReviewStatus): Promise<void> {
    const detail = this.selected();
    const version = this.selectedRevisionVersion();
    if (!detail || !version || !this.selectedRevisionCanReview() || this.revisionReviewSaving()) return;
    this.revisionReviewSaving.set(true);
    this.revisionHistoryError.set(null);
    try {
      const response = await updateClinicalSessionRevisionClinicalReview(detail.session_id, version.version_id, {
        clinical_review_status: status,
        reviewer_note: this.revisionReviewNote().trim() || null,
        reviewed_by: this.revisionReviewEditedBy().trim() || null,
        metadata: {},
      });
      this.sessionVersions.update((items) =>
        items.map((item) => (item.version_id === response.version.version_id ? response.version : item)),
      );
      this.selectedRevisionReviews.update((items) => [response.review_action, ...items]);
      this.revisionReviewNote.set('');
    } catch (error) {
      this.revisionHistoryError.set(formatUnknownError(error, 'Failed to update human review status.'));
    } finally {
      this.revisionReviewSaving.set(false);
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
            return;
          }
          const detail = this.selected();
          const pipelineRunId = this.pipelineRunIdFromJobResult(status.result);
          if (detail) {
            await this.loadRevisionHistory(detail.session_id, {
              preferredPipelineRunId: pipelineRunId,
            });
          }
          return;
        }
        if (status.status === 'failed' || status.status === 'cancelled') {
          this.stopPoller();
          this.revisionStatus.set(
            formatErrorMessage(
              status.error || (status.status === 'cancelled' ? 'Revision cancelled.' : 'Revision failed.'),
            ),
          );
          const detail = this.selected();
          const pipelineRunId = this.pipelineRunIdFromJobResult(status.result);
          if (detail) {
            await this.loadRevisionHistory(detail.session_id, {
              preferredPipelineRunId: pipelineRunId,
            });
          }
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
      await this.loadTimelineHistory(detail.session_id);
      this.saveStatus.set('Timeline generated.');
    } catch (error) {
      this.saveStatus.set(formatUnknownError(error, 'Failed to create timeline.'));
    }
  }

  async openTimelineView(preview: InspectionSessionTimelinePreview): Promise<void> {
    if (preview.timeline_id) {
      await this.router.navigate(['/sessions', preview.session_id, 'timetable', preview.timeline_id]);
      return;
    }
    await this.router.navigate(['/sessions', preview.session_id, 'timetable']);
  }

  timelinePreviewRangeLabel(preview: InspectionSessionTimelinePreview): string {
    if (preview.start_date && preview.end_date) {
      const start = this.formatTimelinePreviewDate(preview.start_date);
      const end = this.formatTimelinePreviewDate(preview.end_date);
      return start === end ? start : `${start} - ${end}`;
    }
    if (preview.start_date) {
      return this.formatTimelinePreviewDate(preview.start_date);
    }
    return 'Undated chronology';
  }

  timelinePreviewSourceLabel(preview: InspectionSessionTimelinePreview): string {
    const model = preview.source_model?.trim();
    if (model) {
      return model;
    }
    return preview.generation_status === 'fallback' ? 'Fallback chronology' : 'Timeline extraction model';
  }

  timelinePreviewStatusLabel(preview: InspectionSessionTimelinePreview): string {
    return preview.generation_status === 'fallback' ? 'Fallback' : 'LLM generated';
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
    const provider = this.revisionModelProvider();
    const useCloudServices = provider !== 'ollama';
    return {
      provider: useCloudServices ? provider : null,
      use_cloud_services: useCloudServices,
      clinical_model: this.revisionClinicalModel().trim() || null,
      text_extraction_model: this.revisionTextParsingModel().trim() || null,
      use_rag: this.revisionRagSearch(),
    };
  }

  private async loadRevisionModelCatalog(): Promise<void> {
    try {
      const payload = await fetchModelConfigState(true);
      this.revisionLocalModels.set(payload.local_models || []);
      this.revisionCloudChoices.set(resolveCloudChoices(payload.cloud_model_choices));
      this.revisionModelDefaults.set({
        clinicalModel: payload.clinical_model || '',
        textExtractionModel: payload.text_extraction_model || '',
      });
      if (!this.selected()) {
        this.revisionModelProvider.set(payload.use_cloud_services ? payload.llm_provider : 'ollama');
      }
      this.timelineModelName.set(payload.text_extraction_model || '');
      this.timelineModelSource.set(payload.use_cloud_services ? 'cloud' : 'local');
      this.syncRevisionModelSelections();
      this.syncTimelineModelSelection(this.selected());
    } catch {
      // Keep existing revision form values if the shared model catalog cannot be loaded.
    }
  }

  private syncTimelineModelSelection(detail: ClinicalSessionDetail | null): void {
    const configuredModel = this.timelineModelName().trim();
    const detailModel = detail?.text_extraction_model?.trim() || '';
    this.timelineModelName.set(configuredModel || detailModel);
    if (!configuredModel && detailModel) {
      this.timelineModelSource.set('local');
    }
  }

  private resolveRevisionProvider(detail: ClinicalSessionDetail): RevisionProvider {
    const payloadProvider = this.stringValue(detail.result_payload?.['cloud_provider'])
      || this.stringValue(detail.result_payload?.['llm_provider']);
    if (payloadProvider === 'openai' || payloadProvider === 'gemini') {
      return payloadProvider;
    }

    const selectedModels = [
      this.stringValue(detail.clinical_model),
      this.stringValue(detail.text_extraction_model),
    ].filter((value): value is string => Boolean(value));

    const localModels = new Set(
      this.revisionLocalModels()
        .filter((model) => model.available_in_ollama)
        .map((model) => model.name),
    );
    if (selectedModels.length && selectedModels.every((model) => localModels.has(model))) {
      return 'ollama';
    }

    const cloudChoices = this.revisionCloudChoices();
    for (const provider of ['openai', 'gemini'] as const) {
      const providerModels = new Set(cloudChoices[provider] || []);
      if (selectedModels.some((model) => providerModels.has(model))) {
        return provider;
      }
    }

    return detail.result_payload?.['cloud_model'] ? 'openai' : 'ollama';
  }

  private syncRevisionModelSelections(): void {
    const options = this.revisionAvailableModels();
    this.revisionClinicalModel.set(
      this.resolveRevisionModelSelection(
        this.revisionClinicalModel(),
        this.revisionModelDefaults().clinicalModel,
        options,
      ),
    );
    this.revisionTextParsingModel.set(
      this.resolveRevisionModelSelection(
        this.revisionTextParsingModel(),
        this.revisionModelDefaults().textExtractionModel,
        options,
      ),
    );
  }

  private resolveRevisionModelSelection(
    currentValue: string,
    defaultValue: string,
    options: string[],
  ): string {
    const current = currentValue.trim();
    if (current && options.includes(current)) return current;
    const preferred = defaultValue.trim();
    if (preferred && options.includes(preferred)) return preferred;
    if (options.length) return options[0];
    return '';
  }

  private resolvePersistedRagPreference(detail: ClinicalSessionDetail): boolean {
    const directMetadata = this.booleanValue(detail.metadata?.['use_rag']);
    if (directMetadata !== null) return directMetadata;

    const revisionRecord = this.recordValue(detail.result_payload?.['revision']);
    const revisionMetadata = this.recordValue(revisionRecord?.['metadata']);
    const revisionMetadataValue = this.booleanValue(revisionMetadata?.['use_rag']);
    if (revisionMetadataValue !== null) return revisionMetadataValue;

    const overrideRecord = this.recordValue(revisionMetadata?.['model_overrides']);
    const overrideValue = this.booleanValue(overrideRecord?.['use_rag']);
    if (overrideValue !== null) return overrideValue;

    return false;
  }

  private booleanValue(value: unknown): boolean | null {
    if (typeof value === 'boolean') return value;
    if (typeof value === 'string') {
      const normalized = value.trim().toLowerCase();
      if (normalized === 'true') return true;
      if (normalized === 'false') return false;
    }
    return null;
  }

  private revisedSessionId(result: Record<string, unknown> | null): number | null {
    const value = result?.['session_id'];
    return typeof value === 'number' && Number.isInteger(value) ? value : null;
  }

  private pipelineRunIdFromJobResult(result: Record<string, unknown> | null): string | null {
    const value = result?.['pipeline_run_id'];
    return typeof value === 'string' && value.trim() ? value.trim() : null;
  }

  revisionStatusLabel(value: string | null | undefined): string {
    if (!value) return 'Unknown';
    return value
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());
  }

  revisionArtifactSummary(artifact: RevisionArtifact): string {
    const payload = artifact.payload || {};
    if (artifact.artifact_kind === 'llm_qa_output') {
      const blockingIssues = Array.isArray(payload['blocking_issues']) ? payload['blocking_issues'].length : 0;
      const manualReviewRequired = payload['manual_review_required'] === true;
      return [
        blockingIssues ? `${blockingIssues} blocking issue${blockingIssues === 1 ? '' : 's'}` : 'No blocking issues',
        manualReviewRequired ? 'Manual review required' : 'No manual review flag',
      ].join(' · ');
    }
    if (artifact.artifact_kind === 'report_comparison') {
      const outcome = typeof payload['outcome'] === 'string' ? payload['outcome'] : artifact.status || 'unknown';
      const manualReview = typeof payload['manual_review'] === 'string' ? payload['manual_review'] : null;
      return manualReview ? `Outcome ${outcome} · Manual review ${manualReview}` : `Outcome ${outcome}`;
    }
    if (artifact.artifact_kind === 'structured_case_entity') {
      const role = typeof payload['role'] === 'string' ? payload['role'] : null;
      return role ? `${this.revisionStatusLabel(artifact.entity_type)} · ${role}` : this.revisionStatusLabel(artifact.entity_type);
    }
    const keys = Object.keys(payload);
    return keys.length
      ? keys.slice(0, 3).map((key) => this.revisionStatusLabel(key)).join(' · ')
      : 'No structured payload summary saved.';
  }

  revisionComparisonEntitySummary(items: RevisionEntityDiff[]): string {
    if (!items.length) return 'None';
    return items.map((item) => item.summary).join(', ');
  }

  revisionArtifactPrimaryLabel(artifact: RevisionArtifact): string {
    if (artifact.artifact_kind === 'structured_case_entity') {
      return artifact.entity_name || artifact.artifact_key || 'Structured entity';
    }
    return artifact.artifact_key || this.revisionStatusLabel(artifact.artifact_kind);
  }

  revisionArtifactPayloadScalar(
    artifact: RevisionArtifact | null | undefined,
    key: string,
    fallback = 'Not recorded',
  ): string {
    const payload = this.recordValue(artifact?.payload);
    return this.stringValue(payload?.[key]) || fallback;
  }

  revisionArtifactPayloadFlag(
    artifact: RevisionArtifact | null | undefined,
    key: string,
    trueLabel = 'Yes',
    falseLabel = 'No',
    fallback = 'Not recorded',
  ): string {
    const payload = this.recordValue(artifact?.payload);
    const value = payload?.[key];
    if (typeof value === 'boolean') return value ? trueLabel : falseLabel;
    return fallback;
  }

  revisionEntityPrimaryLabel(entity: RevisionEntity): string {
    return entity.revised_name || entity.original_name || entity.original_entity_id || 'Unnamed entity';
  }

  revisionEntitySummary(entity: RevisionEntity): string {
    const parts = [
      this.revisionStatusLabel(entity.source_section),
      this.revisionStatusLabel(entity.entity_revision_status),
    ];
    if (entity.requires_human_review) {
      parts.push('Requires human review');
    }
    return parts.filter((part): part is string => Boolean(part)).join(' · ');
  }

  private formatTimelinePreviewDate(value: string): string {
    const parsed = new Date(`${value.slice(0, 10)}T00:00:00Z`);
    if (Number.isNaN(parsed.getTime())) {
      return value;
    }
    return new Intl.DateTimeFormat(undefined, {
      month: 'short',
      year: 'numeric',
      timeZone: 'UTC',
    }).format(parsed);
  }

  private stopPoller(): void {
    this.pollCancelled = true;
  }
}
