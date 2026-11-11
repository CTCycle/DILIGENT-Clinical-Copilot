// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, ParamMap, Router, RouterLink } from '@angular/router';
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
  deleteSessionAttachment,
  fetchClinicalSessionDetail,
  fetchSessionAttachmentBlob,
  fetchSessionAttachments,
  manualEditClinicalSessionReport,
  updateClinicalSession,
  uploadSessionAttachment,
} from '../../core/services/clinical-sessions-api';
import { ApiHttpError } from '../../core/services/http-api';
import {
  fetchInspectionLiverToxCatalog,
  fetchInspectionRxNavCatalog,
} from '../../core/services/knowledge-catalog-api';
import {
  cancelSessionRevisionJob,
  fetchRevisionArtifacts,
  fetchRevisionPipelineSteps,
  fetchSessionRevisionJobStatus,
  fetchSessionVersions,
  startSessionRevisionJob,
  updateRevisionClinicalReview,
} from '../../core/services/session-revision-api';
import {
  ClinicalSessionAttachment,
  ClinicalSessionDetail,
  InspectionSessionItem,
  InspectionSessionStatus,
} from '../../core/models/inspection-types';
import {
  RevisionArtifact,
  RevisionJobResult,
  RevisionPipelineStep,
  SessionVersionSummary,
} from '../../core/models/revision-types';
import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';
import { JobPollingService } from '../../core/services/job-polling.service';
import { AgentsStateService } from '../../core/state/agents-state.service';
import { formatErrorMessage, formatUnknownError, isRecord } from '../../core/utils';
import {
  DEFAULT_CLINICAL_SESSION_METADATA_TEXT,
  normalizeClinicalSessionMetadata,
} from './clinical-session-metadata';
import { ClinicalSessionEditorToolbarComponent } from './components/clinical-session-editor-toolbar.component';
import { ClinicalSessionRevisionAuditComponent } from './components/clinical-session-revision-audit.component';
import { ClinicalSessionTimelineWorkspaceComponent } from './components/clinical-session-timeline-workspace.component';
import { ClinicalSessionsFiltersComponent } from './components/clinical-sessions-filters.component';
import { ClinicalSessionsStore } from './clinical-sessions-store';
import { formatRevisionAgentProvenance } from './clinical-session-revision-display';
import { applyMarkdownCommand } from './markdown-editor';
import {
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
  terminologyResolvedLabel,
} from './clinical-session-preview';
import { TerminologyCandidate } from '../../core/models/inspection-types';
import { decideTerminologyAlias } from '../../core/services/terminology-api';
import {
  isActiveSessionStatus,
  sessionStatusLabel,
  sessionStatusTone,
} from '../../core/utils/session-status';
import { terminologyStatusLabel, terminologyStatusTone } from '../../core/utils/terminology-status';

function selectPersistedRevision(
  versions: SessionVersionSummary[],
  sessionId: number,
): SessionVersionSummary | undefined {
  const sessionVersions = versions
    .filter((version) => version.session_id === sessionId)
    .sort((left, right) => right.version_number - left.version_number);
  const currentSessionVersionId = sessionVersions[0]?.version_id ?? null;
  const candidates = versions
    .filter((version) => version.revision_kind === 'llm_assisted_revision')
    .filter((version) => (
      version.session_id === sessionId
      || (currentSessionVersionId !== null
        && version.source_version_id === currentSessionVersionId)
    ))
    .sort((left, right) => (
      Date.parse(right.updated_at) - Date.parse(left.updated_at)
      || right.version_number - left.version_number
      || right.version_id - left.version_id
    ));
  return candidates[0];
}

const ACTIVE_REFRESH_MS = 3000;
const SECTIONS: readonly ClinicalSessionSection[] = ['preview', 'editor', 'metadata', 'revision', 'timeline'];

function parseQueryId(value: string | null): number | null {
  const parsed = Number(value);
  return value !== null && Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function parseSection(value: string | null): ClinicalSessionSection | null {
  return SECTIONS.find((section) => section === value) ?? null;
}

@Component({
  selector: 'app-clinical-sessions-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    RouterLink,
    ModalShellComponent,
    HelpPopoverComponent,
    ClinicalSessionsFiltersComponent,
    ClinicalSessionEditorToolbarComponent,
    ClinicalSessionRevisionAuditComponent,
    ClinicalSessionTimelineWorkspaceComponent,
    LucideFileText,
    LucideFlaskConical,
    LucideHeartPulse,
    LucideImage,
    LucidePill,
    LucideTrash2,
  ],
  providers: [ClinicalSessionsStore],
  templateUrl: './clinical-sessions-page.component.html',
  styleUrl: './clinical-sessions-page.component.scss',
})
export class ClinicalSessionsPageComponent implements OnInit, OnDestroy {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly markdownRenderer = inject(MarkdownRendererService);
  private readonly jobPolling = inject(JobPollingService);
  private readonly agentsState = inject(AgentsStateService);

  readonly store = inject(ClinicalSessionsStore);
  readonly sessions = this.store.visible;
  readonly statusLabel = sessionStatusLabel;
  readonly statusTone = sessionStatusTone;
  readonly isActiveStatus = isActiveSessionStatus;
  readonly selectedIsInFlight = computed(() => {
    const status = this.selected()?.status;
    return status === 'draft' || status === 'queued' || status === 'running';
  });
  readonly selected = signal<ClinicalSessionDetail | null>(null);
  readonly loading = this.store.loading;
  readonly detailLoading = signal(false);
  readonly listError = signal<string | null>(null);
  readonly detailError = signal<string | null>(null);
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
  readonly attachments = signal<ClinicalSessionAttachment[]>([]);
  readonly imageAttachments = computed(() => this.attachments().filter((item) => item.kind === 'image'));
  readonly documentAttachments = computed(() => this.attachments().filter((item) => item.kind !== 'image'));
  readonly activeSection = signal<ClinicalSessionSection>('preview');
  readonly saveStatus = signal('');
  readonly staleEditDetected = signal(false);
  readonly deletingSessionId = signal<number | null>(null);
  readonly detectedDrugEvidence = signal<DetectedDrugEvidence[]>([]);
  readonly terminologyDecisionPending = signal(false);
  readonly terminologyStatusLabel = terminologyStatusLabel;
  readonly terminologyStatusTone = terminologyStatusTone;
  readonly terminologyResolvedLabel = terminologyResolvedLabel;
  readonly detectedDiseases = signal<string[]>([]);
  readonly labSummary = signal<Array<{ label: string; value: string }>>([]);
  readonly labTimeline = signal<LabTimelineRow[]>([]);
  readonly hepatotoxicityPattern = signal<string>('N/A');
  readonly revisionInstruction = signal('');
  readonly revisionStatus = signal('');
  readonly revisionRunning = signal(false);
  readonly revisionJobId = signal<string | null>(null);
  readonly revisionVersionId = signal<number | null>(null);
  readonly revisionVersionStatus = signal<SessionVersionSummary['version_status'] | null>(null);
  readonly revisionSteps = signal<RevisionPipelineStep[]>([]);
  readonly revisionArtifacts = signal<RevisionArtifact[]>([]);
  readonly revisionDraftReport = signal('');
  readonly revisionDraftReportHtml = computed(() => this.markdownRenderer.render(this.revisionDraftReport()).html);
  readonly revisionReviewAvailable = computed(() => (
    this.revisionVersionId() !== null
    && !this.revisionRunning()
    && this.revisionVersionStatus() === 'llm_qa_passed'
  ));
  readonly revisionAgentId = signal<string | null>(null);
  readonly revisionAgents = this.agentsState.revisionAgents;
  readonly revisionAgent = computed(() => {
    const selected = this.revisionAgentId();
    return (selected ? this.agentsState.byId(selected) : null) ?? this.agentsState.defaultRevisionAgent();
  });
  readonly revisionAgentSummary = computed(() => {
    const config = this.revisionAgent()?.revision.configuration;
    if (!config || config.kind !== 'revision') return '';
    const { planning } = config.models;
    const { max_tasks: tasks, max_tool_iterations: iterations } = config.limits;
    return `${planning.provider} / ${planning.model} · up to ${tasks} tasks, ${iterations} tool iterations`;
  });
  readonly revisionAgentLoading = computed(() => this.agentsState.status() === 'loading');
  readonly revisionAgentError = computed(() => {
    const error = this.agentsState.error();
    return error ? formatUnknownError(error, 'Unable to load revision agents.') : null;
  });
  readonly revisionRunAgent = signal<string | null>(null);
  readonly revisionAuditVisible = computed(() => (
    this.revisionRunning()
    || this.revisionVersionId() !== null
    || this.revisionSteps().length > 0
    || this.revisionArtifacts().length > 0
  ));
  private revisionPollCancelled = false;
  private revisionLoadGeneration = 0;
  private detailLoadGeneration = 0;
  private requestedSessionId: number | null = null;
  private deepLinkPending = false;
  private refreshTimer: ReturnType<typeof globalThis.setTimeout> | null = null;

  ngOnInit(): void {
    this.deepLinkPending = parseQueryId(this.route.snapshot.queryParamMap.get('session')) !== null;
    this.route.queryParamMap.subscribe((params) => this.applyQueryParams(params));
    void this.loadSessions();
    void this.loadRevisionAgents();
  }

  ngOnDestroy(): void {
    this.stopPoller();
    this.clearRefreshTimer();
    this.store.destroy();
  }

  private applyQueryParams(params: ParamMap): void {
    const sessionId = parseQueryId(params.get('session'));
    const tab = parseSection(params.get('tab'));
    if (sessionId !== null && sessionId !== this.requestedSessionId) {
      void this.openSession(sessionId, tab ?? 'preview', false);
    } else if (tab !== null && sessionId === this.requestedSessionId) {
      this.activeSection.set(tab);
    }
  }

  private syncQueryParams(replaceUrl = false): void {
    void this.router.navigate([], {
      relativeTo: this.route,
      queryParams: {
        session: this.requestedSessionId,
        tab: this.requestedSessionId === null ? null : this.activeSection(),
      },
      queryParamsHandling: 'merge',
      replaceUrl,
    });
  }

  private scheduleRefresh(): void {
    this.clearRefreshTimer();
    if (!this.store.hasActiveSessions()) return;
    this.refreshTimer = globalThis.setTimeout(() => {
      this.refreshTimer = null;
      void this.syncActiveSessions();
    }, ACTIVE_REFRESH_MS);
  }

  private clearRefreshTimer(): void {
    if (this.refreshTimer !== null) {
      globalThis.clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
  }

  /** Full reload of the 90-day cache; filters and selection are left untouched. */
  async loadSessions(quiet = false): Promise<void> {
    this.clearRefreshTimer();
    this.listError.set(null);
    await this.store.loadRecent(quiet);
    if (this.store.loaded()) {
      this.reopenSelectedWhenStatusChanged(this.store.visible());
      await this.ensureSelectedSessionVisible();
      this.deepLinkPending = false;
    }
    this.scheduleRefresh();
  }

  private async syncActiveSessions(): Promise<void> {
    const changed = await this.store.syncChanges();
    this.reopenSelectedWhenStatusChanged(changed);
    this.scheduleRefresh();
  }

  async openSession(
    sessionId: number,
    section: ClinicalSessionSection = 'preview',
    updateUrl = true,
  ): Promise<void> {
    this.requestedSessionId = sessionId;
    const detailLoadGeneration = ++this.detailLoadGeneration;
    const isCurrentLoad = (): boolean => detailLoadGeneration === this.detailLoadGeneration;
    const revisionLoadGeneration = ++this.revisionLoadGeneration;
    this.revisionPollCancelled = true;
    this.revisionInstruction.set('');
    this.revisionStatus.set('');
    this.revisionRunning.set(false);
    this.revisionJobId.set(null);
    this.revisionVersionId.set(null);
    this.revisionVersionStatus.set(null);
    this.revisionSteps.set([]);
    this.revisionArtifacts.set([]);
    this.revisionDraftReport.set('');
    this.detailLoading.set(true);
    this.detailError.set(null);
    try {
      const detail = await fetchClinicalSessionDetail(sessionId);
      if (!isCurrentLoad()) return;
      this.selected.set(detail);
      this.editorText.set(this.previewOfficialReport(detail));
      this.editorUndoStack = [];
      this.editorRedoStack = [];
      this.editorViewMode.set('source');
      this.manualEditReviewerNote.set('');
      this.staleEditDetected.set(false);
      this.manualEditEditedBy.set(this.defaultReviewerLabel(detail));
      this.metadataText.set(JSON.stringify(normalizeClinicalSessionMetadata(detail.metadata || {}), null, 2));
      this.activeSection.set(section);
      this.detectedDiseases.set(this.previewDetectedDiseases(detail));
      this.labSummary.set(this.previewLaboratorySummary(detail));
      this.labTimeline.set(this.previewLabTimeline(detail));
      this.hepatotoxicityPattern.set(this.previewHepatotoxicityPattern(detail));
      void this.loadAttachments(detail.session_id, detailLoadGeneration);
      void this.loadDetectedDrugEvidence(detail, detailLoadGeneration);
      void this.loadPersistedRevision(detail.session_id, revisionLoadGeneration);
      if (updateUrl) this.syncQueryParams();
    } catch (error) {
      if (isCurrentLoad()) {
        this.detailError.set(formatUnknownError(error, 'Failed to open session.'));
      }
    } finally {
      if (isCurrentLoad()) this.detailLoading.set(false);
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
      this.store.remove(session.session_id);
      if (this.selected()?.session_id === session.session_id) {
        this.clearSelectedSession();
        await this.ensureSelectedSessionVisible();
      }
    } catch (error) {
      this.listError.set(
        error instanceof ApiHttpError && error.statusCode === 409
          ? '[ERROR] This session cannot be deleted while it is running or has revised child sessions. Delete the child sessions first.'
          : formatUnknownError(error, 'Failed to delete clinical session.'),
      );
    } finally {
      this.deletingSessionId.set(null);
    }
  }

  /** The open session stays open when filters hide it; only an empty selection picks the first row. */
  private async ensureSelectedSessionVisible(): Promise<void> {
    if (this.selected() || (this.deepLinkPending && this.requestedSessionId !== null)) return;
    const nextSession = this.sessions()[0];
    if (nextSession) {
      await this.openSession(nextSession.session_id);
      return;
    }
    this.clearSelectedSession();
    if (this.route.snapshot.queryParamMap.has('session')) this.syncQueryParams(true);
  }

  /** Lifecycle changes (e.g. running to completed) alter what the open session can show. */
  private reopenSelectedWhenStatusChanged(items: InspectionSessionItem[]): void {
    const selected = this.selected();
    const current = selected && items.find((item) => item.session_id === selected.session_id);
    if (selected && current && current.status !== selected.status) {
      void this.openSession(selected.session_id, this.activeSection(), false);
    }
  }

  private clearSelectedSession(): void {
    this.requestedSessionId = null;
    this.detailLoadGeneration += 1;
    this.revisionLoadGeneration += 1;
    this.selected.set(null);
    this.editorText.set('');
    this.manualEditReviewerNote.set('');
    this.manualEditEditedBy.set('');
    this.metadataText.set(DEFAULT_CLINICAL_SESSION_METADATA_TEXT);
    this.attachments.set([]);
    this.detectedDrugEvidence.set([]);
    this.detectedDiseases.set([]);
    this.labSummary.set([]);
    this.labTimeline.set([]);
    this.hepatotoxicityPattern.set('N/A');
    this.revisionPollCancelled = true;
    this.revisionInstruction.set('');
    this.revisionStatus.set('');
    this.revisionRunning.set(false);
    this.revisionJobId.set(null);
    this.revisionVersionId.set(null);
    this.revisionVersionStatus.set(null);
    this.revisionSteps.set([]);
    this.revisionArtifacts.set([]);
    this.revisionDraftReport.set('');
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

  selectRevisionAgent(agentId: string): void {
    this.revisionAgentId.set(agentId || null);
  }

  async loadRevisionAgents(): Promise<void> {
    try {
      await this.agentsState.load();
    } catch {
      // The shared agents resource exposes the visible error state.
    }
  }

  async startRevision(): Promise<void> {
    const detail = this.selected();
    if (!detail || this.revisionRunning()) return;
    const revisionLoadGeneration = ++this.revisionLoadGeneration;
    const isCurrentLoad = (): boolean => (
      revisionLoadGeneration === this.revisionLoadGeneration
      && this.selected()?.session_id === detail.session_id
    );
    this.revisionVersionId.set(null);
    this.revisionVersionStatus.set(null);
    this.revisionSteps.set([]);
    this.revisionArtifacts.set([]);
    this.revisionRunning.set(true);
    const agent = this.revisionAgent();
    this.revisionRunAgent.set(agent ? `${agent.name} · revision ${agent.current_revision}` : null);
    this.revisionStatus.set('Starting revision agent...');
    this.revisionPollCancelled = false;
    try {
      const started = await startSessionRevisionJob(detail.session_id, {
        revision_instruction: this.revisionInstruction().trim() || null,
        agent_id: agent?.id ?? null,
      });
      if (!isCurrentLoad()) return;
      this.revisionJobId.set(started.job_id);
      await this.pollRevision(detail.session_id, started.job_id, revisionLoadGeneration);
    } catch (error) {
      if (isCurrentLoad()) {
        this.revisionStatus.set(formatUnknownError(error, 'Failed to start revision agent.'));
        this.revisionRunning.set(false);
      }
    }
  }

  async cancelRevision(): Promise<void> {
    const jobId = this.revisionJobId();
    const detail = this.selected();
    const revisionLoadGeneration = this.revisionLoadGeneration;
    const isCurrentLoad = (): boolean => (
      revisionLoadGeneration === this.revisionLoadGeneration
      && (!detail || this.selected()?.session_id === detail.session_id)
    );
    if (!jobId) return;
    this.revisionPollCancelled = true;
    if (isCurrentLoad()) {
      this.revisionRunning.set(false);
      this.revisionStatus.set('Cancellation requested.');
    }
    try {
      await cancelSessionRevisionJob(jobId);
      if (detail && isCurrentLoad()) {
        await this.loadPersistedRevision(detail.session_id, revisionLoadGeneration);
      }
    } catch (error) {
      if (isCurrentLoad()) {
        this.revisionStatus.set(formatUnknownError(error, 'Failed to cancel revision.'));
      }
    }
  }

  async recordRevisionReview(status: 'approved' | 'rejected'): Promise<void> {
    const detail = this.selected();
    const versionId = this.revisionVersionId();
    if (!detail || versionId === null) return;
    const revisionLoadGeneration = this.revisionLoadGeneration;
    const isCurrentLoad = (): boolean => (
      revisionLoadGeneration === this.revisionLoadGeneration
      && this.selected()?.session_id === detail.session_id
    );
    try {
      await updateRevisionClinicalReview(detail.session_id, versionId, {
        clinical_review_status: status === 'approved' ? 'approved_by_human' : 'rejected_by_human',
        reviewed_by: this.manualEditEditedBy().trim() || null,
        reviewer_note: this.revisionInstruction().trim() || null,
      });
      if (isCurrentLoad()) this.revisionStatus.set(`Draft ${status} by human reviewer.`);
    } catch (error) {
      if (isCurrentLoad()) {
        this.revisionStatus.set(formatUnknownError(error, 'Failed to record clinical review.'));
      }
    }
  }
  private async pollRevision(
    sessionId: number,
    jobId: string,
    revisionLoadGeneration: number,
  ): Promise<void> {
    const isCurrentLoad = (): boolean => (
      revisionLoadGeneration === this.revisionLoadGeneration
      && this.selected()?.session_id === sessionId
    );
    await this.jobPolling.run({
      intervalMs: 1000,
      isCancelled: () => this.revisionPollCancelled || !isCurrentLoad(),
      pollStep: async () => {
        const status = await fetchSessionRevisionJobStatus(jobId);
        if (this.revisionPollCancelled || !isCurrentLoad()) return false;
        const result = status.result;
        this.revisionStatus.set(this.revisionJobStatusLabel(status.status, result));
        if (typeof result?.revision_version_id === 'number') this.revisionVersionId.set(result.revision_version_id);
        if (this.isRevisionVersionStatus(result?.revision_status)) {
          this.revisionVersionStatus.set(result.revision_status);
        }
        if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
          this.revisionRunning.set(false);
          this.revisionJobId.set(null);
          if (status.status === 'cancelled') {
            this.revisionVersionStatus.set('cancelled');
          }
          const pipelineRunId = typeof result?.pipeline_run_id === 'string' ? result.pipeline_run_id : null;
          const versionId = this.revisionVersionId();
          if (pipelineRunId) {
            const steps = (await fetchRevisionPipelineSteps(pipelineRunId)).items;
            if (!isCurrentLoad()) return false;
            this.revisionSteps.set(steps);
          }
          if (versionId) {
            const artifacts = (await fetchRevisionArtifacts(sessionId, versionId)).items;
            if (!isCurrentLoad()) return false;
            this.revisionArtifacts.set(artifacts);
            this.applyRevisionDraftArtifact(artifacts);
          }
          return false;
        }
        return true;
      },
    });
  }

  private revisionJobStatusLabel(
    jobStatus: string,
    result: RevisionJobResult | null | undefined,
  ): string {
    if (result?.revision_status === 'qa_failed') return 'Revision completed with QA issues.';
    if (result?.revision_status === 'requires_human_review') return 'Revision requires human review.';
    if (jobStatus === 'cancelled') return 'Revision cancelled.';
    if (jobStatus !== 'running') return jobStatus;
    switch (result?.revision_phase) {
      case 'planning':
        return 'Planning revision...';
      case 'tool_selection':
        return 'Selecting revision evidence...';
      case 'editing':
        return 'Generating revision draft...';
      case 'quality_review':
        return result.repair_attempt
          ? `Reviewing repaired draft (attempt ${result.repair_attempt})...`
          : 'Reviewing revision draft...';
      case 'repairing':
        return `Repairing revision draft (attempt ${result.repair_attempt || 1})...`;
      case 'finalizing':
        return 'Finalizing revision audit...';
      default:
        return 'Revision agent is working...';
    }
  }

  private async loadPersistedRevision(sessionId: number, revisionLoadGeneration: number): Promise<void> {
    const isCurrentLoad = (): boolean =>
      revisionLoadGeneration === this.revisionLoadGeneration
      && this.selected()?.session_id === sessionId;
    try {
      const latest = selectPersistedRevision(
        (await fetchSessionVersions(sessionId)).items,
        sessionId,
      );
      if (!latest || !isCurrentLoad()) return;
      this.revisionVersionId.set(latest.version_id);
      this.revisionVersionStatus.set(latest.version_status);
      const configuration = latest.model_configuration || {};
      this.revisionRunAgent.set(formatRevisionAgentProvenance(configuration));
      const pipelineRunId = latest.pipeline_run_id;
      if (pipelineRunId) {
        const steps = (await fetchRevisionPipelineSteps(pipelineRunId)).items;
        if (!isCurrentLoad()) return;
        this.revisionSteps.set(steps);
      }
      const artifacts = (await fetchRevisionArtifacts(sessionId, latest.version_id)).items;
      if (!isCurrentLoad()) return;
      this.revisionArtifacts.set(artifacts);
      this.applyRevisionDraftArtifact(artifacts);
      const jobId = typeof configuration['job_id'] === 'string' ? configuration['job_id'] : null;
      if (jobId && ['draft_revision', 'pending_qa'].includes(latest.version_status)) {
        if (!isCurrentLoad()) return;
        this.revisionJobId.set(jobId);
        this.revisionRunning.set(true);
        this.revisionStatus.set('Revision agent is working...');
        this.revisionPollCancelled = false;
        await this.pollRevision(sessionId, jobId, revisionLoadGeneration);
        return;
      }
      if (!isCurrentLoad()) return;
      this.revisionStatus.set(
        this.revisionVersionStatusLabel(latest.version_status),
      );
    } catch (error) {
      if (isCurrentLoad()) {
        this.revisionStatus.set(formatUnknownError(error, 'Unable to restore the saved revision draft.'));
      }
    }
  }

  private isRevisionVersionStatus(value: unknown): value is SessionVersionSummary['version_status'] {
    return [
      'current',
      'superseded',
      'draft_revision',
      'pending_qa',
      'cancelled',
      'qa_failed',
      'requires_human_review',
      'llm_qa_passed',
      'human_approved',
      'human_rejected',
    ].includes(value as SessionVersionSummary['version_status']);
  }

  private revisionVersionStatusLabel(status: SessionVersionSummary['version_status']): string {
    switch (status) {
      case 'cancelled':
        return 'Revision cancelled.';
      case 'qa_failed':
        return 'Revision completed with QA issues.';
      case 'requires_human_review':
        return 'Revision requires human review.';
      case 'llm_qa_passed':
        return 'Revision draft loaded from persisted session data.';
      case 'human_approved':
        return 'Draft approved by human reviewer.';
      case 'human_rejected':
        return 'Draft rejected by human reviewer.';
      default:
        return 'Revision draft loaded from persisted session data.';
    }
  }

  private applyRevisionDraftArtifact(artifacts: RevisionArtifact[]): void {
    const artifact = artifacts.find((candidate) => candidate.artifact_key === 'revision_agent_draft_report_repair')
      || artifacts.find((candidate) => candidate.artifact_key === 'revision_agent_draft_report');
    const payload = artifact?.payload;
    const report = payload && typeof payload['revised_report_text'] === 'string'
      ? payload['revised_report_text']
      : '';
    if (report.trim()) this.revisionDraftReport.set(report);
  }

  sectionHelpTitle(): string {
    return {
      preview: 'Preview help',
      editor: 'Text Editor help',
      metadata: 'Metadata help',
      revision: 'Revision help',
      timeline: 'Timeline help',
    }[this.activeSection()];
  }

  sectionHelpBody(): string {
    return {
      preview: 'Read the persisted report and its extracted clinical evidence. This view does not change the session.',
      editor: 'Edit the session text or report in Markdown, switch to the rendered view, and save a manual audit entry when finished.',
      metadata: 'Attach source images or documents, then save valid JSON metadata so later review and timeline work can use that context.',
      revision: 'Revision creates an auditable draft using the configured Revision role. Inspect the draft and trace before approving or rejecting it as a human reviewer.',
      timeline: 'Timeline generation builds a dated chronology from the session using the configured Timeline role. Saved timelines can be reopened and compared after generation.',
    }[this.activeSection()];
  }

  updateManualEditReviewerNote(value: string): void {
    this.manualEditReviewerNote.set(value);
  }

  updateManualEditEditedBy(value: string): void {
    this.manualEditEditedBy.set(value);
  }

  setSection(section: ClinicalSessionSection): void {
    this.activeSection.set(section);
    this.syncQueryParams(true);
  }

  async saveManualReportEdit(): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    this.saveStatus.set('Saving manual report edit...');
    const persistedEditorValue = this.editorText();
    try {
      const response = await manualEditClinicalSessionReport(detail.session_id, {
        report_text: persistedEditorValue,
        expected_version: detail.version,
        edited_fields: ['report_text'],
        reviewer_note: this.manualEditReviewerNote().trim() || null,
        edited_by: this.manualEditEditedBy().trim() || null,
      });
      this.selected.set(response.session);
      this.store.patch(response.session.session_id, { version: response.session.version });
      this.editorText.set(persistedEditorValue);
      this.manualEditReviewerNote.set('');
      const revisionLoadGeneration = ++this.revisionLoadGeneration;
      this.revisionPollCancelled = true;
      this.revisionInstruction.set('');
      this.revisionStatus.set('');
      this.revisionRunning.set(false);
      this.revisionJobId.set(null);
      this.revisionVersionId.set(null);
      this.revisionVersionStatus.set(null);
      this.revisionSteps.set([]);
      this.revisionArtifacts.set([]);
      this.revisionDraftReport.set('');
      void this.loadPersistedRevision(response.session.session_id, revisionLoadGeneration);
      this.staleEditDetected.set(false);
      this.saveStatus.set('Manual report edit saved.');
    } catch (error) {
      if (error instanceof ApiHttpError && error.statusCode === 409) {
        this.staleEditDetected.set(true);
        this.saveStatus.set(
          '[ERROR] The report changed since you opened it. Reload the latest version; your draft is kept.',
        );
        return;
      }
      this.saveStatus.set(formatUnknownError(error, 'Failed to save manual report edit.'));
    }
  }

  /** Refetch the latest session without discarding the unsaved editor draft. */
  async reloadLatestSession(): Promise<void> {
    const detail = this.selected();
    if (!detail) return;
    try {
      const latest = await fetchClinicalSessionDetail(detail.session_id);
      this.selected.set(latest);
      this.store.patch(latest.session_id, { version: latest.version });
      this.staleEditDetected.set(false);
      this.saveStatus.set('Latest report loaded. Your draft was kept; review it before saving.');
    } catch (error) {
      this.saveStatus.set(formatUnknownError(error, 'Failed to reload the latest report.'));
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

  inFlightPreviewNote(status: InspectionSessionStatus): string {
    return status === 'draft'
      ? 'This session is a draft. Edit and run it to generate a report.'
      : 'The analysis has not finished. Open the report to follow its progress.';
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

  async confirmDrugIdentity(drugName: string, candidate: TerminologyCandidate): Promise<void> {
    const confirmed = globalThis.confirm(
      `Confirm "${drugName}" as ${candidate.name}? Future assessments will use this identity; this session is not changed.`,
    );
    if (!confirmed) return;
    this.terminologyDecisionPending.set(true);
    try {
      await decideTerminologyAlias('confirm', drugName, candidate.drug_id);
      this.detailError.set(null);
      this.detectedDrugEvidence.update((rows) => rows.map((row) => (
        row.name === drugName && row.terminology
          ? {
            ...row,
            terminology: {
              ...row.terminology,
              status: 'verified',
              accepted: true,
              method: 'confirmed_local',
              drug_id: candidate.drug_id,
              preferred_name: candidate.name,
              ingredients: [{
                drug_id: candidate.drug_id,
                name: candidate.name,
                concept_type: candidate.concept_type,
                parent_drug_id: null,
                parent_name: null,
              }],
              candidates: [],
            },
          }
          : row
      )));
    } catch (error) {
      this.detailError.set(error instanceof Error ? error.message : 'Failed to confirm the drug identity.');
    } finally {
      this.terminologyDecisionPending.set(false);
    }
  }

  private async loadDetectedDrugEvidence(
    detail: ClinicalSessionDetail,
    detailLoadGeneration: number,
  ): Promise<void> {
    const isCurrentLoad = (): boolean => (
      detailLoadGeneration === this.detailLoadGeneration
      && this.selected()?.session_id === detail.session_id
    );
    const rows = buildPersistedDrugEvidence(detail);
    if (isCurrentLoad()) {
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
    if (isCurrentLoad()) {
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


  async downloadAttachment(item: ClinicalSessionAttachment): Promise<void> {
    try {
      const blob = await fetchSessionAttachmentBlob(item.session_id, item.attachment_id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = item.file_name;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      this.metadataSaveStatus.set(formatUnknownError(error, 'Failed to download attachment.'));
    }
  }

  private async loadAttachments(sessionId: number, generation: number): Promise<void> {
    try {
      const items = await fetchSessionAttachments(sessionId);
      if (generation === this.detailLoadGeneration) this.attachments.set(items);
    } catch (error) {
      if (generation === this.detailLoadGeneration) {
        this.metadataSaveStatus.set(formatUnknownError(error, 'Failed to load attachments.'));
      }
    }
  }

  async onMetadataFilesSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement | null;
    const files = Array.from(input?.files || []);
    const detail = this.selected();
    if (input) input.value = '';
    if (!files.length || !detail) return;
    this.metadataSaveStatus.set(`Uploading ${files.length} file(s)...`);
    try {
      for (const file of files) {
        await uploadSessionAttachment(detail.session_id, file);
      }
      await this.loadAttachments(detail.session_id, this.detailLoadGeneration);
      this.metadataSaveStatus.set('Attachments saved.');
    } catch (error) {
      this.metadataSaveStatus.set(formatUnknownError(error, 'Failed to upload attachment.'));
    }
  }

  async removeAttachment(item: ClinicalSessionAttachment): Promise<void> {
    try {
      await deleteSessionAttachment(item.session_id, item.attachment_id);
      this.attachments.update((items) => items.filter((entry) => entry.attachment_id !== item.attachment_id));
    } catch (error) {
      this.metadataSaveStatus.set(formatUnknownError(error, 'Failed to remove attachment.'));
    }
  }

  private stopPoller(): void {
    this.revisionPollCancelled = true;
  }
}


