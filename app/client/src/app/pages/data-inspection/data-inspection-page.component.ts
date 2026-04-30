import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { InspectionActionIconButtonComponent } from '../../components/inspection-action-icon-button/inspection-action-icon-button.component';
import { ModalShellComponent } from '../../components/modal-shell/modal-shell.component';
import { InspectionCatalogToolbarComponent } from '../../components/inspection-catalog-toolbar/inspection-catalog-toolbar.component';
import {
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRagUpdateJob,
  cancelInspectionRxNavUpdateJob,
  deleteInspectionLiverToxDrug,
  deleteInspectionRxNavDrug,
  deleteInspectionSession,
  fetchInspectionLiverToxCatalog,
  fetchInspectionLiverToxExcerpt,
  fetchInspectionLiverToxUpdateConfig,
  fetchInspectionLiverToxUpdateJobStatus,
  fetchInspectionRagDocuments,
  fetchInspectionRagUpdateConfig,
  fetchInspectionRagUpdateJobStatus,
  fetchInspectionRagVectorStore,
  fetchInspectionRxNavAliases,
  fetchInspectionRxNavCatalog,
  fetchInspectionRxNavUpdateConfig,
  fetchInspectionRxNavUpdateJobStatus,
  fetchInspectionSessionReport,
  fetchInspectionSessions,
  startInspectionLiverToxUpdateJob,
  startInspectionRagUpdateJob,
  startInspectionRxNavUpdateJob,
} from '../../core/services/inspection-api';
import { JobPollingService } from '../../core/services/job-polling.service';
import {
  InspectionDateFilterMode,
  InspectionDrugAliasesResponse,
  InspectionLiverToxExcerptResponse,
  InspectionLiverToxItem,
  InspectionRagDocumentRow,
  InspectionRagVectorStoreSummary,
  InspectionRxNavItem,
  InspectionSessionItem,
  InspectionSessionStatus,
  InspectionUpdateTarget,
} from '../../core/models/types';
import { InspectionDetailResource } from './inspection-detail-resource';
import { InspectionPagedResource } from './inspection-paged-resource';
import { InspectionUpdateJobResource, InspectionUpdateTargetActionsMap } from './inspection-update-job-resource';

type InspectionViewId =
  | 'sessions'
  | 'rxnav'
  | 'livertox'
  | 'rag';

const INSPECTION_VIEWS: Array<{ id: InspectionViewId; label: string }> = [
  { id: 'sessions', label: 'Sessions' },
  { id: 'rxnav', label: 'Drug Catalog' },
  { id: 'livertox', label: 'LiverTox' },
  { id: 'rag', label: 'RAG' },
];

function inspectionTabId(view: InspectionViewId): string {
  return `inspection-tab-${view}`;
}

function formatDateTime(value: string | null): string {
  if (!value) return 'N/A';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatDuration(seconds: number | null): string {
  if (typeof seconds !== 'number' || Number.isNaN(seconds) || seconds < 0) return 'N/A';
  const rounded = Math.round(seconds);
  if (rounded < 60) return `${rounded}s`;
  return `${Math.floor(rounded / 60)}m ${rounded % 60}s`;
}

function resolveRagDocumentsPath(
  vectorStore: InspectionRagVectorStoreSummary | null,
): string {
  if (!vectorStore) {
    return '';
  }
  const explicitPath = vectorStore.source_documents_path?.trim();
  if (explicitPath) {
    return explicitPath;
  }
  const vectorDbPath = vectorStore.vector_db_path?.trim();
  if (!vectorDbPath) {
    return '';
  }
  return vectorDbPath.replace(new RegExp(String.raw`[\\/]vectors$`, 'i'), (match) =>
    match.startsWith('\\') ? '\\documents' : '/documents',
  );
}

@Component({
  selector: 'app-data-inspection-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ModalShellComponent,
    InspectionActionIconButtonComponent,
    InspectionCatalogToolbarComponent,
  ],
  templateUrl: './data-inspection-page.component.html',
  styleUrl: './data-inspection-page.component.scss',
})
export class DataInspectionPageComponent implements OnInit, OnDestroy {
  @ViewChild('ragFolderInput') private ragFolderInput?: ElementRef<HTMLInputElement>;

  private readonly jobPolling = inject(JobPollingService);
  private readonly router = inject(Router);
  readonly inspectionViews = INSPECTION_VIEWS;
  readonly activeView = signal<InspectionViewId>('sessions');

  readonly sessionStatusFilter = signal<InspectionSessionStatus | 'all'>('all');
  readonly sessionDateMode = signal<InspectionDateFilterMode | 'none'>('none');
  readonly sessionDate = signal('');
  private readonly sessionCatalog = new InspectionPagedResource<InspectionSessionItem>(
    (params) => {
      const statusFilter = this.sessionStatusFilter();
      const dateModeFilter = this.sessionDateMode();
      return fetchInspectionSessions({
        ...params,
        status: statusFilter === 'all' ? undefined : statusFilter,
        date_mode: dateModeFilter === 'none' ? undefined : dateModeFilter,
        date: dateModeFilter === 'none' ? undefined : this.sessionDate() || undefined,
      });
    },
    'Failed to load sessions.',
  );
  readonly sessionItems = this.sessionCatalog.items;
  readonly sessionVisibleItems = this.sessionCatalog.visibleItems;
  readonly sessionVisibleStartIndex = this.sessionCatalog.visibleStartIndex;
  readonly sessionTopPaddingPx = this.sessionCatalog.topPaddingPx;
  readonly sessionBottomPaddingPx = this.sessionCatalog.bottomPaddingPx;
  readonly sessionTotal = this.sessionCatalog.total;
  readonly sessionLoading = this.sessionCatalog.loading;
  readonly sessionLoadingMore = this.sessionCatalog.loadingMore;
  readonly sessionHasMore = this.sessionCatalog.hasMore;
  readonly sessionError = this.sessionCatalog.error;
  readonly sessionSearchInput = this.sessionCatalog.searchInput;

  private readonly rxnavCatalog = new InspectionPagedResource<InspectionRxNavItem>(
    (params) => fetchInspectionRxNavCatalog(params),
    'Failed to load drug catalog.',
  );
  readonly rxnavItems = this.rxnavCatalog.items;
  readonly rxnavVisibleItems = this.rxnavCatalog.visibleItems;
  readonly rxnavVisibleStartIndex = this.rxnavCatalog.visibleStartIndex;
  readonly rxnavTopPaddingPx = this.rxnavCatalog.topPaddingPx;
  readonly rxnavBottomPaddingPx = this.rxnavCatalog.bottomPaddingPx;
  readonly rxnavTotal = this.rxnavCatalog.total;
  readonly rxnavLoading = this.rxnavCatalog.loading;
  readonly rxnavLoadingMore = this.rxnavCatalog.loadingMore;
  readonly rxnavHasMore = this.rxnavCatalog.hasMore;
  readonly rxnavError = this.rxnavCatalog.error;
  readonly rxnavSearchInput = this.rxnavCatalog.searchInput;

  private readonly liverToxCatalog = new InspectionPagedResource<InspectionLiverToxItem>(
    (params) => fetchInspectionLiverToxCatalog(params),
    'Failed to load LiverTox.',
  );
  readonly livertoxItems = this.liverToxCatalog.items;
  readonly livertoxVisibleItems = this.liverToxCatalog.visibleItems;
  readonly livertoxVisibleStartIndex = this.liverToxCatalog.visibleStartIndex;
  readonly livertoxTopPaddingPx = this.liverToxCatalog.topPaddingPx;
  readonly livertoxBottomPaddingPx = this.liverToxCatalog.bottomPaddingPx;
  readonly livertoxTotal = this.liverToxCatalog.total;
  readonly livertoxLoading = this.liverToxCatalog.loading;
  readonly livertoxLoadingMore = this.liverToxCatalog.loadingMore;
  readonly livertoxHasMore = this.liverToxCatalog.hasMore;
  readonly livertoxError = this.liverToxCatalog.error;
  readonly livertoxSearchInput = this.liverToxCatalog.searchInput;

  private readonly ragCatalog = new InspectionPagedResource<InspectionRagDocumentRow>(
    (params) => fetchInspectionRagDocuments(params),
    'Failed to load RAG state.',
  );
  readonly ragDocuments = this.ragCatalog.items;
  readonly ragVisibleItems = this.ragCatalog.visibleItems;
  readonly ragVisibleStartIndex = this.ragCatalog.visibleStartIndex;
  readonly ragTopPaddingPx = this.ragCatalog.topPaddingPx;
  readonly ragBottomPaddingPx = this.ragCatalog.bottomPaddingPx;
  readonly ragTotal = this.ragCatalog.total;
  readonly ragLoading = this.ragCatalog.loading;
  readonly ragLoadingMore = this.ragCatalog.loadingMore;
  readonly ragHasMore = this.ragCatalog.hasMore;
  readonly ragError = this.ragCatalog.error;
  readonly ragSearchInput = this.ragCatalog.searchInput;
  readonly ragVectorStore = signal<InspectionRagVectorStoreSummary | null>(null);
  readonly ragSelectedFolderPath = signal('');

  readonly reportSession = signal<InspectionSessionItem | null>(null);
  readonly reportContent = signal('');
  readonly reportLoading = signal(false);
  readonly reportError = signal<string | null>(null);

  private readonly aliasDetail = new InspectionDetailResource<InspectionDrugAliasesResponse>();
  readonly aliasData = this.aliasDetail.data;
  readonly aliasLoading = this.aliasDetail.loading;
  readonly aliasError = this.aliasDetail.error;

  private readonly excerptDetail = new InspectionDetailResource<InspectionLiverToxExcerptResponse>();
  readonly excerptData = this.excerptDetail.data;
  readonly excerptLoading = this.excerptDetail.loading;
  readonly excerptError = this.excerptDetail.error;

  private readonly updateTargetActions: InspectionUpdateTargetActionsMap = {
    rxnav: {
      fetchConfig: () => fetchInspectionRxNavUpdateConfig(),
      start: (overrides) => startInspectionRxNavUpdateJob(overrides),
      status: (jobId) => fetchInspectionRxNavUpdateJobStatus(jobId),
      cancel: async (jobId) => {
        await cancelInspectionRxNavUpdateJob(jobId);
      },
      refresh: async () => {
        await this.loadRxNav();
      },
    },
    livertox: {
      fetchConfig: () => fetchInspectionLiverToxUpdateConfig(),
      start: (overrides) => startInspectionLiverToxUpdateJob(overrides),
      status: (jobId) => fetchInspectionLiverToxUpdateJobStatus(jobId),
      cancel: async (jobId) => {
        await cancelInspectionLiverToxUpdateJob(jobId);
      },
      refresh: async () => {
        await this.loadLiverTox();
      },
    },
    rag: {
      fetchConfig: () => fetchInspectionRagUpdateConfig(),
      start: (overrides) => startInspectionRagUpdateJob(overrides),
      status: (jobId) => fetchInspectionRagUpdateJobStatus(jobId),
      cancel: async (jobId) => {
        await cancelInspectionRagUpdateJob(jobId);
      },
      refresh: async () => {
        await this.loadRag();
      },
    },
  };
  private readonly updateJob = new InspectionUpdateJobResource(
    this.jobPolling,
    this.updateTargetActions,
    () => this.ragSelectedFolderPath(),
  );
  readonly activeUpdateTarget = this.updateJob.activeTarget;
  readonly updateConfig = this.updateJob.updateConfig;
  readonly updateConfigText = this.updateJob.updateConfigText;
  readonly updateLoading = this.updateJob.updateLoading;
  readonly updateRunning = this.updateJob.updateRunning;
  readonly updateJobId = this.updateJob.updateJobId;
  readonly updateProgress = this.updateJob.updateProgress;
  readonly updateMessage = this.updateJob.updateMessage;
  readonly updateError = this.updateJob.updateError;

  ngOnInit(): void {
    void this.initializePageData();
  }

  private async initializePageData(): Promise<void> {
    await Promise.all([
      this.loadSessions(),
      this.loadRxNav(),
      this.loadLiverTox(),
      this.loadRag(),
    ]);
  }

  ngOnDestroy(): void {
    this.updateJob.dispose();
  }

  formatDateTime(value: string | null): string {
    return formatDateTime(value);
  }

  formatDuration(value: number | null): string {
    return formatDuration(value);
  }

  statusLabel(value: InspectionSessionStatus): string {
    return value === 'failed' ? 'Failed' : 'Successful';
  }

  get displayedRagFolderPath(): string {
    const manualPath = this.ragSelectedFolderPath().trim();
    return manualPath || resolveRagDocumentsPath(this.ragVectorStore()) || 'N/A';
  }

  async loadSessions(): Promise<void> {
    await this.sessionCatalog.loadInitial();
  }

  async loadRxNav(): Promise<void> {
    await this.rxnavCatalog.loadInitial();
  }

  async loadLiverTox(): Promise<void> {
    await this.liverToxCatalog.loadInitial();
  }

  async loadRag(): Promise<void> {
    await this.ragCatalog.loadInitial();
    try {
      const vectorStore = await fetchInspectionRagVectorStore();
      this.ragVectorStore.set(vectorStore);
      if (!this.ragSelectedFolderPath().trim()) {
        this.ragSelectedFolderPath.set(resolveRagDocumentsPath(vectorStore));
      }
    } catch (error) {
      this.ragVectorStore.set(null);
      this.ragError.set(error instanceof Error ? error.message : 'Failed to load RAG state.');
    }
  }

  async openSessionReport(row: InspectionSessionItem): Promise<void> {
    if (!row.has_report) {
      this.reportSession.set(row);
      this.reportContent.set('');
      this.reportLoading.set(false);
      this.reportError.set('[ERROR] Report is not available for this session.');
      return;
    }
    this.reportSession.set(row);
    this.reportLoading.set(true);
    this.reportError.set(null);
    try {
      const payload = await fetchInspectionSessionReport(row.session_id);
      this.reportContent.set(payload.report);
    } catch (error) {
      this.reportContent.set('');
      this.reportError.set(error instanceof Error ? error.message : 'Failed to load session report.');
    } finally {
      this.reportLoading.set(false);
    }
  }

  closeSessionReport(): void {
    this.reportSession.set(null);
    this.reportContent.set('');
    this.reportError.set(null);
  }

  async openPatientTimeline(row: InspectionSessionItem): Promise<void> {
    if (this.canOpenTimeline(row)) {
      await this.router.navigate(['/sessions', row.session_id, 'timetable']);
    }
  }

  async confirmAndRemoveSession(row: InspectionSessionItem): Promise<void> {
    const patientLabel = row.patient_name?.trim() || 'Unknown patient';
    const confirmed = globalThis.confirm(
      `Delete session #${row.session_id} for ${patientLabel}? This action cannot be undone.`,
    );
    if (!confirmed) {
      return;
    }

    try {
      await deleteInspectionSession(row.session_id);
      await this.loadSessions();
    } catch (error) {
      this.sessionError.set(
        error instanceof Error ? error.message : 'Failed to delete session.',
      );
    }
  }

  async openAliases(drugId: number): Promise<void> {
    await this.aliasDetail.load(() => fetchInspectionRxNavAliases(drugId), 'Failed to load aliases.');
  }

  canOpenSessionReport(row: InspectionSessionItem): boolean {
    return row.has_report;
  }

  canOpenTimeline(row: InspectionSessionItem): boolean {
    return row.has_timeline || row.can_generate_timeline;
  }

  closeAliases(): void {
    this.aliasDetail.close();
  }

  async removeRxNavDrug(drugId: number): Promise<void> {
    const confirmed = globalThis.confirm(
      'Delete this drug from the RxNav catalog? This action cannot be undone.',
    );
    if (!confirmed) {
      return;
    }
    try {
      await deleteInspectionRxNavDrug(drugId);
      await this.loadRxNav();
    } catch (error) {
      this.rxnavError.set(
        error instanceof Error ? error.message : 'Failed to delete drug from RxNav catalog.',
      );
    }
  }

  async openExcerpt(drugId: number): Promise<void> {
    await this.excerptDetail.load(() => fetchInspectionLiverToxExcerpt(drugId), 'Failed to load excerpt.');
  }

  closeExcerpt(): void {
    this.excerptDetail.close();
  }

  async removeLiverToxDrug(drugId: number): Promise<void> {
    const confirmed = globalThis.confirm(
      'Delete this drug from the LiverTox catalog? This action cannot be undone.',
    );
    if (!confirmed) {
      return;
    }
    try {
      await deleteInspectionLiverToxDrug(drugId);
      await this.loadLiverTox();
    } catch (error) {
      this.livertoxError.set(
        error instanceof Error ? error.message : 'Failed to delete drug from LiverTox catalog.',
      );
    }
  }

  changeView(view: InspectionViewId): void {
    this.activeView.set(view);
  }

  inspectionTabId(view: InspectionViewId): string {
    return inspectionTabId(view);
  }

  onInspectionTabKeydown(event: KeyboardEvent, view: InspectionViewId): void {
    const currentIndex = this.inspectionViews.findIndex((item) => item.id === view);
    if (currentIndex < 0) return;

    const nextView = (() => {
      switch (event.key) {
        case 'ArrowRight':
        case 'ArrowDown':
          return this.inspectionViews[(currentIndex + 1) % this.inspectionViews.length]?.id;
        case 'ArrowLeft':
        case 'ArrowUp':
          return this.inspectionViews[(currentIndex - 1 + this.inspectionViews.length) % this.inspectionViews.length]?.id;
        case 'Home':
          return this.inspectionViews[0]?.id;
        case 'End':
          return this.inspectionViews.at(-1)?.id;
        default:
          return null;
      }
    })();

    if (!nextView) return;
    event.preventDefault();
    this.changeView(nextView);
  }

  async updateSessionsSearch(value: string): Promise<void> {
    await this.sessionCatalog.updateSearch(value);
  }

  async updateSessionsStatus(value: InspectionSessionStatus | 'all'): Promise<void> {
    this.sessionStatusFilter.set(value);
    await this.loadSessions();
  }

  async updateSessionsDateMode(value: InspectionDateFilterMode | 'none'): Promise<void> {
    this.sessionDateMode.set(value);
    await this.loadSessions();
  }

  async updateSessionsDate(value: string): Promise<void> {
    this.sessionDate.set(value);
    await this.loadSessions();
  }

  async updateRxNavSearch(value: string): Promise<void> {
    await this.rxnavCatalog.updateSearch(value);
  }

  async updateLiverToxSearch(value: string): Promise<void> {
    await this.liverToxCatalog.updateSearch(value);
  }

  async updateRagSearch(value: string): Promise<void> {
    await this.ragCatalog.updateSearch(value);
  }

  onSessionsScroll(event: Event): void {
    this.sessionCatalog.handleScrollEvent(event);
  }

  onRxNavScroll(event: Event): void {
    this.rxnavCatalog.handleScrollEvent(event);
  }

  onLiverToxScroll(event: Event): void {
    this.liverToxCatalog.handleScrollEvent(event);
  }

  onRagScroll(event: Event): void {
    this.ragCatalog.handleScrollEvent(event);
  }

  openRagFolderPicker(): void {
    this.ragFolderInput?.nativeElement.click();
  }

  handleRagFolderSelection(event: Event): void {
    const target = event.target;
    if (!(target instanceof HTMLInputElement) || !target.files || target.files.length === 0) {
      return;
    }
    const firstFile = target.files[0] as File & { path?: string; webkitRelativePath?: string };
    const webkitPath = firstFile.webkitRelativePath || '';
    const rootFolder = webkitPath.split('/')[0]?.trim() || '';
    const absoluteCandidate = this.resolveAbsoluteFolderPath(firstFile, webkitPath, rootFolder);
    const resolvedPath = absoluteCandidate || rootFolder;
    if (resolvedPath) {
      this.ragSelectedFolderPath.set(resolvedPath);
    }
  }

  private resolveAbsoluteFolderPath(
    file: File & { path?: string },
    webkitRelativePath: string,
    rootFolder: string,
  ): string {
    const filePath = typeof file.path === 'string' ? file.path.trim() : '';
    if (!filePath || !webkitRelativePath || !rootFolder) {
      return '';
    }
    const normalizedFilePath = filePath.replace(/\\/g, '/');
    const normalizedRelative = webkitRelativePath.replace(/\\/g, '/');
    if (!normalizedFilePath.toLowerCase().endsWith(normalizedRelative.toLowerCase())) {
      return '';
    }
    const base = normalizedFilePath.slice(0, normalizedFilePath.length - normalizedRelative.length);
    const slash = base.endsWith('/') ? '' : '/';
    return `${base}${slash}${rootFolder}`;
  }

  async openUpdateModal(target: InspectionUpdateTarget): Promise<void> {
    await this.updateJob.open(target);
  }

  closeUpdateModal(): void {
    this.updateJob.close();
  }

  setUpdateConfigText(value: string): void {
    this.updateJob.setConfigText(value);
  }

  async startUpdateJob(): Promise<void> {
    await this.updateJob.start();
  }

  async cancelUpdateJob(): Promise<void> {
    await this.updateJob.cancel();
  }
}

