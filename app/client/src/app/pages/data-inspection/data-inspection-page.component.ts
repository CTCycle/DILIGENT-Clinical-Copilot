import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { InspectionActionIconButtonComponent } from '../../components/inspection-action-icon-button/inspection-action-icon-button.component';
import { InspectionCatalogStatusComponent } from '../../components/inspection-catalog-status/inspection-catalog-status.component';
import { ModalShellComponent } from '../../components/modal-shell/modal-shell.component';
import { InspectionCatalogToolbarComponent } from '../../components/inspection-catalog-toolbar/inspection-catalog-toolbar.component';
import {
  deleteInspectionLiverToxDrug,
  deleteInspectionRxNavDrug,
  fetchInspectionLiverToxCatalog,
  fetchInspectionLiverToxExcerpt,
  fetchInspectionRagDocuments,
  fetchInspectionRagVectorStore,
  fetchInspectionRxNavAliases,
  fetchInspectionRxNavCatalog,
  updateInspectionRxNavDrug,
} from '../../core/services/knowledge-catalog-api';
import {
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRagUpdateJob,
  cancelInspectionRxNavUpdateJob,
  fetchInspectionLiverToxUpdateConfig,
  fetchInspectionLiverToxUpdateJobStatus,
  fetchInspectionRagUpdateConfig,
  fetchInspectionRagUpdateJobStatus,
  fetchInspectionRxNavUpdateConfig,
  fetchInspectionRxNavUpdateJobStatus,
  startInspectionLiverToxUpdateJob,
  startInspectionRagUpdateJob,
  startInspectionRxNavUpdateJob,
} from '../../core/services/inspection-jobs-api';
import { JobPollingService } from '../../core/services/job-polling.service';
import { DesktopDialogService } from '../../core/services/desktop-dialog.service';
import {
  InspectionDrugAliasesResponse,
  InspectionLiverToxExcerptResponse,
  InspectionLiverToxItem,
  InspectionRagDocumentRow,
  InspectionRagVectorizationSummary,
  InspectionRagVectorStoreSummary,
  InspectionRxNavItem,
  InspectionUpdateTarget,
} from '../../core/models/inspection-types';
import { InspectionDetailResource } from '../../core/state/inspection-detail-resource';
import { InspectionPagedResource } from '../../core/state/inspection-paged-resource';
import { InspectionUpdateJobResource, InspectionUpdateTargetActionsMap } from '../../core/state/inspection-update-job-resource';
import { InspectionUpdateJobTrackerService } from '../../core/state/inspection-update-job-tracker.service';
import {
  InspectionViewId,
  InspectionViewOption,
  formatInspectionDateTime,
  inspectionTabId,
  resolveRagDocumentsPath,
} from '../../core/utils/inspection-formatting';
import { isRecord } from '../../core/utils';
import { InspectionUpdateControlsComponent } from './components/inspection-update-controls.component';

const INSPECTION_VIEWS: InspectionViewOption[] = [
  {
    id: 'rxnav',
    label: 'Drug Catalog',
    iconPath: 'M7.5 3.75h9a2.25 2.25 0 0 1 2.25 2.25v12a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25V6a2.25 2.25 0 0 1 2.25-2.25Z M9 8.25h6 M9 12h6 M9 15.75h3',
  },
  {
    id: 'livertox',
    label: 'LiverTox',
    iconPath: 'M8.25 4.5h7.5 M10.5 4.5v5.25l-3.75 7.5a2.25 2.25 0 0 0 2.01 3.25h6.48a2.25 2.25 0 0 0 2.01-3.25l-3.75-7.5V4.5 M8.5 15h7',
  },
  {
    id: 'rag',
    label: 'RAG',
    iconPath: 'M5.25 6.75c0-1.24 2.99-2.25 6.75-2.25s6.75 1.01 6.75 2.25-2.99 2.25-6.75 2.25-6.75-1.01-6.75-2.25Z M5.25 6.75v5.25c0 1.24 2.99 2.25 6.75 2.25s6.75-1.01 6.75-2.25V6.75 M5.25 12v5.25c0 1.24 2.99 2.25 6.75 2.25s6.75-1.01 6.75-2.25V12',
  },
];

const RAG_SUMMARY_FIELDS: ReadonlyArray<{
  key: keyof InspectionRagVectorizationSummary;
  label: string;
}> = [
  { key: 'chunk_size', label: 'Chunk size' },
  { key: 'chunk_overlap', label: 'Chunk overlap' },
  { key: 'embedding_batch_size', label: 'Embedding batch size' },
  { key: 'vector_stream_batch_size', label: 'Vector stream batch size' },
  { key: 'embedding_device', label: 'Embedding device' },
  { key: 'embedding_offline_mode', label: 'Offline mode' },
];

function normalizeFolderSeparators(value: string): string {
  return value.replace(/\\/g, '/');
}

function folderBasename(value: string): string {
  const normalized = normalizeFolderSeparators(value).replace(/\/+$/g, '');
  const segments = normalized.split('/');
  return segments.at(-1)?.trim() || '';
}

@Component({
  selector: 'app-data-inspection-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ModalShellComponent,
    InspectionActionIconButtonComponent,
    InspectionCatalogStatusComponent,
    InspectionCatalogToolbarComponent,
    InspectionUpdateControlsComponent,
  ],
  templateUrl: './data-inspection-page.component.html',
  styleUrl: './data-inspection-page.component.scss',
})
export class DataInspectionPageComponent implements OnInit, OnDestroy {
  @ViewChild('ragFolderInput') private ragFolderInput?: ElementRef<HTMLInputElement>;

  private readonly jobPolling = inject(JobPollingService);
  private readonly desktopDialog = inject(DesktopDialogService);
  private readonly inspectionUpdateTracker = inject(InspectionUpdateJobTrackerService);
  readonly isTauriSurface = this.desktopDialog.isTauriSurface();
  readonly inspectionViews = INSPECTION_VIEWS;
  readonly activeView = signal<InspectionViewId>('rxnav');

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
  private readonly aliasDetail = new InspectionDetailResource<InspectionDrugAliasesResponse>();
  readonly aliasData = this.aliasDetail.data;
  readonly aliasLoading = this.aliasDetail.loading;
  readonly aliasError = this.aliasDetail.error;
  readonly rxnavEditItem = signal<InspectionRxNavItem | null>(null);
  readonly rxnavEditValue = signal('');
  readonly rxnavEditError = signal<string | null>(null);
  readonly rxnavEditSaving = signal(false);

  private readonly excerptDetail = new InspectionDetailResource<InspectionLiverToxExcerptResponse>();
  readonly excerptData = this.excerptDetail.data;
  readonly excerptLoading = this.excerptDetail.loading;
  readonly excerptError = this.excerptDetail.error;

  private readonly updateTargetActions: InspectionUpdateTargetActionsMap = {
    rxnav: {
      fetchConfig: () => fetchInspectionRxNavUpdateConfig(),
      start: (overrides) => startInspectionRxNavUpdateJob(overrides),
      status: (jobId, timeoutSeconds) => fetchInspectionRxNavUpdateJobStatus(jobId, timeoutSeconds),
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
      status: (jobId, timeoutSeconds) => fetchInspectionLiverToxUpdateJobStatus(jobId, timeoutSeconds),
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
      status: (jobId, timeoutSeconds) => fetchInspectionRagUpdateJobStatus(jobId, timeoutSeconds),
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
    inject(InspectionUpdateJobTrackerService),
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
  readonly updateTargetState = this.updateJob.targetState;
  readonly isRagUpdateModal = computed(() => this.activeUpdateTarget() === 'rag');
  readonly updateModalSubtitle = computed(() =>
    this.isRagUpdateModal()
      ? 'Review current vectorization settings. Modify RAG settings only from Model Configurations.'
      : 'Configure update parameters and monitor progress.',
  );
  readonly ragUpdateSummaryEntries = computed(() => {
    if (!this.isRagUpdateModal()) {
      return [];
    }
    const config = this.updateConfig();
    if (!isRecord(config)) {
      return [];
    }
    return RAG_SUMMARY_FIELDS.map(({ key, label }) => ({
      key,
      label,
      value: this.formatRagSummaryValue(config[key]),
    }));
  });
  readonly rxnavUpdateRunning = computed(() => this.updateTargetState().rxnav.running);
  readonly rxnavUpdateProgress = computed(() => this.updateTargetState().rxnav.progress);
  readonly rxnavUpdateMessage = computed(() => this.updateTargetState().rxnav.message);
  readonly livertoxUpdateRunning = computed(() => this.updateTargetState().livertox.running);
  readonly livertoxUpdateProgress = computed(() => this.updateTargetState().livertox.progress);
  readonly livertoxUpdateMessage = computed(() => this.updateTargetState().livertox.message);

  ngOnInit(): void {
    void this.initializePageData();
  }

  private async initializePageData(): Promise<void> {
    await this.inspectionUpdateTracker.discover();
    await Promise.all([
      this.loadRxNav(),
      this.loadLiverTox(),
      this.loadRag(),
    ]);
  }

  ngOnDestroy(): void {
    this.updateJob.dispose();
  }

  formatDateTime(value: string | null): string {
    return formatInspectionDateTime(value);
  }

  get displayedRagFolderPath(): string {
    const manualPath = this.ragSelectedFolderPath().trim();
    return manualPath || resolveRagDocumentsPath(this.ragVectorStore()) || 'N/A';
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
        const resolvedPath = resolveRagDocumentsPath(vectorStore);
        this.ragSelectedFolderPath.set(resolvedPath);
      }
    } catch (error) {
      this.ragVectorStore.set(null);
      this.ragError.set(error instanceof Error ? error.message : 'Failed to load RAG state.');
    }
  }

  async openAliases(drugId: number): Promise<void> {
    await this.aliasDetail.load(() => fetchInspectionRxNavAliases(drugId), 'Failed to load aliases.');
  }

  closeAliases(): void {
    this.aliasDetail.close();
  }

  openRxNavEdit(row: InspectionRxNavItem): void {
    this.rxnavEditItem.set(row);
    this.rxnavEditValue.set(row.drug_name);
    this.rxnavEditError.set(null);
  }

  closeRxNavEdit(): void {
    if (this.rxnavEditSaving()) {
      return;
    }
    this.rxnavEditItem.set(null);
    this.rxnavEditValue.set('');
    this.rxnavEditError.set(null);
  }

  setRxNavEditValue(value: string): void {
    this.rxnavEditValue.set(value);
    if (this.rxnavEditError()) {
      this.rxnavEditError.set(null);
    }
  }

  async saveRxNavEdit(): Promise<void> {
    const item = this.rxnavEditItem();
    const drugName = this.rxnavEditValue().trim();
    if (!item) {
      return;
    }
    if (!drugName) {
      this.rxnavEditError.set('[ERROR] Drug name is required.');
      return;
    }
    this.rxnavEditSaving.set(true);
    this.rxnavEditError.set(null);
    try {
      await updateInspectionRxNavDrug(item.drug_id, { drug_name: drugName });
      await this.loadRxNav();
      this.rxnavEditItem.set(null);
      this.rxnavEditValue.set('');
      this.rxnavEditError.set(null);
    } catch (error) {
      this.rxnavEditError.set(
        error instanceof Error ? error.message : '[ERROR] Failed to update the RxNav entry.',
      );
    } finally {
      this.rxnavEditSaving.set(false);
    }
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

  async updateRxNavSearch(value: string): Promise<void> {
    await this.rxnavCatalog.updateSearch(value);
  }

  async updateLiverToxSearch(value: string): Promise<void> {
    await this.liverToxCatalog.updateSearch(value);
  }

  async updateRagSearch(value: string): Promise<void> {
    await this.ragCatalog.updateSearch(value);
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

  async openRagFolderPicker(): Promise<void> {
    if (this.isTauriSurface) {
      try {
        const selectedPath = await this.desktopDialog.openDirectory('Select RAG documents folder');
        if (selectedPath) {
          this.ragSelectedFolderPath.set(selectedPath);
          this.ragError.set(null);
        }
      } catch {
        this.ragError.set('Unable to open the native RAG folder picker.');
      }
      return;
    }

    const input = this.ragFolderInput?.nativeElement;
    if (!input) {
      this.ragError.set('Folder picker is unavailable in this browser runtime.');
      return;
    }
    input.value = '';
    input.click();
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
    if (absoluteCandidate) {
      this.ragSelectedFolderPath.set(absoluteCandidate);
      this.ragError.set(null);
      return;
    }
    const currentPath = this.displayedRagFolderPath.trim();
    if (rootFolder && folderBasename(currentPath).toLowerCase() === rootFolder.toLowerCase()) {
      this.ragSelectedFolderPath.set(currentPath);
      this.ragError.set(null);
      return;
    }
    this.ragError.set(
      'This browser did not expose an absolute folder path from folder selection.',
    );
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
    const normalizedFilePath = normalizeFolderSeparators(filePath);
    const normalizedRelative = normalizeFolderSeparators(webkitRelativePath);
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

  private formatRagSummaryValue(value: unknown): string {
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    if (typeof value === 'number' && Number.isFinite(value)) {
      return String(value);
    }
    if (typeof value === 'string') {
      const normalized = value.trim();
      return normalized || 'Not set';
    }
    return 'Not set';
  }
}
