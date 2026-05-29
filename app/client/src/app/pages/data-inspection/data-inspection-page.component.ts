import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { InspectionActionIconButtonComponent } from '../../components/inspection-action-icon-button/inspection-action-icon-button.component';
import { ModalShellComponent } from '../../components/modal-shell/modal-shell.component';
import { InspectionCatalogToolbarComponent } from '../../components/inspection-catalog-toolbar/inspection-catalog-toolbar.component';
import {
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRagUpdateJob,
  cancelInspectionRxNavUpdateJob,
  deleteInspectionLiverToxDrug,
  deleteInspectionRxNavDrug,
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
  startInspectionLiverToxUpdateJob,
  startInspectionRagUpdateJob,
  startInspectionRxNavUpdateJob,
} from '../../core/services/inspection-api';
import { JobPollingService } from '../../core/services/job-polling.service';
import {
  InspectionDrugAliasesResponse,
  InspectionLiverToxExcerptResponse,
  InspectionLiverToxItem,
  InspectionRagDocumentRow,
  InspectionRagVectorStoreSummary,
  InspectionRxNavItem,
  InspectionUpdateTarget,
} from '../../core/models/types';
import { InspectionDetailResource } from '../../core/state/inspection-detail-resource';
import { InspectionPagedResource } from '../../core/state/inspection-paged-resource';
import { InspectionUpdateJobResource, InspectionUpdateTargetActionsMap } from '../../core/state/inspection-update-job-resource';
import {
  InspectionViewId,
  InspectionViewOption,
  formatInspectionDateTime,
  inspectionTabId,
  resolveRagDocumentsPath,
} from '../../core/utils/inspection-formatting';

const INSPECTION_VIEWS: InspectionViewOption[] = [
  { id: 'rxnav', label: 'Drug Catalog' },
  { id: 'livertox', label: 'LiverTox' },
  { id: 'rag', label: 'RAG' },
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
    InspectionCatalogToolbarComponent,
  ],
  templateUrl: './data-inspection-page.component.html',
  styleUrl: './data-inspection-page.component.scss',
})
export class DataInspectionPageComponent implements OnInit, OnDestroy {
  @ViewChild('ragFolderInput') private ragFolderInput?: ElementRef<HTMLInputElement>;

  private readonly jobPolling = inject(JobPollingService);
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
  readonly updateTargetState = this.updateJob.targetState;
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

  openRagFolderPicker(): void {
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
}

