import { CommonModule } from '@angular/common';
import { Component, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  cancelInspectionDiliPriorsUpdateJob,
  cancelInspectionDrugLabelsUpdateJob,
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRagUpdateJob,
  cancelInspectionRxNavUpdateJob,
  deleteInspectionLiverToxDrug,
  deleteInspectionRxNavDrug,
  deleteInspectionSession,
  fetchInspectionDiliPriorDetails,
  fetchInspectionDiliPriorsCatalog,
  fetchInspectionDiliPriorsUpdateConfig,
  fetchInspectionDiliPriorsUpdateJobStatus,
  fetchInspectionDrugLabelSections,
  fetchInspectionDrugLabelsCatalog,
  fetchInspectionDrugLabelsUpdateConfig,
  fetchInspectionDrugLabelsUpdateJobStatus,
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
  startInspectionDiliPriorsUpdateJob,
  startInspectionDrugLabelsUpdateJob,
  startInspectionLiverToxUpdateJob,
  startInspectionRagUpdateJob,
  startInspectionRxNavUpdateJob,
} from '../../core/services/api';
import {
  InspectionDateFilterMode,
  InspectionDiliPriorDetailResponse,
  InspectionDiliPriorItem,
  InspectionDrugAliasesResponse,
  InspectionDrugLabelItem,
  InspectionDrugLabelSectionsResponse,
  InspectionLiverToxExcerptResponse,
  InspectionLiverToxItem,
  InspectionRagDocumentRow,
  InspectionRagVectorStoreSummary,
  InspectionRxNavItem,
  InspectionSessionItem,
  InspectionSessionStatus,
  InspectionUpdateConfigResponse,
  InspectionUpdateJobStatusResponse,
} from '../../core/models/types';

const PAGE_LIMIT = 10;

type InspectionViewId = 'sessions' | 'rxnav' | 'livertox' | 'dili_priors' | 'drug_labels' | 'rag';
type InspectionUpdateTarget = 'rxnav' | 'livertox' | 'dili_priors' | 'drug_labels' | 'rag';

const INSPECTION_VIEWS: Array<{ id: InspectionViewId; label: string }> = [
  { id: 'sessions', label: 'Sessions' },
  { id: 'rxnav', label: 'Drug Catalog' },
  { id: 'livertox', label: 'LiverTox' },
  { id: 'dili_priors', label: 'DILI priors' },
  { id: 'drug_labels', label: 'Drug labels' },
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

@Component({
  selector: 'app-data-inspection-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './data-inspection-page.component.html',
  styleUrl: './data-inspection-page.component.scss',
})
export class DataInspectionPageComponent implements OnInit {
  readonly inspectionViews = INSPECTION_VIEWS;
  readonly activeView = signal<InspectionViewId>('sessions');

  readonly sessionItems = signal<InspectionSessionItem[]>([]);
  readonly sessionTotal = signal(0);
  readonly sessionOffset = signal(0);
  readonly sessionLoading = signal(false);
  readonly sessionError = signal<string | null>(null);
  readonly sessionSearchInput = signal('');
  readonly sessionStatusFilter = signal<InspectionSessionStatus | 'all'>('all');
  readonly sessionDateMode = signal<InspectionDateFilterMode | 'none'>('none');
  readonly sessionDate = signal('');

  readonly rxnavItems = signal<InspectionRxNavItem[]>([]);
  readonly rxnavTotal = signal(0);
  readonly rxnavOffset = signal(0);
  readonly rxnavLoading = signal(false);
  readonly rxnavError = signal<string | null>(null);
  readonly rxnavSearchInput = signal('');

  readonly livertoxItems = signal<InspectionLiverToxItem[]>([]);
  readonly livertoxTotal = signal(0);
  readonly livertoxOffset = signal(0);
  readonly livertoxLoading = signal(false);
  readonly livertoxError = signal<string | null>(null);
  readonly livertoxSearchInput = signal('');

  readonly diliPriorItems = signal<InspectionDiliPriorItem[]>([]);
  readonly diliPriorTotal = signal(0);
  readonly diliPriorOffset = signal(0);
  readonly diliPriorLoading = signal(false);
  readonly diliPriorError = signal<string | null>(null);
  readonly diliPriorSearchInput = signal('');

  readonly drugLabelItems = signal<InspectionDrugLabelItem[]>([]);
  readonly drugLabelTotal = signal(0);
  readonly drugLabelOffset = signal(0);
  readonly drugLabelLoading = signal(false);
  readonly drugLabelError = signal<string | null>(null);
  readonly drugLabelSearchInput = signal('');

  readonly ragDocuments = signal<InspectionRagDocumentRow[]>([]);
  readonly ragVectorStore = signal<InspectionRagVectorStoreSummary | null>(null);
  readonly ragTotal = signal(0);
  readonly ragOffset = signal(0);
  readonly ragLoading = signal(false);
  readonly ragError = signal<string | null>(null);
  readonly ragSearchInput = signal('');

  readonly reportSession = signal<InspectionSessionItem | null>(null);
  readonly reportContent = signal('');
  readonly reportLoading = signal(false);
  readonly reportError = signal<string | null>(null);

  readonly aliasData = signal<InspectionDrugAliasesResponse | null>(null);
  readonly aliasLoading = signal(false);
  readonly aliasError = signal<string | null>(null);

  readonly excerptData = signal<InspectionLiverToxExcerptResponse | null>(null);
  readonly excerptLoading = signal(false);
  readonly excerptError = signal<string | null>(null);

  readonly diliPriorDetailData = signal<InspectionDiliPriorDetailResponse | null>(null);
  readonly diliPriorDetailLoading = signal(false);
  readonly diliPriorDetailError = signal<string | null>(null);

  readonly drugLabelSectionsData = signal<InspectionDrugLabelSectionsResponse | null>(null);
  readonly drugLabelSectionsLoading = signal(false);
  readonly drugLabelSectionsError = signal<string | null>(null);

  readonly activeUpdateTarget = signal<InspectionUpdateTarget | null>(null);
  readonly updateConfig = signal<Record<string, unknown> | null>(null);
  readonly updateConfigText = signal('{}');
  readonly updateLoading = signal(false);
  readonly updateRunning = signal(false);
  readonly updateJobId = signal<string | null>(null);
  readonly updateProgress = signal(0);
  readonly updateMessage = signal('');
  readonly updateError = signal<string | null>(null);

  async ngOnInit(): Promise<void> {
    await Promise.all([
      this.loadSessions(),
      this.loadRxNav(),
      this.loadLiverTox(),
      this.loadDiliPriors(),
      this.loadDrugLabels(),
      this.loadRag(),
    ]);
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

  rangeEnd(offset: number, total: number): number {
    return Math.min(offset + PAGE_LIMIT, total);
  }

  async loadSessions(): Promise<void> {
    this.sessionLoading.set(true);
    this.sessionError.set(null);
    try {
      const statusFilter = this.sessionStatusFilter();
      const dateMode = this.sessionDateMode();
      const payload = await fetchInspectionSessions({
        search: this.sessionSearchInput(),
        status: statusFilter === 'all' ? undefined : statusFilter,
        date_mode: dateMode === 'none' ? undefined : dateMode,
        date: dateMode === 'none' ? undefined : this.sessionDate() || undefined,
        offset: this.sessionOffset(),
        limit: PAGE_LIMIT,
      });
      this.sessionItems.set(payload.items);
      this.sessionTotal.set(payload.total);
    } catch (error) {
      this.sessionItems.set([]);
      this.sessionTotal.set(0);
      this.sessionError.set(error instanceof Error ? error.message : 'Failed to load sessions.');
    } finally {
      this.sessionLoading.set(false);
    }
  }

  async loadRxNav(): Promise<void> {
    this.rxnavLoading.set(true);
    this.rxnavError.set(null);
    try {
      const payload = await fetchInspectionRxNavCatalog({
        search: this.rxnavSearchInput(),
        offset: this.rxnavOffset(),
        limit: PAGE_LIMIT,
      });
      this.rxnavItems.set(payload.items);
      this.rxnavTotal.set(payload.total);
    } catch (error) {
      this.rxnavItems.set([]);
      this.rxnavTotal.set(0);
      this.rxnavError.set(error instanceof Error ? error.message : 'Failed to load drug catalog.');
    } finally {
      this.rxnavLoading.set(false);
    }
  }

  async loadLiverTox(): Promise<void> {
    this.livertoxLoading.set(true);
    this.livertoxError.set(null);
    try {
      const payload = await fetchInspectionLiverToxCatalog({
        search: this.livertoxSearchInput(),
        offset: this.livertoxOffset(),
        limit: PAGE_LIMIT,
      });
      this.livertoxItems.set(payload.items);
      this.livertoxTotal.set(payload.total);
    } catch (error) {
      this.livertoxItems.set([]);
      this.livertoxTotal.set(0);
      this.livertoxError.set(error instanceof Error ? error.message : 'Failed to load LiverTox.');
    } finally {
      this.livertoxLoading.set(false);
    }
  }

  async loadDiliPriors(): Promise<void> {
    this.diliPriorLoading.set(true);
    this.diliPriorError.set(null);
    try {
      const payload = await fetchInspectionDiliPriorsCatalog({
        search: this.diliPriorSearchInput(),
        offset: this.diliPriorOffset(),
        limit: PAGE_LIMIT,
      });
      this.diliPriorItems.set(payload.items);
      this.diliPriorTotal.set(payload.total);
    } catch (error) {
      this.diliPriorItems.set([]);
      this.diliPriorTotal.set(0);
      this.diliPriorError.set(error instanceof Error ? error.message : 'Failed to load DILI priors.');
    } finally {
      this.diliPriorLoading.set(false);
    }
  }

  async loadDrugLabels(): Promise<void> {
    this.drugLabelLoading.set(true);
    this.drugLabelError.set(null);
    try {
      const payload = await fetchInspectionDrugLabelsCatalog({
        search: this.drugLabelSearchInput(),
        offset: this.drugLabelOffset(),
        limit: PAGE_LIMIT,
      });
      this.drugLabelItems.set(payload.items);
      this.drugLabelTotal.set(payload.total);
    } catch (error) {
      this.drugLabelItems.set([]);
      this.drugLabelTotal.set(0);
      this.drugLabelError.set(error instanceof Error ? error.message : 'Failed to load drug labels.');
    } finally {
      this.drugLabelLoading.set(false);
    }
  }

  async loadRag(): Promise<void> {
    this.ragLoading.set(true);
    this.ragError.set(null);
    try {
      const [documents, vectorStore] = await Promise.all([
        fetchInspectionRagDocuments(),
        fetchInspectionRagVectorStore(),
      ]);
      this.ragDocuments.set(documents.items);
      this.ragTotal.set(documents.total);
      this.ragVectorStore.set(vectorStore);
    } catch (error) {
      this.ragDocuments.set([]);
      this.ragTotal.set(0);
      this.ragVectorStore.set(null);
      this.ragError.set(error instanceof Error ? error.message : 'Failed to load RAG state.');
    } finally {
      this.ragLoading.set(false);
    }
  }

  async openSessionReport(row: InspectionSessionItem): Promise<void> {
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

  async removeSession(sessionId: number): Promise<void> {
    await deleteInspectionSession(sessionId);
    await this.loadSessions();
  }

  async openAliases(drugId: number): Promise<void> {
    this.aliasLoading.set(true);
    this.aliasError.set(null);
    this.aliasData.set(null);
    try {
      this.aliasData.set(await fetchInspectionRxNavAliases(drugId));
    } catch (error) {
      this.aliasError.set(error instanceof Error ? error.message : 'Failed to load aliases.');
    } finally {
      this.aliasLoading.set(false);
    }
  }

  closeAliases(): void {
    this.aliasData.set(null);
    this.aliasError.set(null);
    this.aliasLoading.set(false);
  }

  async removeRxNavDrug(drugId: number): Promise<void> {
    await deleteInspectionRxNavDrug(drugId);
    await this.loadRxNav();
  }

  async openExcerpt(drugId: number): Promise<void> {
    this.excerptLoading.set(true);
    this.excerptError.set(null);
    this.excerptData.set(null);
    try {
      this.excerptData.set(await fetchInspectionLiverToxExcerpt(drugId));
    } catch (error) {
      this.excerptError.set(error instanceof Error ? error.message : 'Failed to load excerpt.');
    } finally {
      this.excerptLoading.set(false);
    }
  }

  closeExcerpt(): void {
    this.excerptData.set(null);
    this.excerptError.set(null);
    this.excerptLoading.set(false);
  }

  async removeLiverToxDrug(drugId: number): Promise<void> {
    await deleteInspectionLiverToxDrug(drugId);
    await this.loadLiverTox();
  }

  async openDiliPriorDetails(drugId: number): Promise<void> {
    this.diliPriorDetailLoading.set(true);
    this.diliPriorDetailError.set(null);
    this.diliPriorDetailData.set(null);
    try {
      this.diliPriorDetailData.set(await fetchInspectionDiliPriorDetails(drugId));
    } catch (error) {
      this.diliPriorDetailError.set(error instanceof Error ? error.message : 'Failed to load DILI prior details.');
    } finally {
      this.diliPriorDetailLoading.set(false);
    }
  }

  closeDiliPriorDetails(): void {
    this.diliPriorDetailData.set(null);
    this.diliPriorDetailError.set(null);
    this.diliPriorDetailLoading.set(false);
  }

  async openDrugLabelSections(drugId: number): Promise<void> {
    this.drugLabelSectionsLoading.set(true);
    this.drugLabelSectionsError.set(null);
    this.drugLabelSectionsData.set(null);
    try {
      this.drugLabelSectionsData.set(await fetchInspectionDrugLabelSections(drugId));
    } catch (error) {
      this.drugLabelSectionsError.set(error instanceof Error ? error.message : 'Failed to load label sections.');
    } finally {
      this.drugLabelSectionsLoading.set(false);
    }
  }

  closeDrugLabelSections(): void {
    this.drugLabelSectionsData.set(null);
    this.drugLabelSectionsError.set(null);
    this.drugLabelSectionsLoading.set(false);
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
          return this.inspectionViews[this.inspectionViews.length - 1]?.id;
        default:
          return null;
      }
    })();

    if (!nextView) return;
    event.preventDefault();
    this.changeView(nextView);
  }

  async updateSessionsSearch(value: string): Promise<void> {
    this.sessionSearchInput.set(value);
    this.sessionOffset.set(0);
    await this.loadSessions();
  }

  async updateSessionsStatus(value: InspectionSessionStatus | 'all'): Promise<void> {
    this.sessionStatusFilter.set(value);
    this.sessionOffset.set(0);
    await this.loadSessions();
  }

  async updateSessionsDateMode(value: InspectionDateFilterMode | 'none'): Promise<void> {
    this.sessionDateMode.set(value);
    this.sessionOffset.set(0);
    await this.loadSessions();
  }

  async updateSessionsDate(value: string): Promise<void> {
    this.sessionDate.set(value);
    this.sessionOffset.set(0);
    await this.loadSessions();
  }

  async pageSessions(direction: -1 | 1): Promise<void> {
    const next = this.sessionOffset() + direction * PAGE_LIMIT;
    if (next < 0 || next >= this.sessionTotal()) return;
    this.sessionOffset.set(next);
    await this.loadSessions();
  }

  async updateRxNavSearch(value: string): Promise<void> {
    this.rxnavSearchInput.set(value);
    this.rxnavOffset.set(0);
    await this.loadRxNav();
  }

  async pageRxNav(direction: -1 | 1): Promise<void> {
    const next = this.rxnavOffset() + direction * PAGE_LIMIT;
    if (next < 0 || next >= this.rxnavTotal()) return;
    this.rxnavOffset.set(next);
    await this.loadRxNav();
  }

  async updateLiverToxSearch(value: string): Promise<void> {
    this.livertoxSearchInput.set(value);
    this.livertoxOffset.set(0);
    await this.loadLiverTox();
  }

  async pageLiverTox(direction: -1 | 1): Promise<void> {
    const next = this.livertoxOffset() + direction * PAGE_LIMIT;
    if (next < 0 || next >= this.livertoxTotal()) return;
    this.livertoxOffset.set(next);
    await this.loadLiverTox();
  }

  async updateDiliPriorsSearch(value: string): Promise<void> {
    this.diliPriorSearchInput.set(value);
    this.diliPriorOffset.set(0);
    await this.loadDiliPriors();
  }

  async pageDiliPriors(direction: -1 | 1): Promise<void> {
    const next = this.diliPriorOffset() + direction * PAGE_LIMIT;
    if (next < 0 || next >= this.diliPriorTotal()) return;
    this.diliPriorOffset.set(next);
    await this.loadDiliPriors();
  }

  async updateDrugLabelsSearch(value: string): Promise<void> {
    this.drugLabelSearchInput.set(value);
    this.drugLabelOffset.set(0);
    await this.loadDrugLabels();
  }

  async pageDrugLabels(direction: -1 | 1): Promise<void> {
    const next = this.drugLabelOffset() + direction * PAGE_LIMIT;
    if (next < 0 || next >= this.drugLabelTotal()) return;
    this.drugLabelOffset.set(next);
    await this.loadDrugLabels();
  }

  async openUpdateModal(target: InspectionUpdateTarget): Promise<void> {
    this.activeUpdateTarget.set(target);
    this.updateLoading.set(true);
    this.updateError.set(null);
    this.updateRunning.set(false);
    this.updateJobId.set(null);
    this.updateProgress.set(0);
    this.updateMessage.set('');
    try {
      const payload = await this.fetchUpdateConfig(target);
      this.updateConfig.set(payload.defaults || {});
      this.updateConfigText.set(JSON.stringify(payload.defaults || {}, null, 2));
    } catch (error) {
      this.updateConfig.set({});
      this.updateConfigText.set('{}');
      this.updateError.set(error instanceof Error ? error.message : 'Failed to load update configuration.');
    } finally {
      this.updateLoading.set(false);
    }
  }

  closeUpdateModal(): void {
    this.activeUpdateTarget.set(null);
    this.updateLoading.set(false);
    this.updateRunning.set(false);
    this.updateJobId.set(null);
    this.updateProgress.set(0);
    this.updateMessage.set('');
    this.updateError.set(null);
  }

  setUpdateConfigText(value: string): void {
    this.updateConfigText.set(value);
  }

  async startUpdateJob(): Promise<void> {
    const target = this.activeUpdateTarget();
    if (!target) {
      return;
    }
    this.updateError.set(null);
    this.updateRunning.set(true);
    this.updateProgress.set(0);
    this.updateMessage.set('Starting update job...');

    let overrides: Record<string, unknown> = {};
    const raw = this.updateConfigText().trim();
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          overrides = parsed as Record<string, unknown>;
        } else {
          this.updateError.set('Overrides must be a JSON object.');
          this.updateRunning.set(false);
          return;
        }
      } catch {
        this.updateError.set('Invalid JSON overrides.');
        this.updateRunning.set(false);
        return;
      }
    }

    try {
      const started = await this.startUpdate(target, overrides);
      this.updateJobId.set(started.job_id);
      this.updateMessage.set(started.message || 'Update running...');
      await this.pollUpdateJob(target, started.job_id);
    } catch (error) {
      this.updateRunning.set(false);
      this.updateError.set(error instanceof Error ? error.message : 'Failed to start update job.');
    }
  }

  async cancelUpdateJob(): Promise<void> {
    const target = this.activeUpdateTarget();
    const jobId = this.updateJobId();
    if (!target || !jobId) {
      return;
    }
    try {
      await this.cancelUpdate(target, jobId);
      this.updateMessage.set('Cancellation requested.');
    } catch (error) {
      this.updateError.set(error instanceof Error ? error.message : 'Failed to cancel update job.');
    }
  }

  private async fetchUpdateConfig(target: InspectionUpdateTarget): Promise<InspectionUpdateConfigResponse> {
    if (target === 'rxnav') return fetchInspectionRxNavUpdateConfig();
    if (target === 'livertox') return fetchInspectionLiverToxUpdateConfig();
    if (target === 'dili_priors') return fetchInspectionDiliPriorsUpdateConfig();
    if (target === 'drug_labels') return fetchInspectionDrugLabelsUpdateConfig();
    return fetchInspectionRagUpdateConfig();
  }

  private async startUpdate(target: InspectionUpdateTarget, overrides: Record<string, unknown>) {
    if (target === 'rxnav') return startInspectionRxNavUpdateJob(overrides);
    if (target === 'livertox') return startInspectionLiverToxUpdateJob(overrides);
    if (target === 'dili_priors') return startInspectionDiliPriorsUpdateJob(overrides);
    if (target === 'drug_labels') return startInspectionDrugLabelsUpdateJob(overrides);
    return startInspectionRagUpdateJob(overrides);
  }

  private async fetchUpdateStatus(
    target: InspectionUpdateTarget,
    jobId: string,
  ): Promise<InspectionUpdateJobStatusResponse> {
    if (target === 'rxnav') return fetchInspectionRxNavUpdateJobStatus(jobId);
    if (target === 'livertox') return fetchInspectionLiverToxUpdateJobStatus(jobId);
    if (target === 'dili_priors') return fetchInspectionDiliPriorsUpdateJobStatus(jobId);
    if (target === 'drug_labels') return fetchInspectionDrugLabelsUpdateJobStatus(jobId);
    return fetchInspectionRagUpdateJobStatus(jobId);
  }

  private async cancelUpdate(target: InspectionUpdateTarget, jobId: string): Promise<void> {
    if (target === 'rxnav') {
      await cancelInspectionRxNavUpdateJob(jobId);
      return;
    }
    if (target === 'livertox') {
      await cancelInspectionLiverToxUpdateJob(jobId);
      return;
    }
    if (target === 'dili_priors') {
      await cancelInspectionDiliPriorsUpdateJob(jobId);
      return;
    }
    if (target === 'drug_labels') {
      await cancelInspectionDrugLabelsUpdateJob(jobId);
      return;
    }
    await cancelInspectionRagUpdateJob(jobId);
  }

  private async pollUpdateJob(target: InspectionUpdateTarget, jobId: string): Promise<void> {
    while (true) {
      const status = await this.fetchUpdateStatus(target, jobId);
      this.updateProgress.set(typeof status.progress === 'number' ? status.progress : 0);
      this.updateMessage.set(
        (status.result?.progress_message as string) ||
          (status.error || `Job status: ${status.status}`),
      );

      if (status.status === 'completed') {
        this.updateRunning.set(false);
        await this.refreshTargetAfterUpdate(target);
        return;
      }
      if (status.status === 'failed' || status.status === 'cancelled') {
        this.updateRunning.set(false);
        this.updateError.set(status.error || `Update job ${status.status}.`);
        return;
      }

      await new Promise((resolve) => globalThis.setTimeout(resolve, 1200));
    }
  }

  private async refreshTargetAfterUpdate(target: InspectionUpdateTarget): Promise<void> {
    if (target === 'rxnav') {
      await this.loadRxNav();
      return;
    }
    if (target === 'livertox') {
      await this.loadLiverTox();
      return;
    }
    if (target === 'dili_priors') {
      await this.loadDiliPriors();
      return;
    }
    if (target === 'drug_labels') {
      await this.loadDrugLabels();
      return;
    }
    await this.loadRag();
  }
}

