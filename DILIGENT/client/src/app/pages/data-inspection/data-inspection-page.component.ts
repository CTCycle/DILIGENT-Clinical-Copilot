import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ModalShellComponent } from '../../components/modal-shell/modal-shell.component';
import { InspectionCatalogToolbarComponent } from '../../components/inspection-catalog-toolbar/inspection-catalog-toolbar.component';
import { InspectionPagerComponent } from '../../components/inspection-pager/inspection-pager.component';
import {
  cancelInspectionDiliPriorsUpdateJob,
  cancelInspectionDrugLabelsUpdateJob,
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRagUpdateJob,
  cancelInspectionRxNavUpdateJob,
  deleteInspectionLiverToxDrug,
  deleteInspectionRxNavDrug,
  deleteInspectionSession,
  fetchInspectionSessionTimeline,
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
  generateInspectionSessionTimeline,
  startInspectionDiliPriorsUpdateJob,
  startInspectionDrugLabelsUpdateJob,
  startInspectionLiverToxUpdateJob,
  startInspectionRagUpdateJob,
  startInspectionRxNavUpdateJob,
} from '../../core/services/inspection-api';
import { JobPollingService } from '../../core/services/job-polling.service';
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
  InspectionSessionTimeline,
  InspectionTimelineEvent,
  InspectionTimelineEventType,
  InspectionSessionStatus,
  InspectionUpdateTarget,
  InspectionUpdateConfigResponse,
  InspectionUpdateJobStatusResponse,
  JobStartResponse,
} from '../../core/models/types';
import { InspectionDetailResource } from './inspection-detail-resource';
import { InspectionPagedResource } from './inspection-paged-resource';

const PAGE_LIMIT = 10;

type InspectionViewId =
  | 'sessions'
  | 'rxnav'
  | 'livertox'
  | 'dili_priors'
  | 'drug_labels'
  | 'rag';

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

function parseOverridesPayload(raw: string): {
  value: Record<string, unknown> | null;
  error: string | null;
} {
  const normalized = raw.trim();
  if (!normalized) {
    return { value: {}, error: null };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(normalized);
  } catch {
    return { value: null, error: 'Invalid JSON overrides.' };
  }
  if (!isRecord(parsed)) {
    return { value: null, error: 'Overrides must be a JSON object.' };
  }
  return { value: parsed, error: null };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function resolveUpdateProgressMessage(status: InspectionUpdateJobStatusResponse): string {
  const message =
    typeof status.result?.progress_message === 'string'
      ? status.result.progress_message
      : null;
  return message || status.error || `Job status: ${status.status}`;
}

function isNotFoundError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }
  return error.message.toLowerCase().includes('not found');
}

type TimelineRenderEvent = {
  raw: InspectionTimelineEvent;
  lane: 'top' | 'bottom';
  x: number;
  cardY: number;
  connectorEndY: number;
  anchorY: number;
  color: string;
  dateLabel: string;
  detailLines: string[];
};

type InspectionUpdateTargetActions = {
  fetchConfig: () => Promise<InspectionUpdateConfigResponse>;
  start: (overrides: Record<string, unknown>) => Promise<JobStartResponse>;
  status: (jobId: string) => Promise<InspectionUpdateJobStatusResponse>;
  cancel: (jobId: string) => Promise<void>;
  refresh: () => Promise<void>;
};

@Component({
  selector: 'app-data-inspection-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ModalShellComponent,
    InspectionCatalogToolbarComponent,
    InspectionPagerComponent,
  ],
  templateUrl: './data-inspection-page.component.html',
  styleUrl: './data-inspection-page.component.scss',
})
export class DataInspectionPageComponent implements OnInit, OnDestroy {
  private readonly jobPolling = inject(JobPollingService);
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

  private readonly rxnavCatalog = new InspectionPagedResource<InspectionRxNavItem>(
    (params) => fetchInspectionRxNavCatalog(params),
    'Failed to load drug catalog.',
  );
  readonly rxnavItems = this.rxnavCatalog.items;
  readonly rxnavTotal = this.rxnavCatalog.total;
  readonly rxnavOffset = this.rxnavCatalog.offset;
  readonly rxnavLoading = this.rxnavCatalog.loading;
  readonly rxnavError = this.rxnavCatalog.error;
  readonly rxnavSearchInput = this.rxnavCatalog.searchInput;

  private readonly liverToxCatalog = new InspectionPagedResource<InspectionLiverToxItem>(
    (params) => fetchInspectionLiverToxCatalog(params),
    'Failed to load LiverTox.',
  );
  readonly livertoxItems = this.liverToxCatalog.items;
  readonly livertoxTotal = this.liverToxCatalog.total;
  readonly livertoxOffset = this.liverToxCatalog.offset;
  readonly livertoxLoading = this.liverToxCatalog.loading;
  readonly livertoxError = this.liverToxCatalog.error;
  readonly livertoxSearchInput = this.liverToxCatalog.searchInput;

  private readonly diliPriorCatalog = new InspectionPagedResource<InspectionDiliPriorItem>(
    (params) => fetchInspectionDiliPriorsCatalog(params),
    'Failed to load DILI priors.',
  );
  readonly diliPriorItems = this.diliPriorCatalog.items;
  readonly diliPriorTotal = this.diliPriorCatalog.total;
  readonly diliPriorOffset = this.diliPriorCatalog.offset;
  readonly diliPriorLoading = this.diliPriorCatalog.loading;
  readonly diliPriorError = this.diliPriorCatalog.error;
  readonly diliPriorSearchInput = this.diliPriorCatalog.searchInput;

  private readonly drugLabelCatalog = new InspectionPagedResource<InspectionDrugLabelItem>(
    (params) => fetchInspectionDrugLabelsCatalog(params),
    'Failed to load drug labels.',
  );
  readonly drugLabelItems = this.drugLabelCatalog.items;
  readonly drugLabelTotal = this.drugLabelCatalog.total;
  readonly drugLabelOffset = this.drugLabelCatalog.offset;
  readonly drugLabelLoading = this.drugLabelCatalog.loading;
  readonly drugLabelError = this.drugLabelCatalog.error;
  readonly drugLabelSearchInput = this.drugLabelCatalog.searchInput;

  readonly ragDocuments = signal<InspectionRagDocumentRow[]>([]);
  readonly ragVectorStore = signal<InspectionRagVectorStoreSummary | null>(null);
  readonly ragDocumentsPathInput = signal('');
  readonly ragTotal = signal(0);
  readonly ragOffset = signal(0);
  readonly ragLoading = signal(false);
  readonly ragError = signal<string | null>(null);
  readonly ragSearchInput = signal('');

  readonly reportSession = signal<InspectionSessionItem | null>(null);
  readonly reportContent = signal('');
  readonly reportLoading = signal(false);
  readonly reportError = signal<string | null>(null);

  readonly timelineSession = signal<InspectionSessionItem | null>(null);
  readonly timelineData = signal<InspectionSessionTimeline | null>(null);
  readonly timelineLoading = signal(false);
  readonly timelineError = signal<string | null>(null);
  private readonly timelineCache = new Map<number, InspectionSessionTimeline>();
  readonly timelineAxisY = 220;
  readonly timelineChartHeight = 500;
  readonly timelineChartPaddingX = 90;
  readonly timelineCardWidth = 232;
  readonly timelineCardHeight = 96;

  private readonly aliasDetail = new InspectionDetailResource<InspectionDrugAliasesResponse>();
  readonly aliasData = this.aliasDetail.data;
  readonly aliasLoading = this.aliasDetail.loading;
  readonly aliasError = this.aliasDetail.error;

  private readonly excerptDetail = new InspectionDetailResource<InspectionLiverToxExcerptResponse>();
  readonly excerptData = this.excerptDetail.data;
  readonly excerptLoading = this.excerptDetail.loading;
  readonly excerptError = this.excerptDetail.error;

  private readonly diliPriorDetail = new InspectionDetailResource<InspectionDiliPriorDetailResponse>();
  readonly diliPriorDetailData = this.diliPriorDetail.data;
  readonly diliPriorDetailLoading = this.diliPriorDetail.loading;
  readonly diliPriorDetailError = this.diliPriorDetail.error;

  private readonly drugLabelSectionsDetail =
    new InspectionDetailResource<InspectionDrugLabelSectionsResponse>();
  readonly drugLabelSectionsData = this.drugLabelSectionsDetail.data;
  readonly drugLabelSectionsLoading = this.drugLabelSectionsDetail.loading;
  readonly drugLabelSectionsError = this.drugLabelSectionsDetail.error;

  readonly activeUpdateTarget = signal<InspectionUpdateTarget | null>(null);
  readonly updateConfig = signal<Record<string, unknown> | null>(null);
  readonly updateConfigText = signal('{}');
  readonly updateLoading = signal(false);
  readonly updateRunning = signal(false);
  readonly updateJobId = signal<string | null>(null);
  readonly updateProgress = signal(0);
  readonly updateMessage = signal('');
  readonly updateError = signal<string | null>(null);
  private updatePollToken = 0;
  private readonly updateTargetActions: Record<
    InspectionUpdateTarget,
    InspectionUpdateTargetActions
  > = {
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
    dili_priors: {
      fetchConfig: () => fetchInspectionDiliPriorsUpdateConfig(),
      start: (overrides) => startInspectionDiliPriorsUpdateJob(overrides),
      status: (jobId) => fetchInspectionDiliPriorsUpdateJobStatus(jobId),
      cancel: async (jobId) => {
        await cancelInspectionDiliPriorsUpdateJob(jobId);
      },
      refresh: async () => {
        await this.loadDiliPriors();
      },
    },
    drug_labels: {
      fetchConfig: () => fetchInspectionDrugLabelsUpdateConfig(),
      start: (overrides) => startInspectionDrugLabelsUpdateJob(overrides),
      status: (jobId) => fetchInspectionDrugLabelsUpdateJobStatus(jobId),
      cancel: async (jobId) => {
        await cancelInspectionDrugLabelsUpdateJob(jobId);
      },
      refresh: async () => {
        await this.loadDrugLabels();
      },
    },
    rag: {
      fetchConfig: () => fetchInspectionRagUpdateConfig(),
      start: (overrides) =>
        startInspectionRagUpdateJob({
          ...overrides,
          documents_path: this.ragDocumentsPathInput().trim() || undefined,
        }),
      status: (jobId) => fetchInspectionRagUpdateJobStatus(jobId),
      cancel: async (jobId) => {
        await cancelInspectionRagUpdateJob(jobId);
      },
      refresh: async () => {
        await this.loadRag();
      },
    },
  };

  ngOnInit(): void {
    void this.initializePageData();
  }

  private async initializePageData(): Promise<void> {
    await Promise.all([
      this.loadSessions(),
      this.loadRxNav(),
      this.loadLiverTox(),
      this.loadDiliPriors(),
      this.loadDrugLabels(),
      this.loadRag(),
    ]);
  }

  ngOnDestroy(): void {
    this.cancelActiveUpdatePolling();
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
    const manualPath = this.ragDocumentsPathInput().trim();
    return manualPath || resolveRagDocumentsPath(this.ragVectorStore()) || 'N/A';
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
    await this.rxnavCatalog.load();
  }

  async loadLiverTox(): Promise<void> {
    await this.liverToxCatalog.load();
  }

  async loadDiliPriors(): Promise<void> {
    await this.diliPriorCatalog.load();
  }

  async loadDrugLabels(): Promise<void> {
    await this.drugLabelCatalog.load();
  }

  async loadRag(): Promise<void> {
    this.ragLoading.set(true);
    this.ragError.set(null);
    try {
      const [documents, vectorStore] = await Promise.all([
        fetchInspectionRagDocuments(),
        fetchInspectionRagVectorStore(),
      ]);

      const normalizedSearch = this.ragSearchInput().trim().toLowerCase();
      const filtered = documents.items.filter((item) => {
        if (!normalizedSearch) {
          return true;
        }
        const haystack = `${item.file_name} ${item.path} ${item.extension}`.toLowerCase();
        return haystack.includes(normalizedSearch);
      });

      let offset = this.ragOffset();
      if (offset >= filtered.length && filtered.length > 0) {
        offset = Math.max(0, filtered.length - PAGE_LIMIT);
        this.ragOffset.set(offset);
      }

      this.ragDocuments.set(filtered.slice(offset, offset + PAGE_LIMIT));
      this.ragTotal.set(filtered.length);
      this.ragVectorStore.set(vectorStore);
      if (!this.ragDocumentsPathInput().trim()) {
        this.ragDocumentsPathInput.set(resolveRagDocumentsPath(vectorStore));
      }
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

  async openPatientTimeline(row: InspectionSessionItem): Promise<void> {
    this.timelineSession.set(row);
    this.timelineError.set(null);
    const cached = this.timelineCache.get(row.session_id) || null;
    if (cached) {
      this.timelineData.set(cached);
      this.timelineLoading.set(false);
      return;
    }

    this.timelineLoading.set(true);
    this.timelineData.set(null);
    try {
      const timeline = await fetchInspectionSessionTimeline(row.session_id);
      this.timelineCache.set(row.session_id, timeline);
      this.timelineData.set(timeline);
    } catch (error) {
      if (isNotFoundError(error)) {
        try {
          const timeline = await generateInspectionSessionTimeline(row.session_id);
          this.timelineCache.set(row.session_id, timeline);
          this.timelineData.set(timeline);
        } catch (generationError) {
          this.timelineError.set(
            generationError instanceof Error
              ? generationError.message
              : 'Failed to generate patient timeline.',
          );
        }
      } else {
        this.timelineError.set(
          error instanceof Error ? error.message : 'Failed to load patient timeline.',
        );
      }
    } finally {
      this.timelineLoading.set(false);
    }
  }

  closePatientTimeline(): void {
    this.timelineSession.set(null);
    this.timelineLoading.set(false);
    this.timelineError.set(null);
  }

  async regeneratePatientTimeline(): Promise<void> {
    const session = this.timelineSession();
    if (!session) {
      return;
    }
    this.timelineLoading.set(true);
    this.timelineError.set(null);
    try {
      const timeline = await generateInspectionSessionTimeline(session.session_id, {
        force_regenerate: true,
      });
      this.timelineCache.set(session.session_id, timeline);
      this.timelineData.set(timeline);
    } catch (error) {
      this.timelineError.set(
        error instanceof Error ? error.message : 'Failed to regenerate patient timeline.',
      );
    } finally {
      this.timelineLoading.set(false);
    }
  }

  timelineSvgWidth(): number {
    const eventCount = this.timelineData()?.events.length ?? 0;
    return Math.max(1120, 280 + eventCount * 220);
  }

  timelineRenderEvents(): TimelineRenderEvent[] {
    const timeline = this.timelineData();
    if (!timeline || timeline.events.length === 0) {
      return [];
    }
    const colors: Record<InspectionTimelineEventType, string> = {
      therapy: '#0f766e',
      disease: '#b91c1c',
      lab: '#0369a1',
      other: '#57534e',
    };
    const events = [...timeline.events].sort((a, b) => a.sort_order - b.sort_order);
    const count = events.length;
    const width = this.timelineSvgWidth();
    const startX = this.timelineChartPaddingX;
    const endX = width - this.timelineChartPaddingX;
    const range = Math.max(endX - startX, 1);

    return events.map((event, index) => {
      const lane: 'top' | 'bottom' = index % 2 === 0 ? 'top' : 'bottom';
      const x = count === 1 ? startX + range / 2 : startX + (range * index) / (count - 1);
      const anchorY = this.timelineAxisY;
      const connectorEndY = lane === 'top' ? anchorY - 70 : anchorY + 70;
      const cardY = lane === 'top' ? 20 : this.timelineAxisY + 86;
      const dateLabel = event.event_date || event.relative_time || 'Date not reported';
      const detailLines = this.buildTimelineDetailLines(event);
      return {
        raw: event,
        lane,
        x,
        cardY,
        connectorEndY,
        anchorY,
        color: colors[event.event_type] || colors.other,
        dateLabel,
        detailLines,
      };
    });
  }

  private buildTimelineDetailLines(event: InspectionTimelineEvent): string[] {
    const fragments: string[] = [];
    if (event.description && event.description.trim()) {
      fragments.push(event.description.trim());
    }
    if (event.source && event.source.trim()) {
      fragments.push(`Source: ${event.source.trim()}`);
    }
    if (typeof event.confidence === 'number' && Number.isFinite(event.confidence)) {
      fragments.push(`Confidence: ${Math.round(event.confidence * 100)}%`);
    }
    if (fragments.length === 0) {
      return ['No additional details.'];
    }
    return this.wrapTimelineText(fragments.join(' • '), 38, 3);
  }

  private wrapTimelineText(text: string, maxCharsPerLine: number, maxLines: number): string[] {
    const normalized = text.replace(/\s+/g, ' ').trim();
    if (!normalized) {
      return [];
    }
    const words = normalized.split(' ');
    const lines: string[] = [];
    let current = '';
    for (const word of words) {
      const candidate = current ? `${current} ${word}` : word;
      if (candidate.length <= maxCharsPerLine) {
        current = candidate;
        continue;
      }
      if (current) {
        lines.push(current);
      }
      current = word;
      if (lines.length >= maxLines) {
        break;
      }
    }
    if (current && lines.length < maxLines) {
      lines.push(current);
    }
    if (words.length > 0 && lines.length > 0 && lines.length === maxLines) {
      const consumed = lines.join(' ');
      if (consumed.length < normalized.length) {
        const last = lines[maxLines - 1] ?? '';
        lines[maxLines - 1] = last.length > 3 ? `${last.slice(0, maxCharsPerLine - 1)}…` : `${last}…`;
      }
    }
    return lines;
  }

  exportTimelinePng(): void {
    const element = document.getElementById('inspection-timeline-svg');
    if (!(element instanceof SVGSVGElement)) {
      this.timelineError.set('Timeline canvas is unavailable for export.');
      return;
    }
    const svg = element;

    const serializer = new XMLSerializer();
    const source = serializer.serializeToString(svg);
    const blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const image = new Image();

    image.onload = () => {
      const width = this.timelineSvgWidth();
      const height = this.timelineChartHeight;
      const canvas = document.createElement('canvas');
      canvas.width = width * 2;
      canvas.height = height * 2;
      const context = canvas.getContext('2d');
      if (!context) {
        URL.revokeObjectURL(url);
        this.timelineError.set('Failed to initialize export renderer.');
        return;
      }
      context.scale(2, 2);
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, width, height);
      context.drawImage(image, 0, 0, width, height);
      canvas.toBlob((pngBlob) => {
        if (!pngBlob) {
          this.timelineError.set('Failed to encode PNG export.');
          URL.revokeObjectURL(url);
          return;
        }
        const downloadUrl = URL.createObjectURL(pngBlob);
        const link = document.createElement('a');
        const sessionId = this.timelineSession()?.session_id ?? 'session';
        link.href = downloadUrl;
        link.download = `patient-timeline-${sessionId}.png`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(downloadUrl);
        URL.revokeObjectURL(url);
      }, 'image/png');
    };

    image.onerror = () => {
      URL.revokeObjectURL(url);
      this.timelineError.set('Failed to render timeline for export.');
    };

    image.src = url;
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
      this.timelineCache.delete(row.session_id);
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

  async openDiliPriorDetails(drugId: number): Promise<void> {
    await this.diliPriorDetail.load(
      () => fetchInspectionDiliPriorDetails(drugId),
      'Failed to load DILI prior details.',
    );
  }

  closeDiliPriorDetails(): void {
    this.diliPriorDetail.close();
  }

  async openDrugLabelSections(drugId: number): Promise<void> {
    await this.drugLabelSectionsDetail.load(
      () => fetchInspectionDrugLabelSections(drugId),
      'Failed to load label sections.',
    );
  }

  closeDrugLabelSections(): void {
    this.drugLabelSectionsDetail.close();
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
    await this.rxnavCatalog.updateSearch(value);
  }

  async pageRxNav(direction: -1 | 1): Promise<void> {
    await this.rxnavCatalog.page(direction);
  }

  async updateLiverToxSearch(value: string): Promise<void> {
    await this.liverToxCatalog.updateSearch(value);
  }

  async pageLiverTox(direction: -1 | 1): Promise<void> {
    await this.liverToxCatalog.page(direction);
  }

  async updateDiliPriorsSearch(value: string): Promise<void> {
    await this.diliPriorCatalog.updateSearch(value);
  }

  async pageDiliPriors(direction: -1 | 1): Promise<void> {
    await this.diliPriorCatalog.page(direction);
  }

  async updateDrugLabelsSearch(value: string): Promise<void> {
    await this.drugLabelCatalog.updateSearch(value);
  }

  async pageDrugLabels(direction: -1 | 1): Promise<void> {
    await this.drugLabelCatalog.page(direction);
  }

  async updateRagSearch(value: string): Promise<void> {
    this.ragSearchInput.set(value);
    this.ragOffset.set(0);
    await this.loadRag();
  }

  async pageRag(direction: -1 | 1): Promise<void> {
    const next = this.ragOffset() + direction * PAGE_LIMIT;
    if (next < 0 || next >= this.ragTotal()) return;
    this.ragOffset.set(next);
    await this.loadRag();
  }

  setRagDocumentsPathInput(value: string): void {
    this.ragDocumentsPathInput.set(value);
  }

  async openUpdateModal(target: InspectionUpdateTarget): Promise<void> {
    this.cancelActiveUpdatePolling();
    this.activeUpdateTarget.set(target);
    this.updateLoading.set(true);
    this.updateError.set(null);
    this.updateRunning.set(false);
    this.updateJobId.set(null);
    this.updateProgress.set(0);
    this.updateMessage.set('');
    try {
      const payload = await this.updateTargetActions[target].fetchConfig();
      const defaults = { ...(payload.defaults ?? undefined) };
      if (target === 'rag' && this.ragDocumentsPathInput().trim()) {
        defaults['documents_path'] = this.ragDocumentsPathInput().trim();
      }
      this.updateConfig.set(defaults);
      this.updateConfigText.set(JSON.stringify(defaults, null, 2));
    } catch (error) {
      this.updateConfig.set({});
      this.updateConfigText.set('{}');
      this.updateError.set(error instanceof Error ? error.message : 'Failed to load update configuration.');
    } finally {
      this.updateLoading.set(false);
    }
  }

  closeUpdateModal(): void {
    this.cancelActiveUpdatePolling();
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

    const parsedOverrides = parseOverridesPayload(this.updateConfigText());
    if (parsedOverrides.error) {
      this.updateError.set(parsedOverrides.error);
      this.updateRunning.set(false);
      return;
    }
    const overrides = parsedOverrides.value ?? {};

    try {
      const started = await this.updateTargetActions[target].start(overrides);
      this.updateJobId.set(started.job_id);
      this.updateMessage.set(started.message || 'Update running...');
      const pollToken = this.beginUpdatePolling();
      await this.pollUpdateJob(target, started.job_id, pollToken);
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
      await this.updateTargetActions[target].cancel(jobId);
      this.updateMessage.set('Cancellation requested.');
    } catch (error) {
      this.updateError.set(error instanceof Error ? error.message : 'Failed to cancel update job.');
    }
  }

  private async pollUpdateJob(
    target: InspectionUpdateTarget,
    jobId: string,
    pollToken: number,
  ): Promise<void> {
    await this.jobPolling.run({
      intervalMs: 1200,
      isCancelled: () => !this.isUpdatePollingActive(pollToken),
      pollStep: async () => {
        const status = await this.updateTargetActions[target].status(jobId);
        if (!this.isUpdatePollingActive(pollToken)) {
          return false;
        }
        this.updateProgress.set(typeof status.progress === 'number' ? status.progress : 0);
        this.updateMessage.set(resolveUpdateProgressMessage(status));

        if (status.status === 'completed') {
          this.updateRunning.set(false);
          await this.updateTargetActions[target].refresh();
          return false;
        }
        if (status.status === 'failed' || status.status === 'cancelled') {
          this.updateRunning.set(false);
          this.updateError.set(status.error || `Update job ${status.status}.`);
          return false;
        }

        return true;
      },
    });
  }

  private beginUpdatePolling(): number {
    this.updatePollToken += 1;
    return this.updatePollToken;
  }

  private cancelActiveUpdatePolling(): void {
    this.updatePollToken += 1;
    this.updateRunning.set(false);
  }

  private isUpdatePollingActive(pollToken: number): boolean {
    return this.updatePollToken === pollToken;
  }
}

