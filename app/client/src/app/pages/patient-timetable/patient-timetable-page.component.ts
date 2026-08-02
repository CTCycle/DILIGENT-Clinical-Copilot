import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, DestroyRef, ElementRef, HostListener, OnInit, ViewChild, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';

import {
  fetchInspectionSessionTimelineById,
  fetchInspectionSessionTimelineList,
  generateInspectionSessionTimeline,
} from '../../core/services/inspection-api';
import {
  InspectionSessionTimeline,
  InspectionTimelineEvent,
  InspectionTimelineEventType,
  InspectionTimelineTimingType,
} from '../../core/models/types';
import {
  createTimelineScale,
  dayToUtcDate,
  normalizeTimelineDate,
  NormalizedTimelineDate,
  TimelineDatePrecision,
} from './timeline-date';
import { packTimelineItems, TimelineCluster } from './timeline-layout';
import {
  TimetableFilterOption,
  TimetableFilterSelectComponent,
} from './components/timetable-filter-select.component';

type TimetableLane = 'clinical' | 'therapy' | 'labs' | 'uncertainty' | 'unanchored';
type TimelineCardAlign = 'start' | 'center' | 'end';
type TimelineEvidenceFilter = 'all' | 'with_evidence' | 'missing_evidence';
type TimelineDensity = 'compact' | 'comfortable' | 'dense';

type RenderedTimelineEvent = {
  event: InspectionTimelineEvent;
  lane: TimetableLane;
  color: string;
  positionPercent: number | null;
  rangeStartPercent: number | null;
  rangeEndPercent: number | null;
  align: TimelineCardAlign;
  stackLevel: number;
};

type TimelineDateRange = {
  start: Date | null;
  end: Date | null;
};

type ResolvedTimelineDate = {
  date: Date;
  precision: TimelineDatePrecision;
};

type TimelineScalePoint = {
  label: string;
  shortLabel: string;
  positionPercent: number;
  isYearBoundary: boolean;
};

const EVENT_COLORS: Record<InspectionTimelineEventType, string> = {
  therapy: '#0f8f83',
  disease: '#dc2626',
  lab: '#1266c3',
  other: '#6b7280',
};

const TIMING_LABELS: Record<InspectionTimelineTimingType, string> = {
  explicit_date: 'Explicit date',
  relative: 'Relative',
  duration: 'Duration',
  recurring: 'Recurring',
  uncertain: 'Uncertain',
  ordering: 'Ordering',
};

const TIMETABLE_LANE_LABELS: Record<TimetableLane, string> = {
  clinical: 'Clinical',
  therapy: 'Medications',
  labs: 'Labs',
  uncertainty: 'Uncertain',
  unanchored: 'Unanchored',
};

const TIMETABLE_LANES: TimetableLane[] = ['clinical', 'therapy', 'labs', 'uncertainty', 'unanchored'];
const TIMELINE_MONTH_FORMATTER = new Intl.DateTimeFormat('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
const TIMELINE_MONTH_LABEL_FORMATTER = new Intl.DateTimeFormat('en-US', { month: 'short', timeZone: 'UTC' });
const EVIDENCE_FILTER_OPTIONS: readonly TimetableFilterOption[] = [
  { value: 'all', label: 'All events' },
  { value: 'with_evidence', label: 'Has source evidence' },
  { value: 'missing_evidence', label: 'Missing source evidence' },
];
const DENSITY_OPTIONS: readonly TimetableFilterOption[] = [
  { value: 'dense', label: 'Dense' },
  { value: 'compact', label: 'Compact' },
  { value: 'comfortable', label: 'Comfortable' },
];

function isTimelineEvidenceFilter(value: string): value is TimelineEvidenceFilter {
  return value === 'all' || value === 'with_evidence' || value === 'missing_evidence';
}

function isTimelineDensity(value: string): value is TimelineDensity {
  return value === 'dense' || value === 'compact' || value === 'comfortable';
}

@Component({
  selector: 'app-patient-timetable-page',
  standalone: true,
  imports: [CommonModule, RouterLink, TimetableFilterSelectComponent],
  templateUrl: './patient-timetable-page.component.html',
  styleUrl: './patient-timetable-page.component.scss',
})
export class PatientTimetablePageComponent implements OnInit, AfterViewInit {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  readonly sessionId = signal<number | null>(null);
  readonly timelineId = signal<number | null>(null);
  readonly timeline = signal<InspectionSessionTimeline | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly selectedEventId = signal<string | null>(null);
  readonly selectedCluster = signal<TimelineCluster | null>(null);
  readonly visibleLanes = signal<Record<TimetableLane, boolean>>({ clinical: true, therapy: true, labs: true, uncertainty: true, unanchored: true });
  readonly collapsedLanes = signal<Record<TimetableLane, boolean>>({ clinical: false, therapy: false, labs: false, uncertainty: false, unanchored: false });
  readonly evidenceFilter = signal<TimelineEvidenceFilter>('all');
  readonly showUncertainEvents = signal(true);
  readonly hideEmptyLanes = signal(false);
  readonly density = signal<TimelineDensity>('comfortable');
  readonly zoom = signal(1);
  readonly canvasWidth = signal(1200);
  readonly scrollOffset = signal(0);
  readonly scrollMax = signal(0);
  readonly selectionAnnouncement = signal('');
  readonly evidenceFilterOptions = EVIDENCE_FILTER_OPTIONS;
  readonly densityOptions = DENSITY_OPTIONS;
  readonly isNarrowScreen = signal(globalThis.matchMedia?.('(max-width: 640px)').matches ?? false);
  private lastFocusedElement: HTMLElement | null = null;
  @ViewChild('timelineArea') private timelineArea?: ElementRef<HTMLElement>;
  @ViewChild('timelineCanvas') private timelineCanvas?: ElementRef<HTMLElement>;

  readonly orderedEvents = computed(() => [...(this.timeline()?.events ?? [])].sort((a, b) => {
    const left = normalizeTimelineDate(a.event_date);
    const right = normalizeTimelineDate(b.event_date);
    if (left && right) return left.startDay - right.startDay || left.endDay - right.endDay || a.sort_order - b.sort_order || a.event_id.localeCompare(b.event_id);
    if (left) return -1;
    if (right) return 1;
    return a.sort_order - b.sort_order || a.event_id.localeCompare(b.event_id);
  }));

  readonly dateRange = computed<TimelineDateRange>(() => {
    const dates = this.orderedEvents().map((event) => this.resolveNormalizedRange(event)).filter((value): value is NormalizedTimelineDate => value !== null);
    return {
      start: dates.length ? dayToUtcDate(Math.min(...dates.map((date) => date.startDay))) : null,
      end: dates.length ? dayToUtcDate(Math.max(...dates.map((date) => date.endDay))) : null,
    };
  });

  readonly timelineScale = computed(() => {
    const range = this.dateRange();
    return range.start && range.end
      ? createTimelineScale(Math.floor(range.start.getTime() / 86_400_000), Math.floor(range.end.getTime() / 86_400_000))
      : null;
  });

  readonly renderedEvents = computed<RenderedTimelineEvent[]>(() => {
    const events = this.filteredEvents();
    const range = this.dateRange();
    const items = events.map((event) => {
      const positionPercent = this.resolveEventPosition(event, range);
      const normalized = this.resolveNormalizedRange(event);
      const scale = this.timelineScale();
      return {
        event,
        lane: this.resolveLane(event),
        color: EVENT_COLORS[event.event_type] ?? EVENT_COLORS.other,
        positionPercent,
        rangeStartPercent: normalized && scale ? scale.toPercent(normalized.startDay) : null,
        rangeEndPercent: normalized && scale ? scale.toPercent(normalized.endDay) : null,
        align: this.resolveEventAlign(positionPercent),
        stackLevel: 0,
      };
    });
    return this.assignStackLevels(items);
  });

  readonly laneClusters = computed<Record<TimetableLane, TimelineCluster[]>>(() => {
    const result = {} as Record<TimetableLane, TimelineCluster[]>;
    for (const lane of TIMETABLE_LANES) {
      const items = this.renderedEvents().filter((item) => item.lane === lane && item.positionPercent !== null);
      result[lane] = packTimelineItems(items.map((item) => ({
        id: item.event.event_id,
        positionPercent: item.positionPercent ?? 0,
        width: this.density() === 'dense' ? 104 : this.density() === 'compact' ? 132 : 176,
        sortOrder: item.event.sort_order,
      })), 1200).clusters;
    }
    return result;
  });

  readonly clusteredEventIds = computed(() => new Set(Object.values(this.laneClusters()).flatMap((clusters) => clusters.flatMap((cluster) => cluster.memberIds))));

  readonly lanes = computed(() =>
    TIMETABLE_LANES.map((lane) => {
      const items = this.renderedEvents().filter((item) => item.lane === lane && !this.clusteredEventIds().has(item.event.event_id));
      const eventCount = this.renderedEvents().filter((item) => item.lane === lane).length;
      const levelCount = Math.max(1, ...items.map((item) => item.stackLevel + 1));
      return {
        id: lane,
        label: TIMETABLE_LANE_LABELS[lane],
        items,
        eventCount,
        levelCount,
        collapsed: this.collapsedLanes()[lane],
        visible: this.visibleLanes()[lane],
        clusters: this.laneClusters()[lane],
      };
    }),
  );

  readonly selectedEvent = computed(() => {
    const selectedId = this.selectedEventId();
    return this.orderedEvents().find((event) => event.event_id === selectedId) ?? null;
  });

  readonly rangeLabel = computed(() => {
    const range = this.dateRange();
    if (!range.start || !range.end) {
      return 'Timeline events';
    }
    if (this.monthLabel(range.start) === this.monthLabel(range.end)) {
      return this.monthLabel(range.start);
    }
    return `${this.monthLabel(range.start)} - ${this.monthLabel(range.end)}`;
  });

  readonly scalePoints = computed<TimelineScalePoint[]>(() => {
    const range = this.dateRange();
    if (!range.start || !range.end) {
      return [
        { label: 'Start', shortLabel: 'Start', positionPercent: 0, isYearBoundary: false },
        { label: 'Middle', shortLabel: 'Middle', positionPercent: 50, isYearBoundary: false },
        { label: 'End', shortLabel: 'End', positionPercent: 100, isYearBoundary: false },
      ];
    }
    const scale = this.timelineScale();
    if (!scale) return [
      { label: 'Start', shortLabel: 'Start', positionPercent: 0, isYearBoundary: false },
      { label: 'End', shortLabel: 'End', positionPercent: 100, isYearBoundary: false },
    ];
    const start = new Date(Date.UTC(range.start.getUTCFullYear(), range.start.getUTCMonth(), 1));
    const end = new Date(Date.UTC(range.end.getUTCFullYear(), range.end.getUTCMonth(), 1));
    const points: TimelineScalePoint[] = [];
    const totalMonths = Math.max(1, this.monthsBetweenCount(start, end));
    const cursor = new Date(start);
    let offset = 0;
    while (cursor.getTime() <= end.getTime() && points.length < 24) {
      const label = this.monthLabel(cursor);
      points.push({
        label,
        shortLabel: TIMELINE_MONTH_LABEL_FORMATTER.format(cursor),
        positionPercent: scale.toPercent(Math.floor(cursor.getTime() / 86_400_000)),
        isYearBoundary: cursor.getUTCMonth() === 0 || offset === 0,
      });
      cursor.setUTCMonth(cursor.getUTCMonth() + 1);
      offset += 1;
    }
    if (points.length === 1) {
      return [
        { ...points[0], positionPercent: 0 },
        { ...points[0], positionPercent: 100 },
      ];
    }
    return points.map((point, index) => ({
      ...point,
      positionPercent: index === points.length - 1 ? 100 : point.positionPercent,
    }));
  });

  readonly isFallbackTimeline = computed(() => this.timeline()?.generation_status === 'fallback');

  readonly generationNote = computed(() => {
    const timeline = this.timeline();
    if (!timeline) {
      return null;
    }
    if (timeline.generation_status === 'fallback') {
      return timeline.generation_note || 'This timetable uses deterministic fallback events because local model extraction was unavailable.';
    }
    const sourceKind = timeline.source_kind === 'cloud' ? 'cloud' : 'local';
    return timeline.generation_note || `Generated with the configured ${sourceKind} timeline extraction model.`;
  });
  readonly visibleLaneRows = computed(() => this.lanes().filter((lane) => lane.visible && (!this.hideEmptyLanes() || lane.items.length > 0)));
  readonly selectedEventIndex = computed(() => this.filteredEvents().findIndex((event) => event.event_id === this.selectedEventId()));
  readonly hasPreviousEvent = computed(() => this.selectedEventIndex() > 0);
  readonly hasNextEvent = computed(() => this.selectedEventIndex() >= 0 && this.selectedEventIndex() < this.filteredEvents().length - 1);
  readonly linkedSelectedEventIds = computed(() => new Set(this.selectedEvent()?.linked_patient_event_ids ?? []));

  readonly filteredEvents = computed(() => this.orderedEvents().filter((event) => {
    if (!this.showUncertainEvents() && (event.timing_type === 'uncertain' || event.timing_type === 'ordering' || !this.resolveEventDate(event.event_date))) return false;
    const hasEvidence = Boolean(event.source_evidence?.trim());
    return this.evidenceFilter() === 'all' || (this.evidenceFilter() === 'with_evidence' ? hasEvidence : !hasEvidence);
  }));

  readonly selectedEventHasSourceEvidence = computed(() => {
    const event = this.selectedEvent();
    return Boolean(event?.source_evidence && event.source_evidence.trim().length > 0);
  });

  readonly sourceModelLabel = computed(() => {
    const timeline = this.timeline();
    if (!timeline?.source_model) {
      return this.generationStatusLabel();
    }
    return timeline.source_model;
  });

  async ngOnInit(): Promise<void> {
    this.route.paramMap
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((params) => {
        const sessionId = Number(params.get('sessionId'));
        const timelineIdRaw = params.get('timelineId');
        const timelineId = timelineIdRaw ? Number(timelineIdRaw) : null;
        void this.handleRouteChange(sessionId, timelineId);
      });
  }

  ngAfterViewInit(): void {
    const canvas = this.timelineCanvas?.nativeElement;
    const area = this.timelineArea?.nativeElement;
    if (!canvas || !area || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => {
      this.canvasWidth.set(Math.max(720, Math.round(area.clientWidth)));
      this.scrollMax.set(Math.max(0, area.scrollWidth - area.clientWidth));
    });
    observer.observe(area);
    this.destroyRef.onDestroy(() => observer.disconnect());
  }

  async loadTimeline(sessionId: number, timelineId: number | null): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const payload = timelineId
        ? await fetchInspectionSessionTimelineById(sessionId, timelineId)
        : await this.fetchLatestTimeline(sessionId);
      this.timeline.set(payload);
      this.timelineId.set(payload.timeline_id ?? timelineId ?? null);
      this.selectedEventId.set(null);
      this.selectedCluster.set(null);
    } catch (error) {
      this.timeline.set(null);
      if (this.isNotFoundError(error)) {
        this.error.set(
          timelineId
            ? '[ERROR] This timeline is no longer available. Open another saved timeline from Clinical Sessions.'
            : '[ERROR] No timetable is available yet. Create one from Clinical Sessions.',
        );
      } else {
        this.error.set(error instanceof Error ? error.message : 'Failed to load timetable.');
      }
    } finally {
      this.loading.set(false);
    }
  }

  async regenerate(): Promise<void> {
    const id = this.sessionId();
    if (!id || this.loading()) {
      return;
    }
    this.loading.set(true);
    this.error.set(null);
    try {
      const payload = await generateInspectionSessionTimeline(id, { force_regenerate: true });
      this.timeline.set(payload);
      this.timelineId.set(payload.timeline_id ?? null);
      this.selectedEventId.set(null);
      this.selectedCluster.set(null);
      if (payload.timeline_id) {
        await this.router.navigate(['/sessions', id, 'timetable', payload.timeline_id], { replaceUrl: true });
      }
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Failed to regenerate timetable.');
    } finally {
      this.loading.set(false);
    }
  }

  selectEvent(event: InspectionTimelineEvent): void {
    const active = document.activeElement;
    this.lastFocusedElement = active instanceof HTMLElement ? active : null;
    const lane = this.resolveLane(event);
    if (this.collapsedLanes()[lane]) this.toggleLaneCollapsed(lane);
    this.selectedCluster.set(null);
    this.selectedEventId.set(event.event_id);
    this.selectionAnnouncement.set(`Selected ${event.title}.`);
    queueMicrotask(() => document.querySelector<HTMLElement>(`[data-event-id="${CSS.escape(event.event_id)}"]`)?.scrollIntoView({ block: 'nearest', inline: 'nearest' }));
  }

  closeInspector(): void {
    this.selectedEventId.set(null);
    this.selectedCluster.set(null);
    queueMicrotask(() => this.lastFocusedElement?.focus());
  }

  @HostListener('document:keydown.escape')
  handleEscape(): void {
    if (this.selectedEventId()) this.closeInspector();
  }

  toggleLaneVisibility(lane: TimetableLane): void { this.visibleLanes.update((value) => ({ ...value, [lane]: !value[lane] })); }
  toggleLaneCollapsed(lane: TimetableLane): void { this.collapsedLanes.update((value) => ({ ...value, [lane]: !value[lane] })); }
  setEvidenceFilter(value: string): void {
    if (isTimelineEvidenceFilter(value)) {
      this.evidenceFilter.set(value);
    }
  }
  toggleUncertainEvents(): void { this.showUncertainEvents.update((value) => !value); }
  toggleHideEmptyLanes(): void { this.hideEmptyLanes.update((value) => !value); }
  setDensity(value: string): void {
    if (isTimelineDensity(value)) {
      this.density.set(value);
    }
  }
  setZoom(value: number): void { this.zoom.set(Math.min(2.5, Math.max(0.75, value))); queueMicrotask(() => this.updateScrollMetrics()); }
  zoomIn(): void { this.setZoom(this.zoom() + 0.25); }
  zoomOut(): void { this.setZoom(this.zoom() - 0.25); }
  fitZoom(): void { this.setZoom(1); this.fitRange(); }
  setScrollOffset(value: number): void {
    const next = Math.min(this.scrollMax(), Math.max(0, value));
    this.scrollOffset.set(next);
    this.timelineArea?.nativeElement.scrollTo({ left: next, behavior: 'auto' });
  }
  handleScrollInput(event: Event): void {
    const target = event.target;
    if (target instanceof HTMLInputElement) {
      this.setScrollOffset(target.valueAsNumber);
    }
  }
  handleEventKeydown(event: KeyboardEvent, item: InspectionTimelineEvent): void {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); this.selectEvent(item); return; }
    if (event.key === 'ArrowLeft') { event.preventDefault(); this.selectPreviousEvent(); }
    if (event.key === 'ArrowRight') { event.preventDefault(); this.selectNextEvent(); }
  }
  selectPreviousEvent(): void { const event = this.filteredEvents()[this.selectedEventIndex() - 1]; if (event) this.selectEvent(event); }
  selectNextEvent(): void { const event = this.filteredEvents()[this.selectedEventIndex() + 1]; if (event) this.selectEvent(event); }
  fitRange(): void { this.setScrollOffset(0); }
  isLinkedToSelectedEvent(event: InspectionTimelineEvent): boolean { return this.linkedSelectedEventIds().has(event.event_id); }
  selectCluster(cluster: TimelineCluster): void {
    this.selectedCluster.set(cluster);
    const event = this.orderedEvents().find((item) => item.event_id === cluster.memberIds[0]);
    if (event) this.selectEvent(event);
    this.selectedCluster.set(cluster);
  }

  clusterMembers(): InspectionTimelineEvent[] {
    const ids = new Set(this.selectedCluster()?.memberIds ?? []);
    return this.orderedEvents().filter((event) => ids.has(event.event_id));
  }

  timingLabel(value: InspectionTimelineTimingType): string {
    return TIMING_LABELS[value] ?? value;
  }

  confidenceLabel(value: number | null): string {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return 'Not scored';
    }
    if (value >= 0.9) return 'Very high';
    if (value >= 0.75) return 'High';
    if (value >= 0.55) return 'Moderate';
    if (value >= 0.35) return 'Low';
    return 'Very low';
  }

  generationStatusLabel(): string {
    return this.isFallbackTimeline() ? 'Fallback' : 'LLM generated';
  }

  eventTimingText(event: InspectionTimelineEvent): string {
    return event.extracted_timing_text || event.relative_time || event.event_date || this.timingLabel(event.timing_type);
  }

  private async handleRouteChange(sessionId: number, timelineId: number | null): Promise<void> {
    if (!Number.isFinite(sessionId) || sessionId <= 0) {
      this.error.set('Invalid session id.');
      this.timeline.set(null);
      return;
    }
    if (timelineId !== null && (!Number.isFinite(timelineId) || timelineId <= 0)) {
      this.error.set('Invalid timeline id.');
      this.timeline.set(null);
      return;
    }
    this.sessionId.set(sessionId);
    this.timelineId.set(timelineId);
    await this.loadTimeline(sessionId, timelineId);
  }

  private async fetchLatestTimeline(sessionId: number): Promise<InspectionSessionTimeline> {
    const timelines = await fetchInspectionSessionTimelineList(sessionId);
    const latest = timelines.items[0];
    if (!latest?.timeline_id) {
      throw new Error('Not found');
    }
    return fetchInspectionSessionTimelineById(sessionId, latest.timeline_id);
  }

  private resolveEventDate(value: string | null): ResolvedTimelineDate | null {
    const normalized = normalizeTimelineDate(value);
    return normalized ? { date: dayToUtcDate(normalized.startDay), precision: normalized.precision } : null;
  }

  private resolveNormalizedRange(event: InspectionTimelineEvent): NormalizedTimelineDate | null {
    const start = normalizeTimelineDate(event.event_date);
    if (!start) return null;
    const explicitEnd = normalizeTimelineDate(event.event_date_end ?? null);
    if (!explicitEnd || explicitEnd.startDay < start.startDay) return start;
    return { ...start, endDay: explicitEnd.endDay };
  }

  precisionLabel(value: string | null): string {
    const precision = this.resolveEventDate(value)?.precision;
    return precision ? `${precision[0].toUpperCase()}${precision.slice(1)} precision` : 'Not reported';
  }

  private monthLabel(value: Date): string {
    return TIMELINE_MONTH_FORMATTER.format(value);
  }

  private monthsBetweenCount(start: Date, end: Date): number {
    return (end.getUTCFullYear() - start.getUTCFullYear()) * 12 + (end.getUTCMonth() - start.getUTCMonth());
  }

  private resolveEventPosition(
    event: InspectionTimelineEvent,
    range: TimelineDateRange,
  ): number | null {
    const normalized = this.resolveNormalizedRange(event);
    const scale = this.timelineScale();
    if (!normalized || !scale || !range.start || !range.end) return null;
    return scale.toPercent((normalized.startDay + normalized.endDay) / 2);
  }

  private resolveEventAlign(positionPercent: number | null): TimelineCardAlign {
    if (positionPercent === null) return 'start';
    if (positionPercent <= 12) {
      return 'start';
    }
    if (positionPercent >= 88) {
      return 'end';
    }
    return 'center';
  }

  private assignStackLevels(items: RenderedTimelineEvent[]): RenderedTimelineEvent[] {
    const nextItems = items.map((item) => ({ ...item }));
    for (const lane of TIMETABLE_LANES) {
      const laneItems = nextItems.filter((item) => item.lane === lane && item.positionPercent !== null);
      const layout = packTimelineItems(laneItems.map((item) => ({
        id: item.event.event_id,
        positionPercent: item.positionPercent ?? 0,
        width: this.density() === 'dense' ? 104 : this.density() === 'compact' ? 132 : 176,
        sortOrder: item.event.sort_order,
      })), 1200);
      for (const placement of layout.placements) {
        const item = laneItems.find((candidate) => candidate.event.event_id === placement.id);
        if (item) item.stackLevel = placement.row;
      }
    }
    return nextItems;
  }

  private resolveLane(event: InspectionTimelineEvent): TimetableLane {
    if (!normalizeTimelineDate(event.event_date)) return 'unanchored';
    if (event.timing_type === 'uncertain' || event.timing_type === 'ordering') {
      return 'uncertainty';
    }
    if (event.event_type === 'therapy') {
      return 'therapy';
    }
    if (event.event_type === 'lab') {
      return 'labs';
    }
    return 'clinical';
  }

  private updateScrollMetrics(): void {
    const area = this.timelineArea?.nativeElement;
    if (!area) return;
    this.scrollMax.set(Math.max(0, area.scrollWidth - area.clientWidth));
    this.scrollOffset.set(Math.min(this.scrollOffset(), this.scrollMax()));
  }

  private isNotFoundError(error: unknown): boolean {
    return error instanceof Error && error.message.toLowerCase().includes('not found');
  }
}
