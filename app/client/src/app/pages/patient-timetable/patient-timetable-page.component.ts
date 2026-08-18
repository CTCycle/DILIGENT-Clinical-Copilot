import { CommonModule } from '@angular/common';
import { Component, DestroyRef, HostListener, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';

import { HelpPopoverComponent } from '../../core/guidance/help-popover.component';
import {
  fetchInspectionSessionTimelineById,
  fetchInspectionSessionTimelineJobStatus,
  fetchInspectionSessionTimelineList,
  startInspectionSessionTimelineJob,
} from '../../core/services/inspection-api';
import { JobPollingService } from '../../core/services/job-polling.service';
import {
  InspectionSessionTimeline,
  InspectionTimelineJobStatusResponse,
  InspectionTimelineEvent,
  InspectionTimelineEventType,
  InspectionTimelineTimingType,
} from '../../core/models/types';
import {
  dayToUtcDate,
  normalizeTimelineDate,
  NormalizedTimelineDate,
  TimelineDatePrecision,
} from './timeline-date';
import {
  TimetableFilterOption,
  TimetableFilterSelectComponent,
} from './components/timetable-filter-select.component';

type TimetableLane = 'clinical' | 'therapy' | 'labs' | 'uncertainty' | 'unanchored';
type TimelineEvidenceFilter = 'all' | 'with_evidence' | 'missing_evidence';
type TimelineDensity = 'compact' | 'comfortable' | 'dense';

type TimelineEventGroup = {
  key: string;
  eyebrow: string;
  label: string;
  dateHint: string;
  isUnanchored: boolean;
  events: InspectionTimelineEvent[];
};

type TimelineLaneSummary = {
  id: TimetableLane;
  label: string;
  count: number;
  collapsed: boolean;
};

const EVENT_COLORS: Record<InspectionTimelineEventType, string> = {
  therapy: '#0f8f83',
  disease: '#dc2626',
  lab: '#1266c3',
  other: '#6b7280',
};

const EVENT_TYPE_LABELS: Record<InspectionTimelineEventType, string> = {
  therapy: 'Medication',
  disease: 'Clinical',
  lab: 'Laboratory',
  other: 'Other',
};

const TIMING_LABELS: Record<InspectionTimelineTimingType, string> = {
  explicit_date: 'Explicit date',
  relative: 'Relative timing',
  duration: 'Duration',
  recurring: 'Recurring',
  uncertain: 'Uncertain timing',
  ordering: 'Relative order',
};

const TIMETABLE_LANE_LABELS: Record<TimetableLane, string> = {
  clinical: 'Clinical',
  therapy: 'Medications',
  labs: 'Laboratory',
  uncertainty: 'Uncertain',
  unanchored: 'Date not reported',
};

const TIMETABLE_LANES: TimetableLane[] = ['clinical', 'therapy', 'labs', 'uncertainty', 'unanchored'];
const TIMELINE_DAY_FORMATTER = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  timeZone: 'UTC',
});
const TIMELINE_MONTH_FORMATTER = new Intl.DateTimeFormat('en-US', {
  month: 'long',
  year: 'numeric',
  timeZone: 'UTC',
});
const TIMELINE_YEAR_FORMATTER = new Intl.DateTimeFormat('en-US', {
  year: 'numeric',
  timeZone: 'UTC',
});
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
  imports: [CommonModule, RouterLink, TimetableFilterSelectComponent, HelpPopoverComponent],
  templateUrl: './patient-timetable-page.component.html',
  styleUrl: './patient-timetable-page.component.scss',
})
export class PatientTimetablePageComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);
  private readonly jobPolling = inject(JobPollingService);

  readonly sessionId = signal<number | null>(null);
  readonly timelineId = signal<number | null>(null);
  readonly timeline = signal<InspectionSessionTimeline | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly selectedEventId = signal<string | null>(null);
  readonly collapsedLanes = signal<Record<TimetableLane, boolean>>({
    clinical: false,
    therapy: false,
    labs: false,
    uncertainty: false,
    unanchored: false,
  });
  readonly evidenceFilter = signal<TimelineEvidenceFilter>('all');
  readonly showUncertainEvents = signal(true);
  readonly hideEmptyLanes = signal(false);
  readonly density = signal<TimelineDensity>('comfortable');
  readonly selectionAnnouncement = signal('');
  readonly evidenceFilterOptions = EVIDENCE_FILTER_OPTIONS;
  readonly densityOptions = DENSITY_OPTIONS;
  readonly isNarrowScreen = signal(globalThis.matchMedia?.('(max-width: 720px)').matches ?? false);
  private lastFocusedElement: HTMLElement | null = null;

  readonly orderedEvents = computed(() => [...(this.timeline()?.events ?? [])].sort((a, b) => {
    const left = normalizeTimelineDate(a.event_date);
    const right = normalizeTimelineDate(b.event_date);
    if (left && right) {
      return left.startDay - right.startDay
        || left.endDay - right.endDay
        || a.sort_order - b.sort_order
        || a.event_id.localeCompare(b.event_id);
    }
    if (left) return -1;
    if (right) return 1;
    return a.sort_order - b.sort_order || a.event_id.localeCompare(b.event_id);
  }));

  readonly dateRange = computed<{ start: Date | null; end: Date | null }>(() => {
    const dates = this.orderedEvents()
      .map((event) => this.resolveNormalizedRange(event))
      .filter((value): value is NormalizedTimelineDate => value !== null);
    return {
      start: dates.length ? dayToUtcDate(Math.min(...dates.map((date) => date.startDay))) : null,
      end: dates.length ? dayToUtcDate(Math.max(...dates.map((date) => date.endDay))) : null,
    };
  });

  readonly filteredEvents = computed(() => this.orderedEvents().filter((event) => {
    if (!this.showUncertainEvents() && this.isEventUncertain(event)) return false;
    const hasEvidence = this.hasEventEvidence(event);
    return this.evidenceFilter() === 'all'
      || (this.evidenceFilter() === 'with_evidence' ? hasEvidence : !hasEvidence);
  }));

  readonly laneSummaries = computed<TimelineLaneSummary[]>(() => TIMETABLE_LANES
    .map((id) => ({
      id,
      label: TIMETABLE_LANE_LABELS[id],
      count: this.filteredEvents().filter((event) => this.resolveLane(event) === id).length,
      collapsed: this.collapsedLanes()[id],
    }))
    .filter((lane) => !this.hideEmptyLanes() || lane.count > 0));

  readonly timelineGroups = computed<TimelineEventGroup[]>(() => {
    const groups = new Map<string, TimelineEventGroup>();
    for (const event of this.filteredEvents()) {
      const lane = this.resolveLane(event);
      if (this.collapsedLanes()[lane]) continue;

      const normalized = normalizeTimelineDate(event.event_date);
      const key = normalized ? `date:${normalized.startDay}` : 'unanchored';
      const existing = groups.get(key);
      if (existing) {
        existing.events.push(event);
        continue;
      }

      groups.set(key, normalized
        ? {
          key,
          eyebrow: 'Chronology',
          label: this.eventDateLabel(event),
          dateHint: this.dateHint(event),
          isUnanchored: false,
          events: [event],
        }
        : {
          key,
          eyebrow: 'Review required',
          label: 'Date not reported',
          dateHint: 'Events are kept in source order until a canonical date is available.',
          isUnanchored: true,
          events: [event],
        });
    }
    return [...groups.values()];
  });

  readonly visibleEventsCount = computed(() => this.timelineGroups().reduce((total, group) => total + group.events.length, 0));
  readonly selectedEvent = computed(() => {
    const selectedId = this.selectedEventId();
    return this.orderedEvents().find((event) => event.event_id === selectedId) ?? null;
  });
  readonly rangeLabel = computed(() => {
    const range = this.dateRange();
    if (!range.start || !range.end) return 'Timeline events';
    if (this.monthLabel(range.start) === this.monthLabel(range.end)) return this.monthLabel(range.start);
    return `${this.monthLabel(range.start)} – ${this.monthLabel(range.end)}`;
  });
  readonly isFallbackTimeline = computed(() => this.timeline()?.generation_status === 'fallback');
  readonly generationNote = computed(() => {
    const timeline = this.timeline();
    if (!timeline) return null;
    if (timeline.generation_status === 'fallback') {
      return timeline.generation_note
        || 'This timetable uses deterministic fallback events because model extraction was unavailable.';
    }
    const sourceKind = timeline.source_kind === 'cloud' ? 'cloud' : 'local';
    return timeline.generation_note || `Generated with the configured ${sourceKind} timeline extraction model.`;
  });
  readonly generationErrorLabel = computed(() => {
    const code = this.timeline()?.generation_error_code;
    if (!code) return null;
    const labels: Record<string, string> = {
      network_unavailable: 'Provider network unavailable',
      timeout: 'Provider timeout',
      authentication: 'Provider authentication rejected',
      rate_limited: 'Provider rate limit reached',
      upstream_error: 'Provider upstream error',
      invalid_response: 'Invalid structured provider response',
      configuration: 'Provider configuration incomplete',
      provider_error: 'Provider request failed',
      unknown: 'Unknown generation failure',
    };
    return labels[code] || 'Generation failure';
  });
  readonly selectedEventIndex = computed(() => this.filteredEvents().findIndex((event) => event.event_id === this.selectedEventId()));
  readonly hasPreviousEvent = computed(() => this.selectedEventIndex() > 0);
  readonly hasNextEvent = computed(() => this.selectedEventIndex() >= 0 && this.selectedEventIndex() < this.filteredEvents().length - 1);
  readonly linkedSelectedEventIds = computed(() => new Set(this.selectedEvent()?.linked_patient_event_ids ?? []));
  readonly selectedEventHasSourceEvidence = computed(() => this.hasEventEvidence(this.selectedEvent()));

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

  @HostListener('window:resize')
  handleResize(): void {
    this.isNarrowScreen.set(globalThis.matchMedia?.('(max-width: 720px)').matches ?? false);
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
    } catch (error) {
      this.timeline.set(null);
      if (this.isNotFoundError(error)) {
        this.error.set(
          timelineId
            ? 'This timeline is no longer available. Open another saved timeline from Clinical Sessions.'
            : 'No timetable is available yet. Create one from Clinical Sessions.',
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
    if (!id || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    try {
      const started = await startInspectionSessionTimelineJob(id, { force_regenerate: true });
      const job = await this.waitForTimelineJob(id, started.job_id, started.poll_interval);
      if (!job) return;
      if (job.status !== 'completed') {
        throw new Error(job.error || 'Timeline generation did not complete.');
      }
      const generatedTimelineId = job.result?.timeline_id;
      const payload = typeof generatedTimelineId === 'number'
        ? await fetchInspectionSessionTimelineById(id, generatedTimelineId)
        : await this.fetchLatestTimeline(id);
      this.timeline.set(payload);
      this.timelineId.set(payload.timeline_id ?? null);
      this.selectedEventId.set(null);
      if (payload.timeline_id) {
        await this.router.navigate(['/sessions', id, 'timetable', payload.timeline_id], { replaceUrl: true });
      }
    } catch (error) {
      if (!this.destroyRef.destroyed) {
        this.error.set(error instanceof Error ? error.message : 'Failed to regenerate timetable.');
      }
    } finally {
      if (!this.destroyRef.destroyed) this.loading.set(false);
    }
  }

  selectEvent(event: InspectionTimelineEvent): void {
    const active = document.activeElement;
    this.lastFocusedElement = active instanceof HTMLElement ? active : null;
    this.selectedEventId.set(event.event_id);
    this.selectionAnnouncement.set(`Selected ${event.title}.`);
    queueMicrotask(() => document.querySelector<HTMLElement>(`[data-event-id="${CSS.escape(event.event_id)}"]`)?.scrollIntoView({ block: 'nearest', inline: 'nearest' }));
  }

  closeInspector(): void {
    this.selectedEventId.set(null);
    queueMicrotask(() => this.lastFocusedElement?.focus());
  }

  @HostListener('document:keydown.escape')
  handleEscape(): void {
    if (this.selectedEventId()) this.closeInspector();
  }

  toggleLaneCollapsed(lane: TimetableLane): void {
    this.collapsedLanes.update((value) => ({ ...value, [lane]: !value[lane] }));
  }

  setEvidenceFilter(value: string): void {
    if (isTimelineEvidenceFilter(value)) this.evidenceFilter.set(value);
  }

  toggleUncertainEvents(): void {
    this.showUncertainEvents.update((value) => !value);
  }

  toggleHideEmptyLanes(): void {
    this.hideEmptyLanes.update((value) => !value);
  }

  setDensity(value: string): void {
    if (isTimelineDensity(value)) this.density.set(value);
  }

  handleEventKeydown(event: KeyboardEvent, item: InspectionTimelineEvent): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.selectEvent(item);
      return;
    }
    if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
      event.preventDefault();
      this.selectPreviousEvent();
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowRight') {
      event.preventDefault();
      this.selectNextEvent();
    }
  }

  selectPreviousEvent(): void {
    const event = this.filteredEvents()[this.selectedEventIndex() - 1];
    if (event) this.selectEvent(event);
  }

  selectNextEvent(): void {
    const event = this.filteredEvents()[this.selectedEventIndex() + 1];
    if (event) this.selectEvent(event);
  }

  eventTypeLabel(event: InspectionTimelineEvent): string {
    return EVENT_TYPE_LABELS[event.event_type] ?? EVENT_TYPE_LABELS.other;
  }

  eventLaneLabel(event: InspectionTimelineEvent): string {
    return TIMETABLE_LANE_LABELS[this.resolveLane(event)];
  }

  eventColor(event: InspectionTimelineEvent): string {
    return EVENT_COLORS[event.event_type] ?? EVENT_COLORS.other;
  }

  timingLabel(value: InspectionTimelineTimingType): string {
    return TIMING_LABELS[value] ?? value;
  }

  confidenceLabel(value: number | null): string {
    if (typeof value !== 'number' || !Number.isFinite(value)) return 'Not scored';
    if (value >= 0.9) return 'Very high';
    if (value >= 0.75) return 'High';
    if (value >= 0.55) return 'Moderate';
    if (value >= 0.35) return 'Low';
    return 'Very low';
  }

  generationStatusLabel(): string {
    return this.isFallbackTimeline() ? 'Fallback chronology' : 'LLM generated';
  }

  eventTimingText(event: InspectionTimelineEvent): string {
    return event.extracted_timing_text || event.relative_time || event.event_date || this.timingLabel(event.timing_type);
  }

  eventDateLabel(event: InspectionTimelineEvent): string {
    const resolved = this.resolveEventDate(event.event_date);
    if (!resolved) return 'Date not reported';
    if (resolved.precision === 'day') return TIMELINE_DAY_FORMATTER.format(resolved.date);
    if (resolved.precision === 'month') return TIMELINE_MONTH_FORMATTER.format(resolved.date);
    return TIMELINE_YEAR_FORMATTER.format(resolved.date);
  }

  eventDateSummary(event: InspectionTimelineEvent): string {
    const resolved = this.resolveEventDate(event.event_date);
    if (!resolved) {
      const extracted = event.extracted_timing_text || event.relative_time;
      return extracted ? `No canonical date · ${extracted}` : 'No canonical date reported';
    }

    let summary = this.eventDateLabel(event);
    const explicitEnd = normalizeTimelineDate(event.event_date_end ?? null);
    if (explicitEnd && explicitEnd.startDay > resolved.date.getTime() / 86_400_000) {
      summary += ` – ${this.formatEndDate(explicitEnd, resolved.precision)}`;
    }
    if (resolved.precision !== 'day') summary += ` · ${resolved.precision}-level date`;
    const extracted = event.extracted_timing_text || event.relative_time;
    return extracted && extracted !== event.event_date ? `${summary} · ${extracted}` : summary;
  }

  precisionLabel(value: string | null): string {
    const precision = this.resolveEventDate(value)?.precision;
    return precision ? `${precision[0].toUpperCase()}${precision.slice(1)} precision` : 'Not reported';
  }

  isEventUncertain(event: InspectionTimelineEvent): boolean {
    return event.date_certainty === 'uncertain'
      || event.timing_type === 'uncertain'
      || event.timing_type === 'ordering'
      || !normalizeTimelineDate(event.event_date);
  }

  hasEventEvidence(event: InspectionTimelineEvent | null): boolean {
    return Boolean(event?.source_evidence?.trim());
  }

  isLinkedToSelectedEvent(event: InspectionTimelineEvent): boolean {
    return this.linkedSelectedEventIds().has(event.event_id);
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
    if (!latest?.timeline_id) throw new Error('Not found');
    return fetchInspectionSessionTimelineById(sessionId, latest.timeline_id);
  }

  private async waitForTimelineJob(
    sessionId: number,
    jobId: string,
    pollIntervalSeconds: number,
  ): Promise<InspectionTimelineJobStatusResponse | null> {
    const deadline = Date.now() + 360_000;
    const delayMs = Math.max(
      250,
      Math.round(
        (Number.isFinite(pollIntervalSeconds) && pollIntervalSeconds > 0
          ? pollIntervalSeconds
          : 1) * 1000,
      ),
    );
    let terminalStatus: InspectionTimelineJobStatusResponse | null = null;
    await this.jobPolling.run({
      intervalMs: delayMs,
      isCancelled: () => this.destroyRef.destroyed || Date.now() >= deadline,
      pollStep: async () => {
        const job = await fetchInspectionSessionTimelineJobStatus(sessionId, jobId);
        if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
          terminalStatus = job;
          return false;
        }
        return true;
      },
    });
    if (terminalStatus) return terminalStatus;
    if (this.destroyRef.destroyed) return null;
    throw new Error('Timeline generation timed out. Check the saved timeline history and retry if needed.');
  }

  private resolveEventDate(value: string | null): { date: Date; precision: TimelineDatePrecision } | null {
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

  private resolveLane(event: InspectionTimelineEvent): TimetableLane {
    if (!normalizeTimelineDate(event.event_date)) return 'unanchored';
    if (event.timing_type === 'uncertain' || event.timing_type === 'ordering') return 'uncertainty';
    if (event.event_type === 'therapy') return 'therapy';
    if (event.event_type === 'lab') return 'labs';
    return 'clinical';
  }

  private dateHint(event: InspectionTimelineEvent): string {
    if (this.isEventUncertain(event)) return 'Date present, but timing requires review.';
    const precision = this.resolveEventDate(event.event_date)?.precision;
    return precision === 'day' ? 'Day-level placement' : `${precision ?? 'Unknown'}-level placement`;
  }

  private formatEndDate(value: NormalizedTimelineDate, precision: TimelineDatePrecision): string {
    const date = dayToUtcDate(value.endDay);
    if (precision === 'year') return TIMELINE_YEAR_FORMATTER.format(date);
    if (precision === 'month') return TIMELINE_MONTH_FORMATTER.format(dayToUtcDate(value.startDay));
    return TIMELINE_DAY_FORMATTER.format(date);
  }

  private monthLabel(value: Date): string {
    return TIMELINE_MONTH_FORMATTER.format(value);
  }

  private isNotFoundError(error: unknown): boolean {
    return error instanceof Error && error.message.toLowerCase().includes('not found');
  }
}
