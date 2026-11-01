// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { CommonModule } from '@angular/common';
import { Component, DestroyRef, HostListener, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import {
  LucideArrowLeft,
  LucideChartGantt,
  LucideChevronDown,
  LucideChevronsDownUp,
  LucideChevronsUpDown,
  LucideCircleHelp,
  LucideFilterX,
  LucideList,
  LucideX,
} from '@lucide/angular';

import { HelpPopoverComponent } from '../../core/guidance/help-popover.component';
import {
  fetchInspectionSessionTimelineById,
  fetchInspectionSessionTimelineJobStatus,
  fetchInspectionSessionTimelineList,
  startInspectionSessionTimelineJob,
} from '../../core/services/session-timeline-api';
import { JobPollingService } from '../../core/services/job-polling.service';
import {
  InspectionSessionTimeline,
  InspectionTimelineJobStatusResponse,
  InspectionTimelineEvent,
} from '../../core/models/inspection-types';
import {
  dayToUtcDate,
  formatDayGap,
  normalizeTimelineDate,
  NormalizedTimelineDate,
} from './timeline-date';
import {
  eventCategory,
  eventColor,
  eventDateLabel,
  formatMonth,
  formatTimelineDate,
  hasEventEvidence,
  isDateInferred,
  isEventUncertain,
  TIMELINE_CATEGORIES,
  TIMELINE_CATEGORY_LABELS,
  TimelineCategory,
  TimelineEventGroup,
  TimelineViewMode,
} from './timeline-presentation';
import { TimelineEventDetailsComponent } from './components/timeline-event-details.component';
import { TimelineHorizontalViewComponent } from './components/timeline-horizontal-view.component';

type TimelineCategorySummary = {
  id: TimelineCategory;
  label: string;
  count: number;
  hidden: boolean;
};

const UNANCHORED_GROUP_KEY = 'unanchored';

const PRECISION_HINTS: Record<NormalizedTimelineDate['precision'], string> = {
  day: 'Day-level date',
  month: 'Month only · day not stated',
  year: 'Year only · day and month not stated',
};

@Component({
  selector: 'app-patient-timetable-page',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    HelpPopoverComponent,
    TimelineEventDetailsComponent,
    TimelineHorizontalViewComponent,
    LucideArrowLeft,
    LucideChartGantt,
    LucideChevronDown,
    LucideChevronsDownUp,
    LucideChevronsUpDown,
    LucideCircleHelp,
    LucideFilterX,
    LucideList,
    LucideX,
  ],
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
  readonly viewMode = signal<TimelineViewMode>('list');
  readonly selectedEventId = signal<string | null>(null);
  readonly expandedEventIds = signal<ReadonlySet<string>>(new Set());
  readonly collapsedGroupKeys = signal<ReadonlySet<string>>(new Set([UNANCHORED_GROUP_KEY]));
  readonly hiddenCategories = signal<ReadonlySet<TimelineCategory>>(new Set());
  readonly showUncertainEvents = signal(true);
  readonly selectionAnnouncement = signal('');
  private routeLoadGeneration = 0;

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

  readonly eventsById = computed(() => new Map(this.orderedEvents().map((event) => [event.event_id, event])));

  readonly dateRange = computed<{ start: Date | null; end: Date | null }>(() => {
    const dates = this.orderedEvents()
      .map((event) => this.resolveNormalizedRange(event))
      .filter((value): value is NormalizedTimelineDate => value !== null);
    return {
      start: dates.length ? dayToUtcDate(Math.min(...dates.map((date) => date.startDay))) : null,
      end: dates.length ? dayToUtcDate(Math.max(...dates.map((date) => date.endDay))) : null,
    };
  });

  /** Events passing the timing filter, before category visibility is applied. */
  private readonly timingFilteredEvents = computed(() => this.orderedEvents()
    .filter((event) => this.showUncertainEvents() || !isEventUncertain(event)));

  readonly filteredEvents = computed(() => this.timingFilteredEvents()
    .filter((event) => !this.hiddenCategories().has(eventCategory(event))));

  readonly categorySummaries = computed<TimelineCategorySummary[]>(() => TIMELINE_CATEGORIES
    .map((id) => ({
      id,
      label: TIMELINE_CATEGORY_LABELS[id],
      count: this.timingFilteredEvents().filter((event) => eventCategory(event) === id).length,
      hidden: this.hiddenCategories().has(id),
    }))
    .filter((category) => category.count > 0 || category.hidden));

  readonly visibleCategories = computed(() => this.categorySummaries()
    .filter((category) => !category.hidden && category.count > 0)
    .map((category) => category.id));

  readonly timelineGroups = computed<TimelineEventGroup[]>(() => {
    const groups = new Map<string, TimelineEventGroup>();
    let previousDayStart: number | null = null;
    for (const event of this.filteredEvents()) {
      const normalized = normalizeTimelineDate(event.event_date);
      const key = normalized ? `${normalized.precision}:${normalized.startDay}` : UNANCHORED_GROUP_KEY;
      const existing = groups.get(key);
      if (existing) {
        existing.events.push(event);
        continue;
      }
      if (!normalized) {
        groups.set(key, {
          key,
          label: 'Date not reported',
          dateHint: 'Kept in source order until a date is documented',
          isUnanchored: true,
          startDay: null,
          gapLabel: null,
          events: [event],
        });
        continue;
      }
      // Gaps are only meaningful between day-level dates.
      const gapLabel = previousDayStart !== null && normalized.precision === 'day'
        ? formatDayGap(normalized.startDay - previousDayStart)
        : null;
      groups.set(key, {
        key,
        label: formatTimelineDate(normalized),
        dateHint: PRECISION_HINTS[normalized.precision],
        isUnanchored: false,
        startDay: normalized.startDay,
        gapLabel,
        events: [event],
      });
      if (normalized.precision === 'day') previousDayStart = normalized.startDay;
    }
    // Dated groups first in chronological order, the undated review group last.
    return [...groups.values()].sort((left, right) => Number(left.isUnanchored) - Number(right.isUnanchored));
  });

  readonly datedGroups = computed(() => this.timelineGroups().filter((group) => !group.isUnanchored));
  readonly undatedGroup = computed(() => this.timelineGroups().find((group) => group.isUnanchored) ?? null);
  readonly displayedEvents = computed(() => this.timelineGroups().flatMap((group) => group.events));
  readonly visibleEventsCount = computed(() => this.displayedEvents().length);
  readonly reviewEventsCount = computed(() => this.orderedEvents()
    .filter((event) => isEventUncertain(event) || isDateInferred(event)).length);
  readonly allExpanded = computed(() => {
    const expanded = this.expandedEventIds();
    const events = this.displayedEvents();
    return events.length > 0
      && events.every((event) => expanded.has(event.event_id))
      && this.timelineGroups().every((group) => !this.collapsedGroupKeys().has(group.key));
  });

  readonly selectedEvent = computed(() => this.eventsById().get(this.selectedEventId() ?? '') ?? null);
  readonly rangeLabel = computed(() => {
    const range = this.dateRange();
    if (!range.start || !range.end) return 'Timeline events';
    if (formatMonth(range.start) === formatMonth(range.end)) return formatMonth(range.start);
    return `${formatMonth(range.start)} – ${formatMonth(range.end)}`;
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
  readonly selectedEventIndex = computed(() => this.displayedEvents().findIndex((event) => event.event_id === this.selectedEventId()));
  readonly hasPreviousEvent = computed(() => this.selectedEventIndex() > 0);
  readonly hasNextEvent = computed(() => this.selectedEventIndex() < this.displayedEvents().length - 1);

  readonly eventColor = eventColor;
  readonly eventDateLabel = eventDateLabel;
  readonly hasEventEvidence = hasEventEvidence;
  readonly isDateInferred = isDateInferred;
  readonly isEventUncertain = isEventUncertain;
  readonly categoryLabel = (event: InspectionTimelineEvent): string => TIMELINE_CATEGORY_LABELS[eventCategory(event)];

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

  async loadTimeline(
    sessionId: number,
    timelineId: number | null,
    loadGeneration = ++this.routeLoadGeneration,
  ): Promise<void> {
    const isCurrentLoad = (): boolean => (
      loadGeneration === this.routeLoadGeneration
      && this.sessionId() === sessionId
    );
    this.loading.set(true);
    this.error.set(null);
    try {
      const payload = timelineId
        ? await fetchInspectionSessionTimelineById(sessionId, timelineId)
        : await this.fetchLatestTimeline(sessionId);
      if (!isCurrentLoad()) return;
      this.applyTimeline(payload);
      this.timelineId.set(payload.timeline_id ?? timelineId ?? null);
    } catch (error) {
      if (!isCurrentLoad()) return;
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
      if (isCurrentLoad()) this.loading.set(false);
    }
  }

  async regenerate(): Promise<void> {
    const id = this.sessionId();
    const loadGeneration = this.routeLoadGeneration;
    const isCurrentGeneration = (): boolean => (
      loadGeneration === this.routeLoadGeneration
      && this.sessionId() === id
    );
    if (!id || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    try {
      const started = await startInspectionSessionTimelineJob(id, { force_regenerate: true });
      if (!isCurrentGeneration()) return;
      const job = await this.waitForTimelineJob(id, started.job_id, started.poll_interval, loadGeneration);
      if (!job) return;
      if (!isCurrentGeneration()) return;
      if (job.status !== 'completed') {
        throw new Error(job.error || 'Timeline generation did not complete.');
      }
      const generatedTimelineId = job.result?.timeline_id;
      const payload = typeof generatedTimelineId === 'number'
        ? await fetchInspectionSessionTimelineById(id, generatedTimelineId)
        : await this.fetchLatestTimeline(id);
      if (!isCurrentGeneration()) return;
      this.applyTimeline(payload);
      this.timelineId.set(payload.timeline_id ?? null);
      if (payload.timeline_id) {
        if (!isCurrentGeneration()) return;
        await this.router.navigate(['/sessions', id, 'timetable', payload.timeline_id], { replaceUrl: true });
      }
    } catch (error) {
      if (!this.destroyRef.destroyed && isCurrentGeneration()) {
        this.error.set(error instanceof Error ? error.message : 'Failed to regenerate timetable.');
      }
    } finally {
      if (!this.destroyRef.destroyed && isCurrentGeneration()) this.loading.set(false);
    }
  }

  setViewMode(mode: TimelineViewMode): void {
    this.viewMode.set(mode);
  }

  toggleCategory(category: TimelineCategory): void {
    this.hiddenCategories.update((current) => this.toggled(current, category));
  }

  toggleUncertainEvents(): void {
    this.showUncertainEvents.update((value) => !value);
  }

  isExpanded(event: InspectionTimelineEvent): boolean {
    return this.expandedEventIds().has(event.event_id);
  }

  toggleEvent(event: InspectionTimelineEvent): void {
    this.expandedEventIds.update((current) => this.toggled(current, event.event_id));
  }

  isGroupCollapsed(group: TimelineEventGroup): boolean {
    return this.collapsedGroupKeys().has(group.key);
  }

  toggleGroup(group: TimelineEventGroup): void {
    this.collapsedGroupKeys.update((current) => this.toggled(current, group.key));
  }

  expandAll(): void {
    this.collapsedGroupKeys.set(new Set());
    this.expandedEventIds.set(new Set(this.displayedEvents().map((event) => event.event_id)));
  }

  collapseAll(): void {
    this.expandedEventIds.set(new Set());
  }

  linkedEvents(event: InspectionTimelineEvent): InspectionTimelineEvent[] {
    const byId = this.eventsById();
    return event.linked_patient_event_ids
      .map((id) => byId.get(id))
      .filter((linked): linked is InspectionTimelineEvent => Boolean(linked));
  }

  isLinkedToSelectedEvent(event: InspectionTimelineEvent): boolean {
    return this.selectedEvent()?.linked_patient_event_ids.includes(event.event_id) ?? false;
  }

  /** Select an event, reveal it in the active view, and expand it in the list. */
  selectEvent(event: InspectionTimelineEvent): void {
    this.selectedEventId.set(event.event_id);
    this.selectionAnnouncement.set(`Selected ${event.title}.`);
    this.expandedEventIds.update((current) => new Set(current).add(event.event_id));
    const group = this.timelineGroups().find((candidate) => candidate.events.includes(event));
    if (group) this.collapsedGroupKeys.update((current) => {
      const next = new Set(current);
      next.delete(group.key);
      return next;
    });
    if (this.viewMode() === 'list') {
      queueMicrotask(() => [...document.querySelectorAll<HTMLElement>('[data-event-row]')]
        .find((row) => row.dataset['eventRow'] === event.event_id)
        ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
    }
  }

  clearSelection(): void {
    this.selectedEventId.set(null);
  }

  @HostListener('document:keydown.escape')
  handleEscape(): void {
    if (this.selectedEventId()) this.clearSelection();
  }

  selectPreviousEvent(): void {
    const event = this.displayedEvents()[this.selectedEventIndex() - 1];
    if (event) this.selectEvent(event);
  }

  selectNextEvent(): void {
    const event = this.displayedEvents()[this.selectedEventIndex() + 1];
    if (event) this.selectEvent(event);
  }

  private applyTimeline(payload: InspectionSessionTimeline): void {
    this.timeline.set(payload);
    this.selectedEventId.set(null);
    this.expandedEventIds.set(new Set());
    this.collapsedGroupKeys.set(new Set([UNANCHORED_GROUP_KEY]));
  }

  private toggled<T>(current: ReadonlySet<T>, value: T): ReadonlySet<T> {
    const next = new Set(current);
    if (!next.delete(value)) next.add(value);
    return next;
  }

  private async handleRouteChange(sessionId: number, timelineId: number | null): Promise<void> {
    const loadGeneration = ++this.routeLoadGeneration;
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
    await this.loadTimeline(sessionId, timelineId, loadGeneration);
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
    loadGeneration: number,
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
      isCancelled: () => (
        this.destroyRef.destroyed
        || Date.now() >= deadline
        || loadGeneration !== this.routeLoadGeneration
        || this.sessionId() !== sessionId
      ),
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
    if (
      this.destroyRef.destroyed
      || loadGeneration !== this.routeLoadGeneration
      || this.sessionId() !== sessionId
    ) return null;
    throw new Error('Timeline generation timed out. Check the saved timeline history and retry if needed.');
  }

  private resolveNormalizedRange(event: InspectionTimelineEvent): NormalizedTimelineDate | null {
    const start = normalizeTimelineDate(event.event_date);
    if (!start) return null;
    const explicitEnd = normalizeTimelineDate(event.event_date_end ?? null);
    if (!explicitEnd || explicitEnd.startDay < start.startDay) return start;
    return { ...start, endDay: explicitEnd.endDay };
  }

  private isNotFoundError(error: unknown): boolean {
    return error instanceof Error && error.message.toLowerCase().includes('not found');
  }
}
