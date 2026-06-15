import { CommonModule } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';

import {
  fetchInspectionSessionTimeline,
  fetchInspectionSessionTimelineById,
  generateInspectionSessionTimeline,
} from '../../core/services/inspection-api';
import {
  InspectionSessionTimeline,
  InspectionTimelineEvent,
  InspectionTimelineEventType,
  InspectionTimelineTimingType,
} from '../../core/models/types';

type TimetableLane = 'clinical' | 'therapy' | 'labs' | 'uncertainty';
type TimelineCardAlign = 'start' | 'center' | 'end';

type RenderedTimelineEvent = {
  event: InspectionTimelineEvent;
  lane: TimetableLane;
  color: string;
  positionPercent: number;
  align: TimelineCardAlign;
  stackLevel: number;
};

type TimelineDateRange = {
  start: Date | null;
  end: Date | null;
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
};

const TIMETABLE_LANES: TimetableLane[] = ['clinical', 'therapy', 'labs', 'uncertainty'];
const CARD_COLLISION_GAP_PERCENT = 20;

@Component({
  selector: 'app-patient-timetable-page',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './patient-timetable-page.component.html',
  styleUrl: './patient-timetable-page.component.scss',
})
export class PatientTimetablePageComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);

  readonly sessionId = signal<number | null>(null);
  readonly timelineId = signal<number | null>(null);
  readonly timeline = signal<InspectionSessionTimeline | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly selectedEventId = signal<string | null>(null);

  readonly orderedEvents = computed(() =>
    [...(this.timeline()?.events ?? [])].sort((a, b) => a.sort_order - b.sort_order),
  );

  readonly dateRange = computed<TimelineDateRange>(() => {
    const dates = this.orderedEvents()
      .map((event) => this.parseEventDate(event.event_date))
      .filter((value): value is Date => value !== null)
      .sort((a, b) => a.getTime() - b.getTime());
    return {
      start: dates[0] ?? null,
      end: dates[dates.length - 1] ?? null,
    };
  });

  readonly renderedEvents = computed<RenderedTimelineEvent[]>(() => {
    const events = this.orderedEvents();
    const range = this.dateRange();
    const items = events.map((event, index) => {
      const positionPercent = this.resolveEventPosition(event, index, events.length, range);
      return {
        event,
        lane: this.resolveLane(event),
        color: EVENT_COLORS[event.event_type] ?? EVENT_COLORS.other,
        positionPercent,
        align: this.resolveEventAlign(positionPercent),
        stackLevel: 0,
      };
    });
    return this.assignStackLevels(items);
  });

  readonly lanes = computed(() =>
    TIMETABLE_LANES.map((lane) => {
      const items = this.renderedEvents().filter((item) => item.lane === lane);
      const levelCount = Math.max(1, ...items.map((item) => item.stackLevel + 1));
      return {
        id: lane,
        label: TIMETABLE_LANE_LABELS[lane],
        items,
        levelCount,
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
        shortLabel: new Intl.DateTimeFormat(undefined, { month: 'short', timeZone: 'UTC' }).format(cursor),
        positionPercent: totalMonths <= 1 ? 0 : (offset / totalMonths) * 100,
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
    return timeline.generation_note || 'Generated with the configured local timeline extraction model.';
  });

  readonly sourceModelLabel = computed(() => {
    const timeline = this.timeline();
    if (!timeline?.source_model) {
      return this.generationStatusLabel();
    }
    return timeline.source_model;
  });

  async ngOnInit(): Promise<void> {
    this.route.paramMap.subscribe((params) => {
      const sessionId = Number(params.get('sessionId'));
      const timelineIdRaw = params.get('timelineId');
      const timelineId = timelineIdRaw ? Number(timelineIdRaw) : null;
      void this.handleRouteChange(sessionId, timelineId);
    });
  }

  async loadTimeline(sessionId: number, timelineId: number | null): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const payload = timelineId
        ? await fetchInspectionSessionTimelineById(sessionId, timelineId)
        : await fetchInspectionSessionTimeline(sessionId);
      this.timeline.set(payload);
      this.timelineId.set(payload.timeline_id ?? timelineId ?? null);
      this.selectedEventId.set(payload.events[0]?.event_id ?? null);
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
      this.selectedEventId.set(payload.events[0]?.event_id ?? null);
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
    this.selectedEventId.set(event.event_id);
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

  private parseEventDate(value: string | null): Date | null {
    if (!value) {
      return null;
    }
    const parsed = new Date(`${value.slice(0, 10)}T00:00:00Z`);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  private monthLabel(value: Date): string {
    return new Intl.DateTimeFormat(undefined, {
      month: 'short',
      year: 'numeric',
      timeZone: 'UTC',
    }).format(value);
  }

  private monthsBetweenCount(start: Date, end: Date): number {
    return (end.getUTCFullYear() - start.getUTCFullYear()) * 12 + (end.getUTCMonth() - start.getUTCMonth());
  }

  private resolveEventPosition(
    event: InspectionTimelineEvent,
    index: number,
    total: number,
    range: TimelineDateRange,
  ): number {
    const eventDate = this.parseEventDate(event.event_date);
    if (eventDate && range.start && range.end) {
      const span = Math.max(1, range.end.getTime() - range.start.getTime());
      const elapsed = eventDate.getTime() - range.start.getTime();
      return Math.min(100, Math.max(0, (elapsed / span) * 100));
    }
    if (total <= 1) {
      return 50;
    }
    return (index / (total - 1)) * 100;
  }

  private resolveEventAlign(positionPercent: number): TimelineCardAlign {
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
      const laneItems = nextItems
        .filter((item) => item.lane === lane)
        .sort((a, b) => a.positionPercent - b.positionPercent || a.event.sort_order - b.event.sort_order);
      const latestPositionByLevel: number[] = [];
      for (const item of laneItems) {
        let level = latestPositionByLevel.findIndex(
          (latestPosition) => item.positionPercent - latestPosition >= CARD_COLLISION_GAP_PERCENT,
        );
        if (level === -1) {
          level = latestPositionByLevel.length;
        }
        item.stackLevel = level;
        latestPositionByLevel[level] = item.positionPercent;
      }
    }
    return nextItems;
  }

  private resolveLane(event: InspectionTimelineEvent): TimetableLane {
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

  private isNotFoundError(error: unknown): boolean {
    return error instanceof Error && error.message.toLowerCase().includes('not found');
  }
}
