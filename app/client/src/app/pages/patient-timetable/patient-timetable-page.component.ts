import { CommonModule } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';

import {
  fetchInspectionSessionTimeline,
  generateInspectionSessionTimeline,
} from '../../core/services/inspection-api';
import {
  InspectionSessionTimeline,
  InspectionTimelineEvent,
  InspectionTimelineEventType,
  InspectionTimelineTimingType,
} from '../../core/models/types';

type TimetableLane = 'clinical' | 'therapy' | 'labs' | 'uncertainty';

type RenderedTimelineEvent = {
  event: InspectionTimelineEvent;
  lane: TimetableLane;
  color: string;
};

type TimelineDateRange = {
  start: Date | null;
  end: Date | null;
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

@Component({
  selector: 'app-patient-timetable-page',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './patient-timetable-page.component.html',
  styleUrl: './patient-timetable-page.component.scss',
})
export class PatientTimetablePageComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);

  readonly sessionId = signal<number | null>(null);
  readonly timeline = signal<InspectionSessionTimeline | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly selectedEventId = signal<string | null>(null);

  readonly orderedEvents = computed(() =>
    [...(this.timeline()?.events ?? [])].sort((a, b) => a.sort_order - b.sort_order),
  );

  readonly renderedEvents = computed<RenderedTimelineEvent[]>(() => {
    const events = this.orderedEvents();
    return events.map((event) => {
      const lane = this.resolveLane(event);
      return {
        event,
        lane,
        color: EVENT_COLORS[event.event_type] ?? EVENT_COLORS.other,
      };
    });
  });

  readonly lanes = computed(() =>
    TIMETABLE_LANES.map((lane) => ({
      id: lane,
      label: TIMETABLE_LANE_LABELS[lane],
      items: this.renderedEvents().filter((item) => item.lane === lane),
    })),
  );

  readonly selectedEvent = computed(() => {
    const selectedId = this.selectedEventId();
    return this.orderedEvents().find((event) => event.event_id === selectedId) ?? null;
  });

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

  readonly scaleLabels = computed(() => {
    const range = this.dateRange();
    if (!range.start || !range.end) {
      return ['Start', 'Middle', 'End'];
    }
    return this.monthsBetween(range.start, range.end);
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

  async ngOnInit(): Promise<void> {
    const id = Number(this.route.snapshot.paramMap.get('sessionId'));
    if (!Number.isFinite(id) || id <= 0) {
      this.error.set('Invalid session id.');
      return;
    }
    this.sessionId.set(id);
    await this.loadTimeline(id);
  }

  async loadTimeline(sessionId: number): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const payload = await fetchInspectionSessionTimeline(sessionId);
      this.timeline.set(payload);
      this.selectedEventId.set(payload.events[0]?.event_id ?? null);
    } catch (error) {
      this.timeline.set(null);
      if (this.isNotFoundError(error)) {
        this.error.set('[ERROR] No timetable is available yet. Select Regenerate to create one.');
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
      this.selectedEventId.set(payload.events[0]?.event_id ?? null);
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

  private monthsBetween(start: Date, end: Date): string[] {
    const labels: string[] = [];
    const cursor = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth(), 1));
    const final = new Date(Date.UTC(end.getUTCFullYear(), end.getUTCMonth(), 1));
    while (cursor.getTime() <= final.getTime() && labels.length < 12) {
      labels.push(this.monthLabel(cursor));
      cursor.setUTCMonth(cursor.getUTCMonth() + 1);
    }
    return labels.length > 0 ? labels : [this.monthLabel(start)];
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
    return error instanceof Error && error.message.includes('not found');
  }
}
