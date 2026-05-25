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
  left: number;
  width: number;
  color: string;
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
    const count = Math.max(events.length, 1);
    return events.map((event, index) => {
      const lane = this.resolveLane(event);
      const left = count === 1 ? 45 : 8 + (index * 84) / Math.max(count - 1, 1);
      const width = event.timing_type === 'duration' ? Math.min(34, 12 + count * 2) : 10;
      return {
        event,
        lane,
        left,
        width,
        color: EVENT_COLORS[event.event_type] ?? EVENT_COLORS.other,
      };
    });
  });

  readonly selectedEvent = computed(() => {
    const selectedId = this.selectedEventId();
    return this.orderedEvents().find((event) => event.event_id === selectedId) ?? null;
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
