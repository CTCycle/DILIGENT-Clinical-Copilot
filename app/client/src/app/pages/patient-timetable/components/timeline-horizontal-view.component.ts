// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, ElementRef, computed, effect, inject, input, output, viewChild } from '@angular/core';
import { LucideChevronLeft, LucideChevronRight } from '@lucide/angular';

import { InspectionTimelineEvent } from '../../../core/models/inspection-types';
import { normalizeTimelineDate } from '../timeline-date';
import {
  eventCategory,
  eventColor,
  eventDateLabel,
  isDateInferred,
  isEventUncertain,
  TIMELINE_CATEGORY_LABELS,
  TimelineCategory,
  TimelineEventGroup,
} from '../timeline-presentation';

type HorizontalLane = {
  category: TimelineCategory;
  label: string;
  barTracks: number;
  cells: InspectionTimelineEvent[][];
};

type RangeBar = {
  event: InspectionTimelineEvent;
  lane: number;
  startColumn: number;
  endColumn: number;
  track: number;
  label: string;
};

@Component({
  selector: 'app-timeline-horizontal-view',
  standalone: true,
  imports: [LucideChevronLeft, LucideChevronRight],
  templateUrl: './timeline-horizontal-view.component.html',
  styleUrl: './timeline-horizontal-view.component.scss',
})
export class TimelineHorizontalViewComponent {
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly scroller = viewChild<ElementRef<HTMLElement>>('scroller');

  /** Dated groups in chronological order; each becomes one column. */
  readonly columns = input.required<readonly TimelineEventGroup[]>();
  readonly categories = input.required<readonly TimelineCategory[]>();
  readonly selectedEventId = input<string | null>(null);
  readonly eventSelect = output<InspectionTimelineEvent>();

  private readonly rangeBarsByLane = computed(() => {
    const columns = this.columns();
    const categories = this.categories();
    const bars: RangeBar[] = [];
    columns.forEach((column, startColumn) => {
      for (const event of column.events) {
        const start = normalizeTimelineDate(event.event_date);
        const end = normalizeTimelineDate(event.event_date_end ?? null);
        const lane = categories.indexOf(eventCategory(event));
        if (!start || !end || end.startDay <= start.startDay || lane < 0) continue;
        let endColumn = startColumn;
        while (endColumn + 1 < columns.length && (columns[endColumn + 1].startDay ?? Infinity) <= end.endDay) endColumn += 1;
        bars.push({ event, lane, startColumn, endColumn, track: 0, label: `${event.title}: ${eventDateLabel(event)}` });
      }
    });
    // Stack overlapping ranges within a lane on separate tracks.
    const trackEnds = new Map<number, number[]>();
    for (const bar of bars) {
      const ends = trackEnds.get(bar.lane) ?? [];
      const track = ends.findIndex((endColumn) => endColumn < bar.startColumn);
      bar.track = track === -1 ? ends.length : track;
      ends[bar.track] = bar.endColumn;
      trackEnds.set(bar.lane, ends);
    }
    return { bars, trackEnds };
  });

  readonly rangeBars = computed(() => this.rangeBarsByLane().bars);

  readonly lanes = computed<HorizontalLane[]>(() => {
    const trackEnds = this.rangeBarsByLane().trackEnds;
    return this.categories().map((category, laneIndex) => ({
      category,
      label: TIMELINE_CATEGORY_LABELS[category],
      barTracks: trackEnds.get(laneIndex)?.length ?? 0,
      cells: this.columns().map((column) => column.events.filter((event) => eventCategory(event) === category)),
    }));
  });

  readonly gridTemplateColumns = computed(() => `var(--lane-label-width) repeat(${this.columns().length}, var(--column-width))`);

  readonly eventColor = eventColor;

  constructor() {
    effect(() => {
      const selectedId = this.selectedEventId();
      if (!selectedId) return;
      queueMicrotask(() => [...this.host.nativeElement.querySelectorAll<HTMLElement>('[data-chip-id]')]
        .find((chip) => chip.dataset['chipId'] === selectedId)
        ?.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' }));
    });
  }

  isApproximate(event: InspectionTimelineEvent): boolean {
    return normalizeTimelineDate(event.event_date)?.precision !== 'day';
  }

  needsReview(event: InspectionTimelineEvent): boolean {
    return isEventUncertain(event) || isDateInferred(event);
  }

  chipLabel(event: InspectionTimelineEvent): string {
    const flags = [
      this.isApproximate(event) ? 'approximate date' : '',
      isDateInferred(event) ? 'inferred date' : '',
    ].filter(Boolean).join(', ');
    return `${event.title}, ${eventDateLabel(event)}${flags ? `, ${flags}` : ''}`;
  }

  scrollTrack(direction: -1 | 1): void {
    const element = this.scroller()?.nativeElement;
    element?.scrollBy({ left: direction * element.clientWidth * 0.8, behavior: 'smooth' });
  }
}
