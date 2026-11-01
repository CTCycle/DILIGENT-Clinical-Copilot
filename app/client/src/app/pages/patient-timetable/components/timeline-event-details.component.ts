// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, input, output } from '@angular/core';

import { InspectionTimelineEvent } from '../../../core/models/inspection-types';
import {
  confidenceLabel,
  eventDateSummary,
  hasEventEvidence,
  precisionLabel,
  timingLabel,
} from '../timeline-presentation';

@Component({
  selector: 'app-timeline-event-details',
  standalone: true,
  templateUrl: './timeline-event-details.component.html',
  styleUrl: './timeline-event-details.component.scss',
})
export class TimelineEventDetailsComponent {
  readonly event = input.required<InspectionTimelineEvent>();
  readonly linkedEvents = input<readonly InspectionTimelineEvent[]>([]);
  readonly linkedSelect = output<InspectionTimelineEvent>();

  readonly hasEvidence = computed(() => hasEventEvidence(this.event()));
  readonly dateSummary = computed(() => eventDateSummary(this.event()));
  readonly certaintyLabel = computed(() => {
    const event = this.event();
    const precision = precisionLabel(event.event_date);
    return event.date_certainty && event.date_certainty !== 'uncertain' && event.event_date
      ? `${precision} · ${event.date_certainty}`
      : precision;
  });

  readonly timingLabel = timingLabel;
  readonly confidenceLabel = confidenceLabel;
}
