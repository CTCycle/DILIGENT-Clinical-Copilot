// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { APP_LOCALE } from '../../core/constants';
import {
  InspectionTimelineEvent,
  InspectionTimelineEventType,
  InspectionTimelineTimingType,
} from '../../core/models/inspection-types';
import { dayToUtcDate, normalizeTimelineDate, NormalizedTimelineDate } from './timeline-date';

export type TimelineCategory = 'clinical' | 'therapy' | 'labs' | 'other';
export type TimelineViewMode = 'list' | 'horizontal';

export type TimelineEventGroup = {
  key: string;
  label: string;
  dateHint: string;
  isUnanchored: boolean;
  startDay: number | null;
  gapLabel: string | null;
  events: InspectionTimelineEvent[];
};

export const TIMELINE_CATEGORIES: readonly TimelineCategory[] = ['clinical', 'therapy', 'labs', 'other'];

export const TIMELINE_CATEGORY_LABELS: Record<TimelineCategory, string> = {
  clinical: 'Clinical',
  therapy: 'Medications',
  labs: 'Laboratory',
  other: 'Other',
};

const CATEGORY_BY_TYPE: Record<InspectionTimelineEventType, TimelineCategory> = {
  disease: 'clinical',
  therapy: 'therapy',
  lab: 'labs',
  other: 'other',
};

const CATEGORY_COLORS: Record<TimelineCategory, string> = {
  clinical: 'var(--color-chart-disease)',
  therapy: 'var(--color-chart-therapy)',
  labs: 'var(--color-chart-lab)',
  other: 'var(--color-chart-other)',
};

const TIMING_LABELS: Record<InspectionTimelineTimingType, string> = {
  explicit_date: 'Explicit date',
  relative: 'Relative timing',
  duration: 'Duration',
  recurring: 'Recurring',
  uncertain: 'Uncertain timing',
  ordering: 'Relative order',
};

const DAY_FORMATTER = new Intl.DateTimeFormat(APP_LOCALE, { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
const MONTH_FORMATTER = new Intl.DateTimeFormat(APP_LOCALE, { month: 'long', year: 'numeric', timeZone: 'UTC' });
const YEAR_FORMATTER = new Intl.DateTimeFormat(APP_LOCALE, { year: 'numeric', timeZone: 'UTC' });

export function eventCategory(event: InspectionTimelineEvent): TimelineCategory {
  return CATEGORY_BY_TYPE[event.event_type] ?? 'other';
}

export function eventColor(event: InspectionTimelineEvent): string {
  return CATEGORY_COLORS[eventCategory(event)];
}

export function timingLabel(value: InspectionTimelineTimingType): string {
  return TIMING_LABELS[value] ?? value;
}

export function confidenceLabel(value: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Not scored';
  if (value >= 0.9) return 'Very high';
  if (value >= 0.75) return 'High';
  if (value >= 0.55) return 'Moderate';
  if (value >= 0.35) return 'Low';
  return 'Very low';
}

export function formatTimelineDate(value: NormalizedTimelineDate, edge: 'start' | 'end' = 'start'): string {
  const date = dayToUtcDate(edge === 'start' ? value.startDay : value.endDay);
  if (value.precision === 'day') return DAY_FORMATTER.format(date);
  if (value.precision === 'month') return MONTH_FORMATTER.format(dayToUtcDate(value.startDay));
  return YEAR_FORMATTER.format(date);
}

export function formatMonth(value: Date): string {
  return MONTH_FORMATTER.format(value);
}

export function eventDateLabel(event: InspectionTimelineEvent): string {
  const start = normalizeTimelineDate(event.event_date);
  if (!start) return 'Date not reported';
  const end = normalizeTimelineDate(event.event_date_end ?? null);
  const label = formatTimelineDate(start);
  return end && end.startDay > start.startDay ? `${label} – ${formatTimelineDate(end, 'end')}` : label;
}

export function eventDateSummary(event: InspectionTimelineEvent): string {
  const start = normalizeTimelineDate(event.event_date);
  const extracted = event.extracted_timing_text || event.relative_time;
  if (!start) return extracted ? `No canonical date · ${extracted}` : 'No canonical date reported';
  let summary = eventDateLabel(event);
  if (start.precision !== 'day') summary += ` · ${start.precision}-level date`;
  return extracted && extracted !== event.event_date ? `${summary} · ${extracted}` : summary;
}

export function precisionLabel(value: string | null): string {
  const precision = normalizeTimelineDate(value)?.precision;
  return precision ? `${precision[0].toUpperCase()}${precision.slice(1)} precision` : 'Not reported';
}

export function isEventUncertain(event: InspectionTimelineEvent): boolean {
  return event.date_certainty === 'uncertain'
    || event.timing_type === 'uncertain'
    || event.timing_type === 'ordering'
    || !normalizeTimelineDate(event.event_date);
}

export function isDateInferred(event: InspectionTimelineEvent): boolean {
  return event.date_certainty === 'inferred' && Boolean(normalizeTimelineDate(event.event_date));
}

export function hasEventEvidence(event: InspectionTimelineEvent | null): boolean {
  return Boolean(event?.source_evidence?.trim());
}
