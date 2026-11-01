// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import {
  InspectionSessionDateField,
  InspectionSessionItem,
  InspectionSessionQuery,
  InspectionSessionSort,
  InspectionSessionStatus,
} from '../../core/models/inspection-types';
import { sessionStatusLabel } from '../../core/utils/session-status';

export const CACHE_WINDOW_DAYS = 90;

export type SessionDatePreset = 'any' | 'today' | 'last7' | 'last30' | 'custom';

export type SessionDateFilter = {
  field: InspectionSessionDateField;
  preset: SessionDatePreset;
  /** Inclusive local calendar day (YYYY-MM-DD), used by the custom preset. */
  from: string;
  /** Inclusive local calendar day (YYYY-MM-DD), used by the custom preset. */
  to: string;
};

export type SessionFilterState = {
  query: string;
  statuses: InspectionSessionStatus[];
  date: SessionDateFilter;
  sort: InspectionSessionSort;
};

/** Half-open range [from, to); either bound may be open. */
export type SessionDateRange = {
  fromInstant: Date | null;
  toInstant: Date | null;
  fromDay: string | null;
  toDay: string | null;
};

export const DEFAULT_SESSION_FILTERS: SessionFilterState = {
  query: '',
  statuses: [],
  date: { field: 'created', preset: 'any', from: '', to: '' },
  sort: 'updated_desc',
};

function dayKey(value: Date): string {
  const month = `${value.getMonth() + 1}`.padStart(2, '0');
  const day = `${value.getDate()}`.padStart(2, '0');
  return `${value.getFullYear()}-${month}-${day}`;
}

function parseDay(value: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return null;
  return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
}

function addDays(value: Date, days: number): Date {
  return new Date(value.getFullYear(), value.getMonth(), value.getDate() + days);
}

function rangeFromDays(from: Date | null, toExclusive: Date | null): SessionDateRange {
  return {
    fromInstant: from,
    toInstant: toExclusive,
    fromDay: from ? dayKey(from) : null,
    toDay: toExclusive ? dayKey(toExclusive) : null,
  };
}

export function resolveDateRange(date: SessionDateFilter, now: Date): SessionDateRange | null {
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  switch (date.preset) {
    case 'today':
      return rangeFromDays(today, addDays(today, 1));
    case 'last7':
      return rangeFromDays(addDays(today, -6), addDays(today, 1));
    case 'last30':
      return rangeFromDays(addDays(today, -29), addDays(today, 1));
    case 'custom': {
      const from = parseDay(date.from);
      const to = parseDay(date.to);
      if (!from && !to) return null;
      return rangeFromDays(from, to ? addDays(to, 1) : null);
    }
    default:
      return null;
  }
}

export function cacheCutoff(now: Date): Date {
  return new Date(now.getTime() - CACHE_WINDOW_DAYS * 24 * 60 * 60 * 1000);
}

/** True when the created-date range asks for sessions older than the cached window. */
export function rangeReachesHistory(filters: SessionFilterState, now: Date): boolean {
  if (filters.date.field !== 'created') return false;
  const range = resolveDateRange(filters.date, now);
  if (!range) return false;
  return range.fromInstant === null || range.fromInstant < cacheCutoff(now);
}

function matchesDate(item: InspectionSessionItem, date: SessionDateFilter, range: SessionDateRange): boolean {
  if (date.field === 'visit') {
    if (!item.visit_date) return false;
    return (range.fromDay === null || item.visit_date >= range.fromDay)
      && (range.toDay === null || item.visit_date < range.toDay);
  }
  const raw = date.field === 'updated' ? item.updated_at : item.created_at;
  const time = Date.parse(raw);
  if (Number.isNaN(time)) return false;
  return (range.fromInstant === null || time >= range.fromInstant.getTime())
    && (range.toInstant === null || time < range.toInstant.getTime());
}

function searchableText(item: InspectionSessionItem): string {
  return [
    `#${item.session_id}`,
    item.patient_name ?? '',
    sessionStatusLabel(item.status),
    item.visit_date ?? '',
  ].join('\n').toLowerCase();
}

export function matchesFilters(
  item: InspectionSessionItem,
  filters: SessionFilterState,
  now: Date,
): boolean {
  if (filters.statuses.length && !filters.statuses.includes(item.status)) return false;
  const query = filters.query.trim().toLowerCase();
  if (query && !searchableText(item).includes(query)) return false;
  const range = resolveDateRange(filters.date, now);
  return range === null || matchesDate(item, filters.date, range);
}

export function compareSessions(sort: InspectionSessionSort): (a: InspectionSessionItem, b: InspectionSessionItem) => number {
  const key = (item: InspectionSessionItem): number => Date.parse(sort === 'updated_desc' ? item.updated_at : item.created_at) || 0;
  const direction = sort === 'created_asc' ? 1 : -1;
  return (a, b) => direction * (key(a) - key(b)) || direction * (a.session_id - b.session_id);
}

export function filterSessions(
  items: readonly InspectionSessionItem[],
  filters: SessionFilterState,
  now: Date,
): InspectionSessionItem[] {
  return items.filter((item) => matchesFilters(item, filters, now)).sort(compareSessions(filters.sort));
}

export function activeFilterCount(filters: SessionFilterState): number {
  return (filters.statuses.length ? 1 : 0)
    + (filters.date.preset !== 'any' ? 1 : 0)
    + (filters.query.trim() ? 1 : 0);
}

/** Builds the server query equivalent to the local filters (used for historical retrieval). */
export function toServerQuery(
  filters: SessionFilterState,
  now: Date,
): Pick<InspectionSessionQuery, 'search' | 'statuses' | 'sort' | 'date_field' | 'date_from' | 'date_to'> {
  const range = resolveDateRange(filters.date, now);
  const visit = filters.date.field === 'visit';
  // Visit dates are calendar days; the server compares the UTC date part of the bound.
  const bound = (instant: Date | null, day: string | null): string | undefined => {
    if (visit) return day ? `${day}T00:00:00Z` : undefined;
    return instant ? instant.toISOString() : undefined;
  };
  return {
    search: filters.query.trim() || undefined,
    statuses: filters.statuses.length ? filters.statuses : undefined,
    sort: filters.sort,
    date_field: range ? filters.date.field : undefined,
    date_from: range ? bound(range.fromInstant, range.fromDay) : undefined,
    date_to: range ? bound(range.toInstant, range.toDay) : undefined,
  };
}
