// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

export type TimelineDatePrecision = 'day' | 'month' | 'year';

export type NormalizedTimelineDate = {
  value: string;
  startDay: number;
  endDay: number;
  precision: TimelineDatePrecision;
};

const MS_PER_DAY = 86_400_000;

function utcDay(year: number, month: number, day: number): number | null {
  const date = new Date(Date.UTC(year, month - 1, day));
  if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) return null;
  return Math.floor(date.getTime() / MS_PER_DAY);
}

export function normalizeTimelineDate(value: string | null): NormalizedTimelineDate | null {
  if (!value) return null;
  const exact = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  const month = /^(\d{4})-(\d{2})$/.exec(value);
  const year = /^(\d{4})$/.exec(value);
  const match = exact ?? month ?? year;
  if (!match) return null;
  const yearNumber = Number(match[1]);
  const monthNumber = exact || month ? Number(match[2]) : 1;
  const dayNumber = exact ? Number(match[3]) : 1;
  const startDay = utcDay(yearNumber, monthNumber, dayNumber);
  if (startDay === null) return null;
  if (exact) return { value, startDay, endDay: startDay, precision: 'day' };
  if (month) {
    const nextMonth = monthNumber === 12 ? utcDay(yearNumber + 1, 1, 1) : utcDay(yearNumber, monthNumber + 1, 1);
    return nextMonth === null ? null : { value, startDay, endDay: nextMonth - 1, precision: 'month' };
  }
  const nextYear = utcDay(yearNumber + 1, 1, 1);
  return nextYear === null ? null : { value, startDay, endDay: nextYear - 1, precision: 'year' };
}

/** Compact label for the distance between two timeline columns, e.g. "+5 d". */
export function formatDayGap(days: number): string | null {
  if (!Number.isFinite(days) || days <= 0) return null;
  if (days < 60) return `+${days} d`;
  if (days < 730) return `+${Math.round(days / 30.44)} mo`;
  return `+${Math.round(days / 365.25)} y`;
}

export function dayToUtcDate(day: number): Date {
  return new Date(day * MS_PER_DAY);
}
