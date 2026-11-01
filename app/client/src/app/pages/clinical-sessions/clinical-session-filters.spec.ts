// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { describe, expect, it } from 'vitest';

import { InspectionSessionItem } from '../../core/models/inspection-types';
import {
  DEFAULT_SESSION_FILTERS,
  SessionFilterState,
  cacheCutoff,
  filterSessions,
  rangeReachesHistory,
  toServerQuery,
} from './clinical-session-filters';

const NOW = new Date(2026, 10, 1, 12, 0, 0);

function item(id: number, overrides: Partial<InspectionSessionItem> = {}): InspectionSessionItem {
  return {
    session_id: id,
    patient_name: `Patient ${id}`,
    visit_date: null,
    session_timestamp: null,
    version: 1,
    original_session_id: null,
    status: 'completed',
    created_at: new Date(2026, 9, id, 9, 0).toISOString(),
    updated_at: new Date(2026, 9, id, 10, 0).toISOString(),
    progress: null,
    progress_stage: null,
    total_duration: null,
    has_report: false,
    has_timeline: false,
    can_generate_timeline: false,
    ...overrides,
  };
}

function filters(patch: Partial<SessionFilterState>): SessionFilterState {
  return { ...DEFAULT_SESSION_FILTERS, ...patch };
}

function dayString(value: Date): string {
  const month = `${value.getMonth() + 1}`.padStart(2, '0');
  return `${value.getFullYear()}-${month}-${`${value.getDate()}`.padStart(2, '0')}`;
}

describe('session filters', () => {
  const sessions = [
    item(1, { status: 'failed', patient_name: 'Antonello Venditti' }),
    item(2, { status: 'queued', patient_name: 'Penelope Cruz' }),
    item(3, { status: 'completed', patient_name: 'venditti test' }),
  ];
  const ids = (state: SessionFilterState, list = sessions): number[] => (
    filterSessions(list, state, NOW).map((entry) => entry.session_id)
  );

  it('combines statuses with OR and categories with AND', () => {
    expect(ids(filters({ statuses: ['failed', 'queued'], sort: 'created_asc' }))).toEqual([1, 2]);
    expect(ids(filters({ statuses: ['failed', 'completed'], query: 'VENDITTI', sort: 'created_asc' }))).toEqual([1, 3]);
    expect(ids(filters({ statuses: ['queued'], query: 'venditti' }))).toEqual([]);
  });

  it('matches partial names and session ids case-insensitively', () => {
    expect(ids(filters({ query: 'nelo' }))).toEqual([2]);
    expect(ids(filters({ query: '#3' }))).toEqual([3]);
  });

  it('applies half-open local-day windows for presets and custom ranges', () => {
    const edge = [
      item(1, { created_at: new Date(2026, 9, 31, 23, 59, 59).toISOString() }),
      item(2, { created_at: new Date(2026, 10, 1, 0, 0, 0).toISOString() }),
    ];
    const today = filters({ date: { field: 'created', preset: 'today', from: '', to: '' } });
    expect(ids(today, edge)).toEqual([2]);
    const custom = filters({ date: { field: 'created', preset: 'custom', from: '2026-10-31', to: '2026-10-31' } });
    expect(ids(custom, edge)).toEqual([1]);
  });

  it('filters on the visit date as a calendar day', () => {
    const visits = [item(1, { visit_date: '2026-10-30' }), item(2, { visit_date: null })];
    const state = filters({ date: { field: 'visit', preset: 'custom', from: '2026-10-30', to: '2026-10-30' } });
    expect(ids(state, visits)).toEqual([1]);
    expect(toServerQuery(state, NOW)).toMatchObject({
      date_field: 'visit',
      date_from: '2026-10-30T00:00:00Z',
      date_to: '2026-10-31T00:00:00Z',
    });
  });

  it('flags created-date ranges that reach beyond the 90-day cache window', () => {
    const cutoff = cacheCutoff(NOW);
    const day = (offset: number): string => (
      dayString(new Date(cutoff.getFullYear(), cutoff.getMonth(), cutoff.getDate() + offset))
    );
    const range = (from: string): SessionFilterState => filters({
      date: { field: 'created', preset: 'custom', from, to: '' },
    });
    expect(rangeReachesHistory(range(day(-1)), NOW)).toBe(true);
    expect(rangeReachesHistory(range(day(1)), NOW)).toBe(false);
    expect(rangeReachesHistory(filters({ date: { field: 'created', preset: 'last30', from: '', to: '' } }), NOW)).toBe(false);
  });
});
