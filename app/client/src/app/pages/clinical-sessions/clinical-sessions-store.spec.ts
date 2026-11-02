// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { InspectionSessionItem } from '../../core/models/inspection-types';
import { ClinicalSessionsStore } from './clinical-sessions-store';

function item(id: number, overrides: Partial<InspectionSessionItem> = {}): InspectionSessionItem {
  const created = new Date(Date.now() - id * 24 * 60 * 60 * 1000).toISOString();
  return {
    session_id: id,
    patient_name: `Patient ${id}`,
    visit_date: null,
    session_timestamp: null,
    version: 1,
    original_session_id: null,
    status: 'completed',
    created_at: created,
    updated_at: created,
    progress: null,
    progress_stage: null,
    total_duration: null,
    has_report: false,
    has_timeline: false,
    can_generate_timeline: false,
    ...overrides,
  };
}

function page(items: InspectionSessionItem[], total: number): Response {
  return new Response(JSON.stringify({ items, total, offset: 0, limit: 100 }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('ClinicalSessionsStore', () => {
  let store: ClinicalSessionsStore;
  let urls: URL[];

  beforeEach(() => {
    store = new ClinicalSessionsStore();
    urls = [];
  });

  afterEach(() => {
    store.destroy();
    vi.restoreAllMocks();
  });

  function mockFetch(handler: (url: URL) => Response): void {
    vi.restoreAllMocks();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = new URL(String(input), 'http://localhost');
      urls.push(url);
      return handler(url);
    });
  }

  it('pages through the 90-day window once and dedupes overlapping pages', async () => {
    mockFetch((url) => (url.searchParams.get('offset') === '0'
      ? page([item(1), item(2)], 3)
      : page([item(2), item(3)], 3)));

    await store.loadRecent();

    expect(urls.map((url) => url.searchParams.get('offset'))).toEqual(['0', '2']);
    expect(urls[0].searchParams.get('date_field')).toBe('created');
    expect(Date.parse(urls[0].searchParams.get('date_from')!)).toBeLessThan(Date.now());
    expect(store.recentCount()).toBe(3);

    store.setFilters({ statuses: ['failed'] });
    expect(urls).toHaveLength(2);
  });

  it('resolves free-text search server-side so recent and older sessions match identically', async () => {
    mockFetch((url) => {
      if (url.searchParams.get('search') === 'hepatic') {
        // Matches report content only: neither patient name contains the term.
        return url.searchParams.has('created_before')
          ? page([item(200)], 1)
          : page([item(2)], 1);
      }
      return page([item(1), item(2)], 2);
    });
    await store.loadRecent();
    store.setIncludeOlder(true);
    await vi.waitFor(() => expect(store.olderTotal()).not.toBeNull());

    vi.useFakeTimers();
    store.setFilters({ query: 'hepatic' });
    await vi.advanceTimersByTimeAsync(300);
    vi.useRealTimers();
    await vi.waitFor(() => expect(store.visible().map((entry) => entry.session_id)).toEqual([2, 200]));

    const searched = urls.filter((url) => url.searchParams.get('search') === 'hepatic');
    expect(searched.map((url) => url.searchParams.has('created_before'))).toEqual(expect.arrayContaining([true, false]));
  });

  it('merges only changed sessions on sync and reflects removals', async () => {
    mockFetch(() => page([item(1, { status: 'running' }), item(2)], 2));
    await store.loadRecent();

    mockFetch(() => page([item(1, { status: 'completed', updated_at: new Date().toISOString() })], 1));
    const changed = await store.syncChanges();

    expect(changed.map((entry) => entry.status)).toEqual(['completed']);
    expect(urls.at(-1)!.searchParams.get('date_field')).toBe('updated');
    expect(store.visible().find((entry) => entry.session_id === 1)?.status).toBe('completed');
    expect(store.hasActiveSessions()).toBe(false);

    store.remove(2);
    expect(store.recentCount()).toBe(1);
  });

  it('queries older sessions server-side only on request, scoped before the cache window', async () => {
    mockFetch((url) => (url.searchParams.has('created_before')
      ? page([item(200), item(201)], 3)
      : page([item(1)], 1)));
    await store.loadRecent();
    expect(urls.some((url) => url.searchParams.has('created_before'))).toBe(false);

    store.setFilters({ statuses: ['failed', 'queued'] });
    store.setIncludeOlder(true);
    await vi.waitFor(() => expect(store.olderCount()).toBe(2));

    const historical = urls.find((url) => url.searchParams.has('created_before'))!;
    expect(historical.searchParams.getAll('status')).toEqual(['failed', 'queued']);
    expect(store.hasMoreOlder()).toBe(true);
    expect(store.olderTotal()).toBe(3);

    store.setIncludeOlder(false);
    expect(store.olderCount()).toBe(0);
  });
});
