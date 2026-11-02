// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable, computed, signal } from '@angular/core';

import { InspectionSessionItem } from '../../core/models/inspection-types';
import { fetchInspectionSessions } from '../../core/services/clinical-sessions-api';
import { isActiveSessionStatus } from '../../core/utils/session-status';
import { formatUnknownError } from '../../core/utils';
import {
  DEFAULT_SESSION_FILTERS,
  SessionFilterState,
  cacheCutoff,
  compareSessions,
  filterSessions,
  rangeReachesHistory,
  toServerQuery,
} from './clinical-session-filters';

const PAGE_SIZE = 100;
const OLDER_QUERY_DEBOUNCE_MS = 250;

/**
 * Client-side session cache. Sessions created in the last 90 days are loaded once and filtered
 * locally; older sessions are queried server-side only when explicitly requested. Free-text search
 * always goes through the server (which also matches report content) for both ranges, so results do
 * not depend on session age; only the matching session ids are kept for the recent set.
 */
@Injectable()
export class ClinicalSessionsStore {
  readonly filters = signal<SessionFilterState>(DEFAULT_SESSION_FILTERS);
  readonly includeOlder = signal(false);
  readonly loading = signal(false);
  readonly loaded = signal(false);
  readonly error = signal<string | null>(null);
  readonly olderLoading = signal(false);
  readonly olderError = signal<string | null>(null);
  readonly olderTotal = signal<number | null>(null);

  private readonly recent = signal<InspectionSessionItem[]>([]);
  private readonly older = signal<InspectionSessionItem[]>([]);
  private readonly now = signal(new Date());
  private readonly recentSearch = signal<{ query: string; ids: ReadonlySet<number> } | null>(null);
  private recentGeneration = 0;
  private olderGeneration = 0;
  private searchGeneration = 0;
  private olderTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private searchTimer: ReturnType<typeof globalThis.setTimeout> | null = null;

  readonly recentCount = computed(() => this.recent().length);
  readonly olderCount = computed(() => this.older().length);
  readonly hasMoreOlder = computed(() => {
    const total = this.olderTotal();
    return this.includeOlder() && total !== null && this.older().length < total;
  });
  readonly hasActiveSessions = computed(() => (
    this.recent().some((item) => isActiveSessionStatus(item.status))
    || this.older().some((item) => isActiveSessionStatus(item.status))
  ));
  readonly visible = computed(() => {
    const filters = this.filters();
    const query = filters.query.trim();
    const search = this.recentSearch();
    // Until the server answers the current query, fall back to the local metadata match.
    const serverMatches = query && search?.query === query ? search.ids : null;
    const recent = filterSessions(
      this.recent(),
      serverMatches ? { ...filters, query: '' } : filters,
      this.now(),
    ).filter((item) => !serverMatches || serverMatches.has(item.session_id));
    if (!this.includeOlder()) return recent;
    // Older items were already matched server-side (including report content search).
    const merged = new Map(recent.map((item) => [item.session_id, item]));
    for (const item of this.older()) merged.set(item.session_id, item);
    return [...merged.values()].sort(compareSessions(filters.sort));
  });

  destroy(): void {
    this.recentGeneration++;
    this.olderGeneration++;
    this.searchGeneration++;
    this.clearOlderTimer();
    this.clearSearchTimer();
  }

  setFilters(patch: Partial<SessionFilterState>): void {
    this.filters.update((current) => ({ ...current, ...patch }));
    if (rangeReachesHistory(this.filters(), this.now())) this.includeOlder.set(true);
    this.scheduleOlderReload();
    this.scheduleRecentSearch();
  }

  resetFilters(): void {
    this.filters.set(DEFAULT_SESSION_FILTERS);
    this.scheduleOlderReload();
    this.scheduleRecentSearch();
  }

  setIncludeOlder(value: boolean): void {
    this.includeOlder.set(value);
    if (value) {
      void this.loadOlder();
    } else {
      this.olderGeneration++;
      this.older.set([]);
      this.olderTotal.set(null);
      this.olderError.set(null);
    }
  }

  async loadRecent(quiet = false): Promise<void> {
    const generation = ++this.recentGeneration;
    if (!quiet) this.loading.set(true);
    this.error.set(null);
    const now = new Date();
    try {
      const items = await this.fetchAll({
        date_field: 'created',
        date_from: cacheCutoff(now).toISOString(),
        sort: 'created_desc',
      });
      if (generation !== this.recentGeneration) return;
      this.now.set(now);
      this.recent.set(items);
      this.loaded.set(true);
    } catch (error) {
      if (generation === this.recentGeneration) {
        this.error.set(formatUnknownError(error, 'Failed to load clinical sessions.'));
      }
    } finally {
      if (generation === this.recentGeneration) this.loading.set(false);
    }
    if (this.filters().query.trim()) void this.searchRecent();
    if (this.includeOlder()) void this.loadOlder();
  }

  /** Fetches only sessions changed since the newest cached update; returns the changed items. */
  async syncChanges(): Promise<InspectionSessionItem[]> {
    const generation = this.recentGeneration;
    const since = this.recent().reduce(
      (latest, item) => (item.updated_at > latest ? item.updated_at : latest),
      cacheCutoff(new Date()).toISOString(),
    );
    try {
      const changed = await this.fetchAll({ date_field: 'updated', date_from: since, sort: 'updated_desc' });
      if (generation !== this.recentGeneration) return [];
      const cutoff = cacheCutoff(new Date()).toISOString();
      const known = new Map(this.recent().map((item) => [item.session_id, item]));
      const changedItems = changed.filter((item) => {
        const previous = known.get(item.session_id);
        return !previous || previous.status !== item.status || previous.updated_at !== item.updated_at;
      });
      this.recent.update((items) => this.mergeById(items, changedItems.filter((item) => item.created_at >= cutoff)));
      const olderIds = new Set(this.older().map((item) => item.session_id));
      const olderChanges = changedItems.filter((item) => olderIds.has(item.session_id));
      if (olderChanges.length) this.older.update((items) => this.mergeById(items, olderChanges));
      return changedItems;
    } catch {
      // A failed background sync keeps the cache; the next tick or a manual refresh retries.
      return [];
    }
  }

  async loadOlder(append = false): Promise<void> {
    if (!this.includeOlder()) return;
    this.clearOlderTimer();
    const generation = ++this.olderGeneration;
    const now = this.now();
    const offset = append ? this.older().length : 0;
    this.olderLoading.set(true);
    this.olderError.set(null);
    try {
      const payload = await fetchInspectionSessions({
        ...toServerQuery(this.filters(), now),
        created_before: cacheCutoff(now).toISOString(),
        offset,
        limit: PAGE_SIZE,
      });
      if (generation !== this.olderGeneration) return;
      this.older.set(append ? this.mergeById(this.older(), payload.items) : payload.items);
      this.olderTotal.set(payload.total);
    } catch (error) {
      if (generation === this.olderGeneration) {
        this.olderError.set(formatUnknownError(error, 'Failed to load older sessions.'));
      }
    } finally {
      if (generation === this.olderGeneration) this.olderLoading.set(false);
    }
  }

  /** Resolves the current free-text query server-side within the cached window. */
  private async searchRecent(): Promise<void> {
    this.clearSearchTimer();
    const query = this.filters().query.trim();
    const generation = ++this.searchGeneration;
    if (!query) {
      this.recentSearch.set(null);
      return;
    }
    try {
      const items = await this.fetchAll({
        date_field: 'created',
        date_from: cacheCutoff(this.now()).toISOString(),
        sort: 'created_desc',
        search: query,
      });
      if (generation !== this.searchGeneration) return;
      this.recentSearch.set({ query, ids: new Set(items.map((item) => item.session_id)) });
    } catch {
      // Keep the local metadata match until the next query change or reload.
    }
  }

  remove(sessionId: number): void {
    this.recent.update((items) => items.filter((item) => item.session_id !== sessionId));
    this.older.update((items) => items.filter((item) => item.session_id !== sessionId));
  }

  patch(sessionId: number, changes: Partial<InspectionSessionItem>): void {
    const apply = (items: InspectionSessionItem[]): InspectionSessionItem[] => items.map((item) => (
      item.session_id === sessionId ? { ...item, ...changes } : item
    ));
    this.recent.update(apply);
    this.older.update(apply);
  }

  private async fetchAll(
    query: {
      date_field: 'created' | 'updated';
      date_from: string;
      sort: 'created_desc' | 'updated_desc';
      search?: string;
    },
  ): Promise<InspectionSessionItem[]> {
    const collected = new Map<number, InspectionSessionItem>();
    let offset = 0;
    for (;;) {
      const page = await fetchInspectionSessions({ ...query, offset, limit: PAGE_SIZE });
      for (const item of page.items) collected.set(item.session_id, item);
      offset += page.items.length;
      if (!page.items.length || offset >= page.total) break;
    }
    return [...collected.values()];
  }

  private mergeById(
    current: InspectionSessionItem[],
    incoming: InspectionSessionItem[],
  ): InspectionSessionItem[] {
    const merged = new Map(current.map((item) => [item.session_id, item]));
    for (const item of incoming) merged.set(item.session_id, item);
    return [...merged.values()];
  }

  private scheduleOlderReload(): void {
    if (!this.includeOlder()) return;
    this.clearOlderTimer();
    this.olderTimer = globalThis.setTimeout(() => {
      this.olderTimer = null;
      void this.loadOlder();
    }, OLDER_QUERY_DEBOUNCE_MS);
  }

  private scheduleRecentSearch(): void {
    this.clearSearchTimer();
    if (!this.filters().query.trim()) {
      this.searchGeneration++;
      this.recentSearch.set(null);
      return;
    }
    if (this.recentSearch()?.query === this.filters().query.trim()) return;
    this.searchTimer = globalThis.setTimeout(() => {
      this.searchTimer = null;
      void this.searchRecent();
    }, OLDER_QUERY_DEBOUNCE_MS);
  }

  private clearSearchTimer(): void {
    if (this.searchTimer !== null) {
      globalThis.clearTimeout(this.searchTimer);
      this.searchTimer = null;
    }
  }

  private clearOlderTimer(): void {
    if (this.olderTimer !== null) {
      globalThis.clearTimeout(this.olderTimer);
      this.olderTimer = null;
    }
  }
}
