import { Injectable, signal } from '@angular/core';

import {
  GuidanceId,
  GuidanceStateEntry,
  GuidanceStatus,
  PersistedGuidanceState,
} from './guidance.types';

export const GUIDANCE_STORAGE_KEY = 'diligent-guidance-state-v1';
const GUIDANCE_SCHEMA_VERSION = 1 as const;
const GUIDANCE_STATUSES: readonly GuidanceStatus[] = ['seen', 'dismissed', 'skipped', 'completed', 'restarted'];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isGuidanceStatus(value: unknown): value is GuidanceStatus {
  return typeof value === 'string' && GUIDANCE_STATUSES.includes(value as GuidanceStatus);
}

function emptyState(): PersistedGuidanceState {
  return { schemaVersion: GUIDANCE_SCHEMA_VERSION, entries: {} };
}

function parseState(raw: string | null): PersistedGuidanceState {
  if (!raw) return emptyState();
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || parsed['schemaVersion'] !== GUIDANCE_SCHEMA_VERSION || !isRecord(parsed['entries'])) {
      return emptyState();
    }
    const rawEntries = parsed['entries'];
    const entries: Partial<Record<GuidanceId, GuidanceStateEntry>> = {};
    for (const [id, value] of Object.entries(rawEntries)) {
      if (
        isRecord(value)
        && typeof value['version'] === 'number'
        && Number.isFinite(value['version'])
        && isGuidanceStatus(value['status'])
        && typeof value['updatedAt'] === 'string'
      ) {
        entries[id as GuidanceId] = {
          version: value['version'],
          status: value['status'],
          updatedAt: value['updatedAt'],
          ...(typeof value['restartCount'] === 'number' ? { restartCount: value['restartCount'] } : {}),
          ...(typeof value['lastRestartedAt'] === 'string' ? { lastRestartedAt: value['lastRestartedAt'] } : {}),
        };
      }
    }
    return { schemaVersion: GUIDANCE_SCHEMA_VERSION, entries };
  } catch {
    return emptyState();
  }
}

@Injectable({ providedIn: 'root' })
export class GuidanceStateService {
  readonly state = signal<PersistedGuidanceState>(this.readState());

  status(id: GuidanceId): GuidanceStateEntry | null {
    return this.state().entries[id] ?? null;
  }

  shouldShow(id: GuidanceId, version: number): boolean {
    const entry = this.status(id);
    return !entry || entry.version < version;
  }

  markSeen(id: GuidanceId, version: number): void {
    this.setStatus(id, version, 'seen');
  }

  dismiss(id: GuidanceId, version: number): void {
    this.setStatus(id, version, 'dismissed');
  }

  skip(id: GuidanceId, version: number): void {
    this.setStatus(id, version, 'skipped');
  }

  complete(id: GuidanceId, version: number): void {
    this.setStatus(id, version, 'completed');
  }

  restart(id: GuidanceId, version: number): void {
    const previous = this.status(id);
    const restartedAt = new Date().toISOString();
    const entries = {
      ...this.state().entries,
      [id]: {
        version,
        status: 'restarted' as const,
        updatedAt: restartedAt,
        restartCount: (previous?.restartCount ?? 0) + 1,
        lastRestartedAt: restartedAt,
      },
    };
    this.persist({ schemaVersion: GUIDANCE_SCHEMA_VERSION, entries });
  }

  private setStatus(id: GuidanceId, version: number, status: GuidanceStatus): void {
    const previous = this.status(id);
    const entries = {
      ...this.state().entries,
      [id]: {
        ...previous,
        version,
        status,
        updatedAt: new Date().toISOString(),
      },
    };
    this.persist({ schemaVersion: GUIDANCE_SCHEMA_VERSION, entries });
  }

  private readState(): PersistedGuidanceState {
    try {
      return parseState(globalThis.localStorage?.getItem(GUIDANCE_STORAGE_KEY) ?? null);
    } catch {
      return emptyState();
    }
  }

  private persist(nextState: PersistedGuidanceState): void {
    this.state.set(nextState);
    try {
      globalThis.localStorage?.setItem(GUIDANCE_STORAGE_KEY, JSON.stringify(nextState));
    } catch {
      // Guidance is optional. The in-memory signal remains useful when storage is unavailable.
    }
  }
}
