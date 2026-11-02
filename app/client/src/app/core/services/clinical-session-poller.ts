// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable, inject } from '@angular/core';

import { ClinicalSessionState } from '../models/types';
import { resolvePollIntervalMs } from './clinical-api';
import { fetchClinicalSessionStatus } from './clinical-session-lifecycle-api';
import { ApiHttpError } from './http-api';
import { isActiveSessionStatus } from '../utils/session-status';
import { JobPollingService } from './job-polling.service';

const MAX_CONSECUTIVE_ERRORS = 5;
const MAX_BACKOFF_MS = 30_000;

/** `retrying`: a request failed and another attempt is scheduled; `disconnected`: retries exhausted, polling stopped. */
export type PollConnection = 'live' | 'retrying' | 'disconnected';

/** Polls one session's status endpoint while it is queued or running. */
@Injectable({ providedIn: 'root' })
export class ClinicalSessionPoller {
  private readonly polling = inject(JobPollingService);

  /**
   * Emits snapshots with a strictly newer `version` than any already seen.
   * Returns a function that stops the poll; the poll also stops once the session
   * is no longer queued or running, a 404, or repeated request failures (reported through `onError`).
   * Failed requests are retried with bounded exponential backoff; `onConnection` reports the transitions.
   */
  watch(
    sessionId: number,
    onState: (state: ClinicalSessionState) => void,
    onError?: (error: unknown) => void,
    onConnection?: (connection: PollConnection) => void,
  ): () => void {
    let cancelled = false;
    let latestVersion = -1;
    let intervalMs = 1000;
    let baseIntervalMs = intervalMs;
    let consecutiveErrors = 0;

    void this.polling.run({
      intervalMs: () => intervalMs,
      isCancelled: () => cancelled,
      pollStep: async () => {
        try {
          const state = await fetchClinicalSessionStatus(sessionId);
          if (cancelled) return false;
          if (consecutiveErrors > 0) onConnection?.('live');
          consecutiveErrors = 0;
          baseIntervalMs = resolvePollIntervalMs(state.poll_interval);
          intervalMs = baseIntervalMs;
          if (state.version > latestVersion) {
            latestVersion = state.version;
            onState(state);
          }
          return isActiveSessionStatus(state.status);
        } catch (error) {
          if (cancelled) return false;
          consecutiveErrors += 1;
          const notFound = error instanceof ApiHttpError && error.statusCode === 404;
          if (notFound || consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            onError?.(error);
            onConnection?.('disconnected');
            return false;
          }
          intervalMs = Math.min(baseIntervalMs * 2 ** consecutiveErrors, MAX_BACKOFF_MS);
          onConnection?.('retrying');
          return true;
        }
      },
    });

    return () => {
      cancelled = true;
    };
  }
}
