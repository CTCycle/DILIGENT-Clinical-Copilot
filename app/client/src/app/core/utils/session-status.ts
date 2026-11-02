// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ClinicalSessionError, ClinicalSessionStatus } from '../models/types';

export type SessionStatusTone = 'neutral' | 'info' | 'warning' | 'success' | 'error';

export const SESSION_STATUSES: readonly ClinicalSessionStatus[] = [
  'draft',
  'queued',
  'running',
  'completed',
  'cancelled',
  'failed',
];

const SESSION_STATUS_LABELS: Record<ClinicalSessionStatus, string> = {
  draft: 'Draft',
  queued: 'Queued',
  running: 'Running',
  completed: 'Completed',
  cancelled: 'Cancelled',
  failed: 'Failed',
};

const SESSION_STATUS_TONES: Record<ClinicalSessionStatus, SessionStatusTone> = {
  draft: 'neutral',
  queued: 'warning',
  running: 'info',
  completed: 'success',
  cancelled: 'neutral',
  failed: 'error',
};

export function sessionStatusLabel(status: ClinicalSessionStatus): string {
  return SESSION_STATUS_LABELS[status] ?? status;
}

export function sessionStatusTone(status: ClinicalSessionStatus): SessionStatusTone {
  return SESSION_STATUS_TONES[status] ?? 'neutral';
}

export function isActiveSessionStatus(status: ClinicalSessionStatus): boolean {
  return status === 'queued' || status === 'running';
}

/** Interruption is a failed session whose reason code is `interrupted`; it is not a separate status. */
export function isInterruptedSession(status: ClinicalSessionStatus, error: ClinicalSessionError | null | undefined): boolean {
  return status === 'failed' && error?.code === 'interrupted';
}
