// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { TerminologyResolutionStatus } from '../models/inspection-types';

export type TerminologyStatusTone = 'is-successful' | 'is-warning' | 'is-failed' | 'is-neutral';

const STATUS_LABELS: Record<TerminologyResolutionStatus, string> = {
  verified: 'Verified',
  probable: 'Probable — review',
  ambiguous: 'Ambiguous',
  unresolved: 'Unresolved',
};

const STATUS_TONES: Record<TerminologyResolutionStatus, TerminologyStatusTone> = {
  verified: 'is-successful',
  probable: 'is-warning',
  ambiguous: 'is-warning',
  unresolved: 'is-failed',
};

function isTerminologyStatus(value: unknown): value is TerminologyResolutionStatus {
  return typeof value === 'string' && value in STATUS_LABELS;
}

export function terminologyStatusLabel(status: unknown): string {
  return isTerminologyStatus(status) ? STATUS_LABELS[status] : 'Not resolved';
}

export function terminologyStatusTone(status: unknown): TerminologyStatusTone {
  return isTerminologyStatus(status) ? STATUS_TONES[status] : 'is-neutral';
}
