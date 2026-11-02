// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, input } from '@angular/core';

import { ClinicalSessionError, ClinicalSessionStatus } from '../../core/models/types';
import { ANALYSIS_PHASES, resolvePhaseIndex } from '../../core/utils/analysis-phases';
import { isActiveSessionStatus, isInterruptedSession, sessionStatusTone } from '../../core/utils/session-status';

const BACKEND_STEP_PREFIX = /^Step\s+\d+\s*(?:\/|of)\s*\d+\s*:\s*/i;

/** Compact lifecycle indicator: a spinner only while the session is queued or running. */
@Component({
  selector: 'app-report-activity',
  templateUrl: './report-activity.component.html',
  styleUrl: './report-activity.component.scss',
})
export class ReportActivityComponent {
  readonly status = input.required<ClinicalSessionStatus>();
  readonly error = input<ClinicalSessionError | null>(null);
  readonly stage = input<string | null>(null);
  readonly message = input<string | null>(null);
  readonly queuePosition = input<number | null>(null);
  readonly stopRequested = input(false);

  readonly active = computed(() => isActiveSessionStatus(this.status()));
  readonly tone = computed(() => (isInterruptedSession(this.status(), this.error()) ? 'warning' : sessionStatusTone(this.status())));
  readonly label = computed(() => {
    const status = this.status();
    if (status === 'queued') {
      const position = this.queuePosition();
      return position === null ? 'Waiting for capacity' : `Waiting for capacity (position ${position})`;
    }
    if (status === 'running') {
      if (this.stopRequested()) return 'Stopping after the current step…';
      const index = resolvePhaseIndex(this.stage());
      return index === null ? 'Starting analysis' : ANALYSIS_PHASES[index];
    }
    if (isInterruptedSession(status, this.error())) return 'Analysis interrupted';
    switch (status) {
      case 'completed':
        return 'Analysis completed';
      case 'failed':
        return 'Analysis failed';
      case 'cancelled':
        return 'Analysis cancelled';
      default:
        return 'Draft — not analysed yet';
    }
  });
  readonly detail = computed(() => {
    const status = this.status();
    if (status === 'running') {
      return this.message()?.replace(BACKEND_STEP_PREFIX, '').trim() || null;
    }
    if (status === 'failed') return this.error()?.message ?? null;
    if (status === 'queued') return 'Starts automatically; you can leave this page.';
    if (status === 'cancelled') return 'Stopped before it finished.';
    return null;
  });
}
