// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, DestroyRef, computed, effect, inject, input, signal } from '@angular/core';

import { JobStatus } from '../../core/models/types';
import {
  ANALYSIS_PHASES,
  formatElapsed,
  resolvePhaseIndex,
  resolveProgressTone,
} from './analysis-progress';

const STALL_THRESHOLD_MS = 600_000;
const TICK_MS = 1000;

@Component({
  selector: 'app-analysis-progress',
  templateUrl: './analysis-progress.component.html',
  styleUrl: './analysis-progress.component.scss',
})
export class AnalysisProgressComponent {
  readonly jobStatus = input<JobStatus | null>(null);
  readonly jobProgress = input(0);
  readonly stage = input<string | null>(null);
  readonly stageMessage = input<string | null>(null);
  readonly startedAtMs = input<number | null>(null);
  readonly lastProgressAtMs = input<number | null>(null);
  readonly errorMessage = input<string | null>(null);

  private readonly now = signal(Date.now());
  private readonly lastPhaseIndex = signal<number | null>(null);

  readonly phaseCount = ANALYSIS_PHASES.length;
  readonly tone = computed(() => resolveProgressTone(this.jobStatus()));
  readonly phaseIndex = this.lastPhaseIndex.asReadonly();
  readonly isIndeterminate = computed(
    () => this.tone() === 'running' && this.jobProgress() <= 0,
  );
  readonly barValue = computed(() =>
    this.tone() === 'completed' ? 100 : Math.min(100, Math.max(0, this.jobProgress())),
  );
  readonly phaseLabel = computed(() => {
    switch (this.tone()) {
      case 'completed':
        return 'Analysis complete';
      case 'failed':
        return 'Analysis failed';
      case 'cancelled':
        return 'Analysis cancelled';
      default: {
        const index = this.phaseIndex();
        return index === null ? 'Starting analysis' : ANALYSIS_PHASES[index];
      }
    }
  });
  readonly stepLabel = computed(() => {
    const index = this.phaseIndex();
    return this.tone() === 'running' && index !== null
      ? `Step ${index + 1} of ${this.phaseCount}`
      : null;
  });
  readonly elapsed = computed(() => {
    const started = this.startedAtMs();
    if (!started) return null;
    const end = this.tone() === 'running' ? this.now() : (this.lastProgressAtMs() ?? this.now());
    return formatElapsed(end - started);
  });
  readonly detail = computed(() => {
    if (this.tone() === 'failed') {
      return this.errorMessage();
    }
    if (this.tone() !== 'running') {
      return null;
    }
    const message = this.stageMessage();
    if (!message) {
      return null;
    }
    return (this.stage() ?? '').includes('extraction')
      ? `${message} Local model extraction can take several minutes.`
      : message;
  });
  readonly stalled = computed(() => {
    const last = this.lastProgressAtMs();
    return this.tone() === 'running'
      && Boolean(this.startedAtMs())
      && last !== null
      && this.now() - last >= STALL_THRESHOLD_MS;
  });

  constructor() {
    effect(() => {
      const index = resolvePhaseIndex(this.stage());
      if (index !== null) {
        this.lastPhaseIndex.set(index);
      }
    });
    const timer = globalThis.setInterval(() => this.now.set(Date.now()), TICK_MS);
    inject(DestroyRef).onDestroy(() => globalThis.clearInterval(timer));
  }
}
