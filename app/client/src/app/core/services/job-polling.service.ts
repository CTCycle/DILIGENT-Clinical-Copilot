// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable } from '@angular/core';

type PollStep = () => Promise<boolean>;
type PollCancelled = () => boolean;

type PollOptions = {
  intervalMs: number | (() => number);
  pollStep: PollStep;
  isCancelled?: PollCancelled;
};

@Injectable({ providedIn: 'root' })
export class JobPollingService {
  async run(options: PollOptions): Promise<void> {
    const { intervalMs, pollStep, isCancelled } = options;
    while (!(isCancelled?.() ?? false)) {
      const shouldContinue = await pollStep();
      if (!shouldContinue) {
        return;
      }
      const delayMs = Math.max(250, typeof intervalMs === 'function' ? intervalMs() : intervalMs);
      await new Promise((resolve) => globalThis.setTimeout(resolve, delayMs));
    }
  }
}
