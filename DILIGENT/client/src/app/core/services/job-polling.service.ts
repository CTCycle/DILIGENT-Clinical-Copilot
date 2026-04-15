import { Injectable } from '@angular/core';

type PollStep = () => Promise<boolean>;
type PollCancelled = () => boolean;

type PollOptions = {
  intervalMs: number;
  pollStep: PollStep;
  isCancelled?: PollCancelled;
};

@Injectable({ providedIn: 'root' })
export class JobPollingService {
  async run(options: PollOptions): Promise<void> {
    const { intervalMs, pollStep, isCancelled } = options;
    const safeIntervalMs = Math.max(250, intervalMs);
    while (!(isCancelled?.() ?? false)) {
      const shouldContinue = await pollStep();
      if (!shouldContinue) {
        return;
      }
      await new Promise((resolve) => globalThis.setTimeout(resolve, safeIntervalMs));
    }
  }
}
