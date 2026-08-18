import { Injectable, signal } from '@angular/core';
import { Router } from '@angular/router';

import { GuidanceStateService } from './guidance-state.service';
import { GuidedTourDefinition } from './guidance.types';

export interface ActiveGuidedTour {
  definition: GuidedTourDefinition;
  stepIndex: number;
}

@Injectable({ providedIn: 'root' })
export class GuidanceTourService {
  readonly activeTour = signal<ActiveGuidedTour | null>(null);

  constructor(
    private readonly router: Router,
    private readonly stateService: GuidanceStateService,
  ) {}

  start(definition: GuidedTourDefinition): void {
    this.stateService.markSeen(definition.id, definition.version);
    if (this.currentPath() === definition.route) {
      this.activeTour.set({ definition, stepIndex: 0 });
      return;
    }

    void this.router.navigateByUrl(definition.route).then(() => {
      queueMicrotask(() => this.activeTour.set({ definition, stepIndex: 0 }));
    });
  }

  next(): void {
    const active = this.activeTour();
    if (!active) return;
    if (active.stepIndex >= active.definition.steps.length - 1) {
      this.stateService.complete(active.definition.id, active.definition.version);
      this.activeTour.set(null);
      return;
    }
    this.activeTour.set({ ...active, stepIndex: active.stepIndex + 1 });
  }

  back(): void {
    const active = this.activeTour();
    if (!active || active.stepIndex === 0) return;
    this.activeTour.set({ ...active, stepIndex: active.stepIndex - 1 });
  }

  skip(): void {
    const active = this.activeTour();
    if (active) {
      this.stateService.skip(active.definition.id, active.definition.version);
    }
    this.activeTour.set(null);
  }

  restart(definition: GuidedTourDefinition): void {
    this.stateService.restart(definition.id, definition.version);
    this.start(definition);
  }

  private currentPath(): string {
    return this.router.url.split('?')[0].split('#')[0] || '/';
  }
}
