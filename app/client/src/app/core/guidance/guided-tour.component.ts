import {
  CommonModule,
} from '@angular/common';
import {
  Component,
  HostListener,
  OnDestroy,
  computed,
  effect,
  inject,
  signal,
} from '@angular/core';
import { LucideX } from '@lucide/angular';

import { ActiveGuidedTour, GuidanceTourService } from './guidance-tour.service';

const FOCUSABLE_SELECTOR = [
  'button:not([disabled])',
  'a[href]',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

@Component({
  selector: 'app-guided-tour',
  standalone: true,
  imports: [CommonModule, LucideX],
  templateUrl: './guided-tour.component.html',
  styleUrl: './guided-tour.component.scss',
})
export class GuidedTourComponent implements OnDestroy {
  private readonly tourService = inject(GuidanceTourService);

  readonly active = this.tourService.activeTour;
  readonly currentStep = computed(() => {
    const active = this.active();
    return active ? active.definition.steps[active.stepIndex] ?? null : null;
  });
  readonly targetRect = signal<DOMRect | null>(null);
  readonly cardPosition = signal({ top: 0, left: 0 });

  private previousActiveElement: HTMLElement | null = null;
  private wasActive = false;
  private currentTarget: HTMLElement | null = null;
  private syncToken = 0;
  private positionRefreshTimer: number | null = null;

  private readonly handleCapturedScroll = (): void => {
    this.queuePositionRefresh();
  };

  constructor() {
    document.addEventListener('scroll', this.handleCapturedScroll, true);
    window.addEventListener('resize', this.handleCapturedScroll);
    effect(() => {
      const active = this.active();
      if (!active) {
        this.targetRect.set(null);
        this.currentTarget = null;
        if (this.wasActive) {
          this.wasActive = false;
          const focusTarget = this.previousActiveElement;
          this.previousActiveElement = null;
          if (focusTarget?.isConnected) queueMicrotask(() => focusTarget.focus());
        }
        return;
      }

      if (!this.wasActive) {
        this.wasActive = true;
        this.previousActiveElement = document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
      }
      const token = ++this.syncToken;
      queueMicrotask(() => this.syncStep(active, token));
    });
  }

  ngOnDestroy(): void {
    document.removeEventListener('scroll', this.handleCapturedScroll, true);
    window.removeEventListener('resize', this.handleCapturedScroll);
    if (this.positionRefreshTimer !== null) {
      window.clearTimeout(this.positionRefreshTimer);
      this.positionRefreshTimer = null;
    }
  }

  next(): void {
    this.tourService.next();
  }

  back(): void {
    this.tourService.back();
  }

  skip(): void {
    this.tourService.skip();
  }

  @HostListener('document:keydown', ['$event'])
  handleKeydown(event: KeyboardEvent): void {
    if (!this.active()) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      this.skip();
      return;
    }
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      this.next();
      return;
    }
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      this.back();
      return;
    }
    if (event.key === 'Tab') {
      this.trapFocus(event);
    }
  }

  private syncStep(active: ActiveGuidedTour, token: number): void {
    const step = active.definition.steps[active.stepIndex];
    const target = step ? document.querySelector<HTMLElement>(step.target) : null;
    this.currentTarget = target;
    if (target) {
      const reducedMotion = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
      target.scrollIntoView({ block: 'center', inline: 'nearest', behavior: reducedMotion ? 'auto' : 'smooth' });
      window.setTimeout(() => {
        if (token !== this.syncToken) return;
        this.refreshTargetPosition();
        this.focusDialog();
      }, reducedMotion ? 0 : 160);
      return;
    }

    this.targetRect.set(null);
    this.positionCard(null, step?.preferredPlacement);
    this.focusDialog();
  }

  private queuePositionRefresh(): void {
    if (!this.active() || this.positionRefreshTimer !== null) return;
    this.positionRefreshTimer = window.setTimeout(() => {
      this.positionRefreshTimer = null;
      this.refreshTargetPosition();
    }, 0);
  }

  private refreshTargetPosition(): void {
    const active = this.active();
    const step = active?.definition.steps[active.stepIndex];
    const target = step ? document.querySelector<HTMLElement>(step.target) : null;
    this.currentTarget = target;
    if (!target) {
      this.targetRect.set(null);
      this.positionCard(null, step?.preferredPlacement);
      return;
    }

    const rect = target.getBoundingClientRect();
    this.targetRect.set(rect);
    this.positionCard(rect, step?.preferredPlacement);
  }

  private focusDialog(): void {
    queueMicrotask(() => {
      const dialog = document.querySelector<HTMLElement>('.guidance-tour-dialog');
      const initial = dialog?.querySelector<HTMLElement>('[data-tour-initial-focus]');
      (initial ?? dialog)?.focus();
    });
  }

  private positionCard(target: DOMRect | null, preferredPlacement = 'bottom'): void {
    const width = Math.min(390, Math.max(280, window.innerWidth - 24));
    const height = document.querySelector<HTMLElement>('.guidance-tour-dialog')?.offsetHeight || 260;
    const margin = 12;
    const gap = 16;
    if (!target) {
      this.cardPosition.set({
        top: Math.max(margin, (window.innerHeight - height) / 2),
        left: Math.max(margin, (window.innerWidth - width) / 2),
      });
      return;
    }

    const candidates = [preferredPlacement, 'bottom', 'top', 'right', 'left'].filter(
      (value, index, values): value is 'top' | 'bottom' | 'left' | 'right' => values.indexOf(value) === index,
    );
    const fits = (placement: string): { top: number; left: number } | null => {
      if (placement === 'top' && target.top - height - gap >= margin) {
        return { top: target.top - height - gap, left: target.left };
      }
      if (placement === 'right' && target.right + width + gap <= window.innerWidth - margin) {
        return { top: target.top, left: target.right + gap };
      }
      if (placement === 'left' && target.left - width - gap >= margin) {
        return { top: target.top, left: target.left - width - gap };
      }
      if (placement === 'bottom' && target.bottom + height + gap <= window.innerHeight - margin) {
        return { top: target.bottom + gap, left: target.left };
      }
      return null;
    };
    const position = candidates.map((candidate) => fits(candidate)).find((value): value is { top: number; left: number } => value !== null)
      ?? { top: Math.max(margin, window.innerHeight - height - margin), left: Math.max(margin, (window.innerWidth - width) / 2) };
    this.cardPosition.set({
      top: Math.min(Math.max(margin, position.top), Math.max(margin, window.innerHeight - height - margin)),
      left: Math.min(Math.max(margin, position.left), Math.max(margin, window.innerWidth - width - margin)),
    });
  }

  private trapFocus(event: KeyboardEvent): void {
    const dialog = document.querySelector<HTMLElement>('.guidance-tour-dialog');
    if (!dialog) return;
    const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
    if (!focusable.length) {
      event.preventDefault();
      dialog.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
}
