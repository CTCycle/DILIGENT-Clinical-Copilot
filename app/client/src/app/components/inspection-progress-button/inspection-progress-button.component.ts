import { Component, input, output } from '@angular/core';

@Component({
  selector: 'app-inspection-progress-button',
  standalone: true,
  template: `
    <button
      class="btn btn-secondary inspection-mini-btn inspection-progress-button"
      type="button"
      [class.is-running]="inProgress()"
      [style.--inspection-progress-width]="displayProgress() + '%'"
      [attr.aria-busy]="inProgress()"
      [attr.title]="buttonTitle()"
      [disabled]="disabled()"
      (click)="activated.emit()"
    >
      <span class="inspection-progress-button-fill" aria-hidden="true"></span>
      <span class="inspection-progress-button-content">
        <span>{{ buttonLabel() }}</span>
        @if (inProgress()) {
          <span class="inspection-progress-button-percent">{{ normalizedProgress() }}%</span>
        }
      </span>
    </button>
  `,
  styles: `
    :host {
      display: contents;
    }

    .inspection-progress-button {
      position: relative;
      overflow: hidden;
      isolation: isolate;
      min-width: 9.5rem;
      border-color: rgba(22, 163, 74, 0.28);
      transition:
        border-color 160ms ease,
        color 160ms ease,
        background-color 160ms ease,
        box-shadow 160ms ease;
    }

    .inspection-progress-button.is-running {
      color: #14532d;
      border-color: rgba(22, 163, 74, 0.45);
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.18);
    }

    .inspection-progress-button-fill {
      position: absolute;
      inset: 0;
      width: var(--inspection-progress-width, 0%);
      background: linear-gradient(
        90deg,
        rgba(187, 247, 208, 0.5) 0%,
        rgba(134, 239, 172, 0.82) 58%,
        rgba(34, 197, 94, 0.94) 100%
      );
      transition: width 320ms ease;
      z-index: 0;
    }

    .inspection-progress-button-content {
      position: relative;
      z-index: 1;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }

    .inspection-progress-button-percent {
      font-variant-numeric: tabular-nums;
      color: rgba(20, 83, 45, 0.92);
    }
  `,
})
export class InspectionProgressButtonComponent {
  readonly buttonLabel = input.required<string>();
  readonly inProgress = input(false);
  readonly progress = input(0);
  readonly progressMessage = input('');
  readonly disabled = input(false);

  readonly activated = output<void>();

  normalizedProgress(): number {
    const value = Number(this.progress());
    if (!Number.isFinite(value)) {
      return 0;
    }
    return Math.min(100, Math.max(0, Math.round(value)));
  }

  displayProgress(): number {
    return this.inProgress() ? this.normalizedProgress() : 0;
  }

  buttonTitle(): string {
    if (!this.inProgress()) {
      return this.buttonLabel();
    }
    return this.progressMessage().trim() || this.buttonLabel();
  }
}
