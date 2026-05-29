import { CommonModule } from '@angular/common';
import { Component, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-inspection-catalog-toolbar',
  standalone: true,
  imports: [CommonModule, FormsModule],
  styles: [
    `
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
  ],
  template: `
    <div class="inspection-widget-header">
      <div class="inspection-controls inspection-controls-half inspection-controls-knowledge">
        <input
          type="search"
          class="inspection-search"
          [placeholder]="searchPlaceholder()"
          [ngModel]="searchValue()"
          (ngModelChange)="searchChange.emit($event)"
          [attr.aria-label]="searchAriaLabel()"
        />
      </div>
      <div class="inspection-widget-header-actions inspection-header-actions-fixed">
        <ng-content select="[inspectionToolbarBeforeActions]"></ng-content>
        <button
          class="btn btn-secondary inspection-mini-btn inspection-progress-button"
          type="button"
          [class.is-running]="updateInProgress()"
          [style.--inspection-progress-width]="displayProgress() + '%'"
          [attr.aria-busy]="updateInProgress()"
          [attr.title]="updateButtonTitle()"
          (click)="updateClick.emit()"
        >
          <span class="inspection-progress-button-fill" aria-hidden="true"></span>
          <span class="inspection-progress-button-content">
            <span>{{ updateButtonLabel() }}</span>
            @if (updateInProgress()) {
              <span class="inspection-progress-button-percent">{{ normalizedProgress() }}%</span>
            }
          </span>
        </button>
      </div>
    </div>
  `,
})
export class InspectionCatalogToolbarComponent {
  readonly searchPlaceholder = input.required<string>();
  readonly searchAriaLabel = input.required<string>();
  readonly searchValue = input.required<string>();
  readonly updateButtonLabel = input.required<string>();
  readonly updateInProgress = input(false);
  readonly updateProgress = input(0);
  readonly updateProgressMessage = input('');

  readonly searchChange = output<string>();
  readonly updateClick = output<void>();

  normalizedProgress(): number {
    const value = Number(this.updateProgress());
    if (!Number.isFinite(value)) {
      return 0;
    }
    return Math.min(100, Math.max(0, Math.round(value)));
  }

  displayProgress(): number {
    return this.updateInProgress() ? this.normalizedProgress() : 0;
  }

  updateButtonTitle(): string {
    if (!this.updateInProgress()) {
      return this.updateButtonLabel();
    }
    return this.updateProgressMessage().trim() || this.updateButtonLabel();
  }
}
