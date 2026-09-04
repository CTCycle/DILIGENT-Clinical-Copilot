import { Component, EventEmitter, Input, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { InspectionUpdateTarget } from '../../../core/models/inspection-types';

export type InspectionUpdateFieldChange = {
  key: string;
  value: unknown;
};

@Component({
  selector: 'app-inspection-update-controls',
  standalone: true,
  imports: [FormsModule],
  template: `
    <section class="inspection-update-parameters" aria-label="Knowledge update parameters">
      @if (target === 'rxnav') {
        <div class="inspection-update-field-grid">
          <label class="inspection-update-field">
            <span>Request timeout</span>
            <input
              type="number"
              min="1"
              max="120"
              step="0.5"
              [ngModel]="numberValue('rxnav_request_timeout', 30)"
              (ngModelChange)="updateNumber('rxnav_request_timeout', $event, 1, 120, false)"
              [disabled]="disabled"
            />
            <small>Seconds, 1 to 120.</small>
          </label>

          <label class="inspection-update-field">
            <span>Max concurrency</span>
            <input
              type="number"
              min="1"
              max="64"
              step="1"
              [ngModel]="numberValue('rxnav_max_concurrency', 4)"
              (ngModelChange)="updateNumber('rxnav_max_concurrency', $event, 1, 64, true)"
              [disabled]="disabled"
            />
            <small>Parallel RxNav requests, 1 to 64.</small>
          </label>
        </div>
      }

      @if (target === 'livertox') {
        <div class="inspection-update-field-grid">
          <label class="inspection-update-field">
            <span>Monograph workers</span>
            <input
              type="number"
              min="1"
              max="32"
              step="1"
              [ngModel]="numberValue('livertox_monograph_max_workers', 4)"
              (ngModelChange)="updateNumber('livertox_monograph_max_workers', $event, 1, 32, true)"
              [disabled]="disabled"
            />
            <small>Parallel parsing workers, 1 to 32.</small>
          </label>

          <label class="inspection-update-field inspection-update-field-wide">
            <span>Archive name</span>
            <input
              type="text"
              maxlength="255"
              [ngModel]="stringValue('livertox_archive')"
              (ngModelChange)="updateString('livertox_archive', $event)"
              [disabled]="disabled"
              autocomplete="off"
            />
            <small>Archive used for this update.</small>
          </label>

          <label class="inspection-update-field">
            <span>Archive handling</span>
            <select
              [ngModel]="booleanValue('redownload', false) ? 'fresh' : 'reuse'"
              (ngModelChange)="updateBoolean('redownload', $event === 'fresh')"
              [disabled]="disabled"
            >
              <option value="reuse">Reuse local archive</option>
              <option value="fresh">Download fresh archive</option>
            </select>
            <small>Download a fresh source only when required.</small>
          </label>
        </div>
      }
    </section>
  `,
  styles: `
    :host {
      display: block;
    }

    .inspection-update-parameters {
      display: grid;
      gap: var(--space-lg);
    }

    .inspection-update-field-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-md);
    }

    .inspection-update-field {
      display: grid;
      align-content: start;
      gap: var(--space-xs);
      min-width: 0;
    }

    .inspection-update-field-wide {
      grid-column: 1 / -1;
    }

    .inspection-update-field > span {
      color: var(--color-text-primary);
      font-size: var(--font-sm);
      font-weight: 700;
    }

    .inspection-update-field input,
    .inspection-update-field select {
      width: 100%;
      min-height: var(--control-height-md);
      padding: 0 var(--space-md);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      background: var(--color-surface);
      color: var(--color-text-primary);
      font: inherit;
    }

    .inspection-update-field small {
      margin: 0;
      color: var(--color-text-muted);
      font-size: var(--font-xs);
      line-height: 1.4;
    }

    @media (max-width: 760px) {
      .inspection-update-field-grid {
        grid-template-columns: 1fr;
      }

      .inspection-update-field-wide {
        grid-column: auto;
      }
    }
  `,
})
export class InspectionUpdateControlsComponent {
  @Input() target: InspectionUpdateTarget | null = null;
  @Input() config: Record<string, unknown> | null = null;
  @Input() disabled = false;
  @Output() readonly configChange = new EventEmitter<InspectionUpdateFieldChange>();

  numberValue(key: string, fallback: number): number {
    const value = this.config?.[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  }

  stringValue(key: string): string {
    const value = this.config?.[key];
    return typeof value === 'string' ? value : '';
  }

  booleanValue(key: string, fallback: boolean): boolean {
    const value = this.config?.[key];
    return typeof value === 'boolean' ? value : fallback;
  }

  updateNumber(key: string, value: string | number, min: number, max: number, integer: boolean): void {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return;
    }
    const bounded = Math.min(max, Math.max(min, numeric));
    this.configChange.emit({ key, value: integer ? Math.round(bounded) : bounded });
  }

  updateString(key: string, value: string): void {
    this.configChange.emit({ key, value });
  }

  updateBoolean(key: string, value: boolean): void {
    this.configChange.emit({ key, value });
  }
}
