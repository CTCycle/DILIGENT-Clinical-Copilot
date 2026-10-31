// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

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
            <span>Refresh mode</span>
            <select
              [ngModel]="booleanValue('force_full', false) ? 'full' : 'incremental'"
              (ngModelChange)="updateBoolean('force_full', $event === 'full')"
              [disabled]="disabled"
            >
              <option value="incremental">Skip unchanged releases</option>
              <option value="full">Force full re-import</option>
            </select>
            <small>RxNorm Prescribable Content and curated aliases; a source is replaced only after a complete download.</small>
          </label>
        </div>
      }

      @if (target === 'terminology') {
        <div class="inspection-update-terminology">
          <div class="inspection-update-field">
            <span id="terminology-refresh-mode">Refresh mode</span>
            <div class="inspection-segmented" role="radiogroup" aria-labelledby="terminology-refresh-mode">
              <label [class.is-selected]="!booleanValue('force_full', false)">
                <input
                  type="radio"
                  name="terminology-refresh-mode"
                  [checked]="!booleanValue('force_full', false)"
                  [disabled]="disabled"
                  (change)="updateBoolean('force_full', false)"
                />
                Skip unchanged
              </label>
              <label [class.is-selected]="booleanValue('force_full', false)">
                <input
                  type="radio"
                  name="terminology-refresh-mode"
                  [checked]="booleanValue('force_full', false)"
                  [disabled]="disabled"
                  (change)="updateBoolean('force_full', true)"
                />
                Full re-import
              </label>
            </div>
            <small>Unchanged source versions are skipped unless a full re-import is chosen.</small>
          </div>

          <fieldset class="inspection-update-sources" [disabled]="disabled">
            <legend>Sources to update</legend>
            <p class="inspection-update-source-fixed">
              <strong>RxNorm and curated aliases</strong>
              <small>Always included. Public domain (NLM) and project-maintained.</small>
            </p>
            @for (source of terminologySources; track source.key) {
              <label class="inspection-update-source">
                <input
                  type="checkbox"
                  [checked]="booleanValue(source.key, true)"
                  (change)="updateBoolean(source.key, $any($event.target).checked)"
                />
                <span>
                  <strong>{{ source.label }}</strong>
                  <small>{{ source.hint }}</small>
                </span>
              </label>
            }
          </fieldset>
          <p class="inspection-update-summary" aria-live="polite">Will update: {{ terminologySummary() }}</p>
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

      @if (target === 'dilirank') {
        <div class="inspection-update-field-grid">
          <label class="inspection-update-field">
            <span>Source handling</span>
            <select
              [ngModel]="booleanValue('redownload', false) ? 'fresh' : 'conditional'"
              (ngModelChange)="updateBoolean('redownload', $event === 'fresh')"
              [disabled]="disabled"
            >
              <option value="conditional">Reuse when FDA source is unchanged</option>
              <option value="fresh">Force fresh FDA download</option>
            </select>
            <small>Validated downloads replace the local cache only after schema checks succeed.</small>
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
      gap: var(--space-md);
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
      min-height: var(--control-height-sm);
      box-sizing: border-box;
      padding: 0 var(--space-sm);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      background: var(--color-surface);
      color: var(--color-text-primary);
      font: inherit;
      font-size: var(--font-sm);
      line-height: 1.2;
    }

    .inspection-update-field input:focus-visible,
    .inspection-update-field select:focus-visible {
      outline: var(--focus-ring);
      outline-offset: var(--focus-ring-offset);
    }

    .inspection-update-field small {
      margin: 0;
      color: var(--color-text-muted);
      font-size: var(--font-xs);
      line-height: 1.4;
    }

    .inspection-update-terminology {
      display: grid;
      gap: var(--space-md);
    }

    .inspection-segmented {
      display: inline-flex;
      width: fit-content;
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      overflow: hidden;
    }

    .inspection-segmented label {
      position: relative;
      padding: 6px var(--space-md);
      background: var(--color-surface);
      color: var(--color-text-secondary);
      font-size: var(--font-sm);
      font-weight: 600;
      cursor: pointer;
    }

    .inspection-segmented label + label {
      border-left: 1px solid var(--color-border);
    }

    .inspection-segmented label.is-selected {
      background: var(--color-brand-bg);
      color: var(--color-brand);
    }

    .inspection-segmented label:has(input:focus-visible) {
      outline: var(--focus-ring);
      outline-offset: -2px;
    }

    .inspection-segmented input {
      position: absolute;
      opacity: 0;
      inset: 0;
      margin: 0;
      cursor: pointer;
    }

    .inspection-update-sources {
      display: grid;
      gap: var(--space-sm);
      min-width: 0;
      margin: 0;
      padding: 0;
      border: 0;
    }

    .inspection-update-sources legend {
      margin-bottom: var(--space-xs);
      padding: 0;
      color: var(--color-text-primary);
      font-size: var(--font-sm);
      font-weight: 700;
    }

    .inspection-update-source,
    .inspection-update-source-fixed {
      display: flex;
      align-items: flex-start;
      gap: var(--space-sm);
      margin: 0;
    }

    .inspection-update-source-fixed {
      flex-direction: column;
      gap: 2px;
      padding-left: 24px;
    }

    .inspection-update-source {
      cursor: pointer;
    }

    .inspection-update-source input {
      margin-top: 3px;
      accent-color: var(--color-brand-ui);
    }

    .inspection-update-source span {
      display: grid;
      gap: 2px;
    }

    .inspection-update-source strong,
    .inspection-update-source-fixed strong {
      font-size: var(--font-sm);
    }

    .inspection-update-source small,
    .inspection-update-source-fixed small {
      color: var(--color-text-muted);
      font-size: var(--font-xs);
      line-height: 1.4;
    }

    .inspection-update-summary {
      margin: 0;
      color: var(--color-text-secondary);
      font-size: var(--font-xs);
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
  readonly terminologySources = [
    { key: 'drugcentral_enabled', label: 'DrugCentral', hint: 'International names and identifiers, CC BY-SA 4.0.' },
    { key: 'chembl_enabled', label: 'ChEMBL', hint: 'INN, BAN, JAN and trade names, CC BY-SA 3.0.' },
    { key: 'swissmedic_enabled', label: 'Swissmedic', hint: 'Swiss authorised products, open use with attribution.' },
  ] as const;

  @Input() target: InspectionUpdateTarget | null = null;
  @Input() config: Record<string, unknown> | null = null;
  @Input() disabled = false;
  @Output() readonly configChange = new EventEmitter<InspectionUpdateFieldChange>();

  terminologySummary(): string {
    const names = ['RxNorm', ...this.terminologySources
      .filter((source) => this.booleanValue(source.key, true))
      .map((source) => source.label), 'curated aliases'];
    return names.join(', ');
  }

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
