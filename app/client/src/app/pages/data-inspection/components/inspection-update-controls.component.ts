import { Component, EventEmitter, Input, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { InspectionUpdateTarget } from '../../../core/models/inspection-types';
import { isRecord } from '../../../core/utils';

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
            <small>Use a fresh source only when required.</small>
          </label>
        </div>
      }

      <details class="inspection-update-advanced">
        <summary>Advanced JSON</summary>
        <p>Edit the same request payload directly when needed.</p>
        <label class="inspection-update-json-label" for="update-overrides">JSON overrides</label>
        <textarea
          id="update-overrides"
          class="field-textarea inspection-update-overrides"
          [ngModel]="configText"
          (ngModelChange)="setJsonText($event)"
          [disabled]="disabled"
          spellcheck="false"
        ></textarea>
        @if (jsonError(); as error) {
          <p class="inspection-update-json-error" role="alert">{{ error }}</p>
        }
      </details>
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

    .inspection-update-field > span,
    .inspection-update-json-label {
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

    .inspection-update-field small,
    .inspection-update-advanced > p {
      margin: 0;
      color: var(--color-text-muted);
      font-size: var(--font-xs);
      line-height: 1.4;
    }

    .inspection-update-advanced {
      padding-top: var(--space-md);
      border-top: 1px solid var(--color-divider);
    }

    .inspection-update-advanced summary {
      width: fit-content;
      color: var(--color-text-secondary);
      font-size: var(--font-sm);
      font-weight: 700;
      cursor: pointer;
    }

    .inspection-update-advanced > p {
      margin: var(--space-xs) 0 var(--space-sm);
    }

    .inspection-update-json-label {
      display: block;
      margin-bottom: var(--space-xs);
    }

    .inspection-update-overrides {
      width: 100%;
      min-height: 180px;
      font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      font-size: var(--font-sm);
    }

    .inspection-update-json-error {
      margin: var(--space-xs) 0 0;
      color: var(--color-status-error-text);
      font-size: var(--font-xs);
      font-weight: 650;
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
  @Input() configText = '{}';
  @Input() disabled = false;
  @Output() readonly configTextChange = new EventEmitter<string>();

  numberValue(key: string, fallback: number): number {
    const value = this.currentValue(key);
    return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  }

  stringValue(key: string): string {
    const value = this.currentValue(key);
    return typeof value === 'string' ? value : '';
  }

  booleanValue(key: string, fallback: boolean): boolean {
    const value = this.currentValue(key);
    return typeof value === 'boolean' ? value : fallback;
  }

  updateNumber(key: string, value: string | number, min: number, max: number, integer: boolean): void {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return;
    }
    const bounded = Math.min(max, Math.max(min, numeric));
    this.updateField(key, integer ? Math.round(bounded) : bounded);
  }

  updateString(key: string, value: string): void {
    this.updateField(key, value);
  }

  updateBoolean(key: string, value: boolean): void {
    this.updateField(key, value);
  }

  setJsonText(value: string): void {
    this.configTextChange.emit(value);
  }

  jsonError(): string | null {
    const normalized = this.configText.trim();
    if (!normalized) {
      return null;
    }
    try {
      const parsed: unknown = JSON.parse(normalized);
      return isRecord(parsed) ? null : 'Overrides must be a JSON object.';
    } catch {
      return 'Invalid JSON overrides.';
    }
  }

  private currentValue(key: string): unknown {
    const parsed = this.parseConfigText();
    if (parsed && key in parsed) {
      return parsed[key];
    }
    return this.config?.[key];
  }

  private updateField(key: string, value: unknown): void {
    const next = {
      ...(this.parseConfigText() ?? this.config ?? {}),
      [key]: value,
    };
    this.configTextChange.emit(JSON.stringify(next, null, 2));
  }

  private parseConfigText(): Record<string, unknown> | null {
    const normalized = this.configText.trim();
    if (!normalized) {
      return {};
    }
    try {
      const parsed: unknown = JSON.parse(normalized);
      return isRecord(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }
}
