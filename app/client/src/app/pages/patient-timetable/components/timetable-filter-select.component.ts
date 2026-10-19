// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output } from '@angular/core';

export type TimetableFilterOption = Readonly<{
  value: string;
  label: string;
}>;

@Component({
  selector: 'app-timetable-filter-select',
  standalone: true,
  template: `
    <label class="toolbar-group">
      {{ label() }}
      <select [value]="value()" (change)="handleChange($event)">
        @for (option of options(); track option.value) {
          <option [value]="option.value">{{ option.label }}</option>
        }
      </select>
    </label>
  `,
  styles: [
    `
      :host {
        display: contents;
      }

      .toolbar-group {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        gap: var(--space-xs);
        min-width: 160px;
        color: var(--color-text-secondary);
        font-size: var(--font-sm);
        font-weight: 600;
        line-height: 1.2;
      }

      select {
        width: 100%;
        min-width: 0;
        min-height: var(--control-height-md);
        padding: var(--space-sm) var(--space-md);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-sm);
        background: var(--color-surface);
        color: var(--color-text-primary);
        font: inherit;
        cursor: pointer;
      }

      select:hover:not(:disabled) {
        border-color: var(--color-brand);
      }

      select:focus-visible {
        outline: var(--focus-ring);
        outline-offset: var(--focus-ring-offset);
      }

      select:disabled {
        cursor: not-allowed;
        opacity: 0.6;
      }
    `,
  ],
})
export class TimetableFilterSelectComponent {
  readonly label = input.required<string>();
  readonly value = input.required<string>();
  readonly options = input.required<readonly TimetableFilterOption[]>();

  readonly valueChange = output<string>();

  handleChange(event: Event): void {
    const target = event.target;
    if (target instanceof HTMLSelectElement) {
      this.valueChange.emit(target.value);
    }
  }
}
