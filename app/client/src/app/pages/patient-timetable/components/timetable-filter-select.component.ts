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
