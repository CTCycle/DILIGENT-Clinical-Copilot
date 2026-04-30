import { Component, input, output } from '@angular/core';

@Component({
  selector: 'app-inspection-action-icon-button',
  standalone: true,
  template: `
    <button
      [class]="buttonClass()"
      type="button"
      [disabled]="disabled()"
      [attr.aria-label]="ariaLabel()"
      [attr.title]="title()"
      (click)="activated.emit()"
    >
      <svg class="inspection-icon" viewBox="0 0 24 24" aria-hidden="true">
        <path [attr.d]="iconPath()" />
      </svg>
    </button>
  `,
})
export class InspectionActionIconButtonComponent {
  readonly ariaLabel = input.required<string>();
  readonly title = input.required<string>();
  readonly iconPath = input.required<string>();
  readonly buttonClass = input('inspection-icon-button');
  readonly disabled = input(false);

  readonly activated = output<void>();
}
