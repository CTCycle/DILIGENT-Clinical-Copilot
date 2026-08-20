import { Component, EventEmitter, Input, Output } from '@angular/core';

import { ModelRole } from '../model-config.types';

@Component({
  selector: 'app-model-role-action-button',
  standalone: true,
  template: `
    <button
      class="access-key-action model-config-role-action"
      [class.is-active]="selected"
      type="button"
      (click)="roleSelected.emit(role)"
      [disabled]="disabled || selected"
      [attr.aria-pressed]="selected"
      [attr.aria-label]="ariaLabel"
      [attr.title]="title"
    >
      <svg viewBox="0 0 24 24" aria-hidden="true">
        @if (role === 'clinical') {
          <path d="M12 21s-6.7-3.5-6.7-9.3A3.7 3.7 0 0 1 9 8a3.9 3.9 0 0 1 3 1.5A3.9 3.9 0 0 1 15 8a3.7 3.7 0 0 1 3.7 3.7C18.7 17.5 12 21 12 21Z" />
        } @else if (role === 'revision') {
          <path d="m6 17 2.4-2.4M8.4 14.6l3.2 3.2M14.5 6.5l3 3M12.5 8.5l3-3M5 19l2.1-2.1M15.9 8.1 19 5" />
          <path d="M9.7 12.3 12 10l2 2-2.3 2.3M7.5 5.5h4v4h-4z" />
        } @else if (role === 'timeline') {
          <path d="M5 7h14M5 12h14M5 17h14" />
          <circle cx="8" cy="7" r="1.5" /><circle cx="15" cy="12" r="1.5" /><circle cx="10" cy="17" r="1.5" />
        } @else {
          <path d="m8 12 2.5 2.5L16 9" />
          <rect x="4" y="4" width="16" height="16" rx="3" />
        }
      </svg>
    </button>
  `,
})
export class ModelRoleActionButtonComponent {
  @Input({ required: true }) role!: ModelRole;
  @Input({ required: true }) modelName = '';
  @Input() selected = false;
  @Input() disabled = false;

  @Output() roleSelected = new EventEmitter<ModelRole>();

  get roleLabel(): string {
    return {
      clinical: 'clinical model',
      text_extraction: 'text extraction model',
      revision: 'Revision model',
      timeline: 'Timeline model',
    }[this.role];
  }

  get selectedLabel(): string {
    return `${this.roleLabel[0].toUpperCase()}${this.roleLabel.slice(1)} selected`;
  }

  get ariaLabel(): string {
    return this.selected ? this.selectedLabel : `Set ${this.modelName} as ${this.roleLabel}`;
  }

  get title(): string {
    return this.selected ? this.selectedLabel : `Set as ${this.roleLabel}`;
  }
}
