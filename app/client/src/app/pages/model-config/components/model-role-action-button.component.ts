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
    return this.role === 'clinical' ? 'clinical model' : 'text extraction model';
  }

  get selectedLabel(): string {
    return this.role === 'clinical' ? 'Clinical model selected' : 'Text extraction model selected';
  }

  get hiddenLabel(): string {
    return this.selected ? this.selectedLabel : `Set as ${this.roleLabel}`;
  }

  get visibleLabel(): string {
    return this.role === 'clinical' ? 'Clinical' : 'Extraction';
  }

  get ariaLabel(): string {
    return this.selected ? this.selectedLabel : `Set ${this.modelName} as ${this.roleLabel}`;
  }

  get title(): string {
    return this.selected ? this.selectedLabel : `Set as ${this.roleLabel}`;
  }
}
