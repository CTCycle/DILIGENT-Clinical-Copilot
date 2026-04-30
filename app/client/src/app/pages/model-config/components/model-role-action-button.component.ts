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
      @if (role === 'clinical') {
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M12 21s-6.8-4.9-9-8a4.8 4.8 0 0 1 7-6l2 2 2-2a4.8 4.8 0 0 1 7 6c-2.2 3.1-9 8-9 8Z" />
        </svg>
      } @else {
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
          <path d="M14 3v6h6" />
          <path d="M8 13h8" />
          <path d="M8 17h5" />
        </svg>
      }
      <span class="visually-hidden">{{ hiddenLabel }}</span>
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

  get ariaLabel(): string {
    return this.selected ? this.selectedLabel : `Set ${this.modelName} as ${this.roleLabel}`;
  }

  get title(): string {
    return this.selected ? this.selectedLabel : `Set as ${this.roleLabel}`;
  }
}
