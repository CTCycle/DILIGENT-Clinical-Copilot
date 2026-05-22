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
      <span>{{ visibleLabel }}</span>
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
    if (this.selected) {
      return this.role === 'clinical' ? 'Clinical' : 'Extraction';
    }
    return this.role === 'clinical' ? 'Use for Clinical' : 'Use for Text Extraction';
  }

  get ariaLabel(): string {
    return this.selected ? this.selectedLabel : `Set ${this.modelName} as ${this.roleLabel}`;
  }

  get title(): string {
    return this.selected ? this.selectedLabel : `Set as ${this.roleLabel}`;
  }
}
