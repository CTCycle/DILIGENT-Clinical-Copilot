// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output } from '@angular/core';

import { ModalShellComponent } from '../modal-shell/modal-shell.component';

/** Confirmation dialog for destructive or hard-to-reverse actions, built on the shared modal shell. */
@Component({
  selector: 'app-confirm-dialog',
  standalone: true,
  imports: [ModalShellComponent],
  template: `
    <app-modal-shell
      [isOpen]="open()"
      [title]="title()"
      [titleId]="'confirm-dialog-title'"
      [ariaLabelledBy]="'confirm-dialog-title'"
      [ariaDescribedBy]="'confirm-dialog-message'"
      [footer]="true"
      closeLabel="Cancel"
      (close)="cancelled.emit()"
    >
      <p id="confirm-dialog-message" class="message">{{ message() }}</p>
      <div modal-footer class="actions">
        <button type="button" class="btn" (click)="cancelled.emit()">{{ cancelLabel() }}</button>
        <button type="button" class="btn" [class.is-danger]="destructive()" [class.is-primary]="!destructive()"
          (click)="confirmed.emit()">{{ confirmLabel() }}</button>
      </div>
    </app-modal-shell>
  `,
  styles: `
    .message { margin: 0; line-height: 1.5; }
    .actions { display: flex; justify-content: flex-end; gap: var(--space-sm); }
    .btn {
      min-height: var(--control-height-md);
      padding: 0 var(--space-md);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      background: var(--color-surface);
      color: var(--color-text-primary);
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    .btn.is-primary { background: var(--color-brand); border-color: var(--color-brand); color: var(--color-text-on-brand); }
    .btn.is-danger { background: var(--color-status-error-text); border-color: var(--color-status-error-text); color: #fff; }
    .btn:focus-visible { outline: var(--focus-ring); outline-offset: var(--focus-ring-offset); }
  `,
})
export class ConfirmDialogComponent {
  readonly open = input(false);
  readonly title = input.required<string>();
  readonly message = input.required<string>();
  readonly confirmLabel = input('Confirm');
  readonly cancelLabel = input('Cancel');
  readonly destructive = input(false);
  readonly confirmed = output<void>();
  readonly cancelled = output<void>();
}
