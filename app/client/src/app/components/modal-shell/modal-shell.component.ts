import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

export type ModalShellDialogClassName = 'modal-container' | 'modal-container modal-container-wide' | string;

@Component({
  selector: 'app-modal-shell',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './modal-shell.component.html',
})
export class ModalShellComponent {
  @Input() isOpen = false;
  @Input() ariaLabelledBy?: string;
  @Input() ariaDescribedBy?: string;
  @Input() ariaLabel?: string;
  @Input() title = '';
  @Input() subtitle?: string;
  @Input() titleId?: string;
  @Input() dialogClassName: ModalShellDialogClassName = 'modal-container';
  @Input() closeLabel = 'Close modal';
  @Input() footer = false;

  @Output() close = new EventEmitter<void>();
}
