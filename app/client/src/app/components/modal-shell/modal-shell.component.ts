import { CommonModule, DOCUMENT } from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  Inject,
  Input,
  OnChanges,
  OnDestroy,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';

export type ModalShellDialogClassName =
  | 'modal-container'
  | 'modal-container modal-container-wide'
  | string;

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

@Component({
  selector: 'app-modal-shell',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './modal-shell.component.html',
  styleUrl: './modal-shell.component.scss',
})
export class ModalShellComponent implements AfterViewInit, OnChanges, OnDestroy {
  @ViewChild('dialog') private dialog?: ElementRef<HTMLDialogElement>;

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
  @Input() showCloseButton = true;
  @Input() closeOnEscape = true;
  @Input() restoreFocus = true;

  @Output() close = new EventEmitter<void>();
  @Output() closed = new EventEmitter<void>();

  private previousActiveElement: HTMLElement | null = null;
  private previousBodyOverflow = '';
  private activated = false;

  constructor(@Inject(DOCUMENT) private readonly document: Document) {}

  ngAfterViewInit(): void {
    if (this.isOpen) {
      queueMicrotask(() => this.activate());
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (!changes['isOpen']) {
      return;
    }
    if (this.isOpen) {
      queueMicrotask(() => this.activate());
      return;
    }
    this.deactivate();
  }

  ngOnDestroy(): void {
    this.deactivate();
  }

  requestClose(): void {
    this.close.emit();
  }

  handleCancel(event: Event): void {
    event.preventDefault();
    if (this.closeOnEscape) {
      this.requestClose();
    }
  }

  @HostListener('document:keydown', ['$event'])
  handleDocumentKeydown(event: KeyboardEvent): void {
    if (!this.isOpen) {
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      if (this.closeOnEscape) {
        this.requestClose();
      }
      return;
    }
    if (event.key === 'Tab') {
      this.trapFocus(event);
    }
  }

  private activate(): void {
    const dialog = this.dialog?.nativeElement;
    if (!dialog) {
      return;
    }
    if (!dialog.open) {
      try {
        dialog.showModal();
      } catch {
        dialog.setAttribute('open', '');
      }
    }
    if (!this.activated) {
      this.previousActiveElement =
        this.document.activeElement instanceof HTMLElement
          ? this.document.activeElement
          : null;
      this.previousBodyOverflow = this.document.body.style.overflow;
      this.document.body.style.overflow = 'hidden';
      this.activated = true;
    }
    const initial = dialog.querySelector<HTMLElement>('[data-modal-initial-focus]');
    const first = this.focusableElements(dialog)[0];
    (initial ?? first ?? dialog).focus();
  }

  private deactivate(): void {
    if (!this.activated) {
      return;
    }
    const dialog = this.dialog?.nativeElement;
    if (dialog?.open) {
      try {
        dialog.close();
      } catch {
        dialog.removeAttribute('open');
      }
    }
    this.document.body.style.overflow = this.previousBodyOverflow;
    const focusTarget = this.previousActiveElement;
    this.previousActiveElement = null;
    this.activated = false;
    if (this.restoreFocus && focusTarget?.isConnected) {
      queueMicrotask(() => focusTarget.focus());
    }
    this.closed.emit();
  }

  private trapFocus(event: KeyboardEvent): void {
    const dialog = this.dialog?.nativeElement;
    if (!dialog) {
      return;
    }
    const focusable = this.focusableElements(dialog);
    if (!focusable.length) {
      event.preventDefault();
      dialog.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = this.document.activeElement;
    if (event.shiftKey && active === first) {
      event.preventDefault();
      last.focus();
      return;
    }
    if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  }

  private focusableElements(dialog: HTMLElement): HTMLElement[] {
    return Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
      (element) => !element.hasAttribute('hidden') && element.offsetParent !== null,
    );
  }
}
