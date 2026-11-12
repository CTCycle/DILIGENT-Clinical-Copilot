// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, ElementRef, HostListener, inject, input, output, signal } from '@angular/core';

import { LucideEllipsis } from '@lucide/angular';

export interface OverflowMenuItem {
  id: string;
  label: string;
  /** Why the item is unavailable; a non-null value disables it and is shown as its tooltip. */
  reason?: string | null;
  danger?: boolean;
}

/** Small "more actions" menu: a trigger button and a popup list of text items. */
@Component({
  selector: 'app-overflow-menu',
  standalone: true,
  imports: [LucideEllipsis],
  template: `
    <button type="button" class="trigger" aria-haspopup="menu" [attr.aria-expanded]="open()"
      [attr.aria-label]="label()" [title]="label()" [disabled]="disabled()" (click)="toggle()">
      <svg lucideEllipsis aria-hidden="true"></svg>
    </button>
    @if (open()) {
      <ul class="menu" role="menu">
        @for (item of items(); track item.id) {
          <li role="none">
            <button type="button" role="menuitem" [class.is-danger]="item.danger"
              [disabled]="item.reason != null" [attr.title]="item.reason ?? null"
              (click)="choose(item)">{{ item.label }}</button>
          </li>
        }
      </ul>
    }
  `,
  styles: `
    :host { position: relative; display: inline-flex; }
    .trigger {
      display: inline-flex; align-items: center; justify-content: center;
      width: var(--control-height-md); height: var(--control-height-md); padding: 0;
      border: 1px solid transparent; border-radius: var(--radius-sm);
      background: transparent; color: var(--color-text-secondary); cursor: pointer;
    }
    .trigger svg { width: 1.1rem; height: 1.1rem; }
    .trigger:hover:not(:disabled), .trigger[aria-expanded='true'] { color: var(--color-brand); border-color: var(--color-border); }
    .trigger:disabled { opacity: 0.5; cursor: not-allowed; }
    .trigger:focus-visible, .menu button:focus-visible { outline: var(--focus-ring); outline-offset: var(--focus-ring-offset); }
    .menu {
      position: absolute; top: calc(100% + 4px); right: 0; z-index: 30;
      min-width: 12rem; margin: 0; padding: var(--space-xs); list-style: none;
      border: 1px solid var(--color-border); border-radius: var(--radius-md);
      background: var(--color-surface); box-shadow: 0 6px 20px rgb(0 0 0 / 0.25);
    }
    .menu button {
      width: 100%; padding: var(--space-sm) var(--space-md); border: 0; border-radius: var(--radius-sm);
      background: transparent; color: var(--color-text-primary); font: inherit; font-size: var(--font-sm);
      text-align: left; cursor: pointer;
    }
    .menu button:hover:not(:disabled) { background: var(--color-surface-alt); }
    .menu button.is-danger { color: var(--color-status-error-text); }
    .menu button:disabled { color: var(--color-text-muted); cursor: not-allowed; }
  `,
})
export class OverflowMenuComponent {
  readonly items = input.required<OverflowMenuItem[]>();
  readonly label = input('More actions');
  readonly disabled = input(false);
  readonly selected = output<string>();
  readonly open = signal(false);

  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);

  toggle(): void {
    this.open.update((value) => !value);
  }

  choose(item: OverflowMenuItem): void {
    this.open.set(false);
    this.selected.emit(item.id);
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: Event): void {
    if (this.open() && !this.host.nativeElement.contains(event.target as Node)) this.open.set(false);
  }

  @HostListener('keydown.escape')
  onEscape(): void {
    if (!this.open()) return;
    this.open.set(false);
    this.host.nativeElement.querySelector<HTMLButtonElement>('.trigger')?.focus();
  }
}
