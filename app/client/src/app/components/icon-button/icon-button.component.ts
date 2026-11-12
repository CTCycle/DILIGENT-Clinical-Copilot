// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, input, output } from '@angular/core';

let nextTooltipId = 0;

/**
 * Compact icon button with a keyboard- and hover-reachable tooltip.
 *
 * The projected content is the icon (for example `<svg lucidePencil></svg>`).
 * `label` is the accessible name; `reason` replaces the tooltip while the button
 * is disabled so the user learns why. The tooltip lives on a wrapper so it also
 * appears over disabled buttons, which do not receive pointer events.
 */
@Component({
  selector: 'app-icon-button',
  standalone: true,
  template: `
    <span class="icon-button-host" [attr.data-tooltip]="tooltip()">
      <button
        type="button"
        class="icon-button"
        [class.is-danger]="variant() === 'danger'"
        [class.is-active]="pressed() !== null && pressed()"
        [attr.aria-label]="label()"
        [attr.aria-describedby]="tooltip() ? tooltipId : null"
        [attr.aria-pressed]="pressed()"
        [disabled]="disabled()"
        (click)="activated.emit()"
      >
        <ng-content></ng-content>
      </button>
      @if (tooltip()) {
        <span class="visually-hidden" [id]="tooltipId">{{ tooltip() }}</span>
      }
    </span>
  `,
  styles: `
    :host { display: inline-flex; }
    .icon-button-host { position: relative; display: inline-flex; }
    .icon-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: var(--control-height-md);
      height: var(--control-height-md);
      padding: 0;
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      background: var(--color-surface);
      color: var(--color-text-secondary);
      cursor: pointer;
    }
    .icon-button :is(svg) { width: 1.1rem; height: 1.1rem; }
    .icon-button:hover:not(:disabled) { color: var(--color-brand); border-color: var(--color-brand); }
    .icon-button.is-active { color: var(--color-brand); background: var(--color-brand-bg); border-color: var(--color-brand); }
    .icon-button.is-danger:hover:not(:disabled) { color: var(--color-status-error-text); border-color: var(--color-status-error-text); }
    .icon-button:disabled { opacity: 0.5; cursor: not-allowed; }
    .icon-button:focus-visible { outline: var(--focus-ring); outline-offset: var(--focus-ring-offset); }

    .icon-button-host[data-tooltip]::after {
      content: attr(data-tooltip);
      position: absolute;
      bottom: calc(100% + 6px);
      left: 50%;
      z-index: 20;
      width: max-content;
      max-width: 16rem;
      padding: var(--space-xs) var(--space-sm);
      border-radius: var(--radius-sm);
      background: var(--color-text-primary);
      color: var(--color-surface);
      font-size: var(--font-xs);
      line-height: 1.35;
      text-align: center;
      white-space: normal;
      pointer-events: none;
      opacity: 0;
      transform: translateX(-50%);
      transition: opacity 0.12s ease;
    }
    .icon-button-host[data-tooltip]:hover::after,
    .icon-button-host[data-tooltip]:focus-within::after { opacity: 1; }
    .visually-hidden {
      position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap;
    }
  `,
})
export class IconButtonComponent {
  readonly label = input.required<string>();
  /** Why the action is unavailable; shown instead of the label while disabled. */
  readonly reason = input<string | null>(null);
  readonly disabled = input(false);
  readonly variant = input<'default' | 'danger'>('default');
  /** Toggle state for aria-pressed; null for plain buttons. */
  readonly pressed = input<boolean | null>(null);
  readonly activated = output<void>();

  readonly tooltipId = `icon-button-tip-${++nextTooltipId}`;
  readonly tooltip = computed(() => (this.disabled() && this.reason() ? this.reason() : this.label()));
}
