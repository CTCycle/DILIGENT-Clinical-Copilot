// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output, signal } from '@angular/core';

import { LucideChevronDown } from '@lucide/angular';

let nextSectionId = 0;

/**
 * Collapsible, independently enabled configuration section.
 *
 * The body stays mounted while collapsed (hidden, not destroyed) so collapsing
 * never discards edits. The enable switch is independent of expansion.
 */
@Component({
  selector: 'app-capability-section',
  standalone: true,
  imports: [LucideChevronDown],
  template: `
    <section class="capability" [class.is-disabled]="!enabled()" [attr.aria-labelledby]="headingId">
      <header class="capability-header">
        <button type="button" class="capability-toggle" [attr.aria-expanded]="expanded()"
          [attr.aria-controls]="bodyId" (click)="expanded.set(!expanded())">
          <svg lucideChevronDown class="chevron" [class.is-collapsed]="!expanded()" aria-hidden="true"></svg>
          <span class="capability-icon" aria-hidden="true"><ng-content select="[section-icon]"></ng-content></span>
          <span class="capability-text">
            <h3 [id]="headingId">{{ title() }}</h3>
            @if (summary()) { <span class="capability-summary">{{ summary() }}</span> }
          </span>
        </button>
        <span class="capability-status" [class.is-on]="enabled()" role="status">
          {{ enabled() ? 'Enabled' : 'Disabled' }}
        </span>
        <label class="toggle-switch" [attr.title]="switchTooltip()">
          <input type="checkbox" role="switch" [checked]="enabled()" [disabled]="locked()"
            [attr.aria-label]="(enabled() ? 'Disable ' : 'Enable ') + title()"
            (change)="enabledChange.emit($any($event.target).checked)" />
          <span class="toggle-track"><span class="toggle-thumb"></span></span>
        </label>
      </header>
      <div class="capability-body" [id]="bodyId" [hidden]="!expanded()">
        @if (!enabled()) {
          <p class="capability-note">
            Disabled: this agent cannot be used for {{ title() }}. Settings below are kept and
            take effect again when the capability is re-enabled.
          </p>
        }
        <ng-content></ng-content>
      </div>
    </section>
  `,
  styles: `
    :host { display: block; min-width: 0; }
    .capability {
      border: 1px solid var(--color-border-subtle);
      border-radius: var(--radius-lg);
      background: var(--color-surface);
      min-width: 0;
    }
    .capability.is-disabled { background: var(--color-surface-alt); }
    .capability-header {
      display: flex;
      align-items: center;
      gap: var(--space-md);
      padding: var(--space-sm) var(--space-lg);
    }
    .capability-toggle {
      display: flex;
      flex: 1 1 auto;
      align-items: center;
      gap: var(--space-md);
      min-width: 0;
      padding: var(--space-xs) 0;
      border: 0;
      background: none;
      color: inherit;
      font: inherit;
      text-align: left;
      cursor: pointer;
    }
    .capability-toggle:focus-visible { outline: var(--focus-ring); outline-offset: var(--focus-ring-offset); }
    .chevron { width: 1.1rem; height: 1.1rem; flex: none; transition: transform 0.15s ease; }
    .chevron.is-collapsed { transform: rotate(-90deg); }
    .capability-icon { display: inline-flex; color: var(--color-brand); }
    .capability.is-disabled .capability-icon { color: var(--color-text-muted); }
    .capability-text { display: grid; min-width: 0; }
    .capability-text h3 { margin: 0; font-size: var(--font-md); }
    .capability-summary { color: var(--color-text-muted); font-size: var(--font-sm); overflow-wrap: anywhere; }
    .capability-status {
      padding: 2px var(--space-sm);
      border: 1px solid var(--color-border-subtle);
      border-radius: 999px;
      font-size: var(--font-xs);
      font-weight: 600;
      color: var(--color-text-muted);
      background: var(--color-surface-alt);
    }
    .capability-status.is-on {
      color: var(--color-status-success-text);
      background: var(--color-status-success-bg);
      border-color: var(--color-status-success-border);
    }
    .capability-body { display: grid; gap: var(--space-lg); padding: 0 var(--space-lg) var(--space-lg); }
    .capability-body[hidden] { display: none; }
    .capability-note { margin: 0; color: var(--color-text-muted); font-size: var(--font-sm); }
    .toggle-switch input:disabled + .toggle-track { opacity: 0.5; cursor: not-allowed; }
  `,
})
export class CapabilitySectionComponent {
  readonly title = input.required<string>();
  readonly summary = input<string>('');
  readonly enabled = input(true);
  /** True when the agent is not editable: the enable switch is read-only. */
  readonly locked = input(false);
  readonly switchTooltip = input<string | null>(null);
  readonly expanded = signal(true);
  readonly enabledChange = output<boolean>();

  private readonly uid = ++nextSectionId;
  readonly headingId = `capability-heading-${this.uid}`;
  readonly bodyId = `capability-body-${this.uid}`;
}
