// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, inject, input, signal } from '@angular/core';

import { AgentModelRef } from '../../core/models/agent-types';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';

/**
 * Model section shared by every capability tab: heading, one "Refresh models" action for all
 * providers the projected model fields use, and the fields themselves.
 */
@Component({
  selector: 'app-agent-models-section',
  standalone: true,
  styleUrl: './agent-sections.scss',
  // As the first block of a tab the section needs no top divider, like any first subsection.
  styles: [':host(:first-child) .subsection { padding-top: 0; border-top: 0; }'],
  template: `
    <section class="subsection" [attr.aria-labelledby]="headingId()">
      <div class="section-head">
        <h3 [id]="headingId()">{{ heading() }}</h3>
        <div class="section-head-actions">
          @if (message()) {
            <span class="field-hint" [class.is-warning]="failed()" role="status">{{ message() }}</span>
          }
          <button type="button" class="btn btn-secondary btn-compact" [disabled]="busy() || disabled()"
            (click)="refresh()">
            {{ busy() ? 'Refreshing…' : 'Refresh models' }}
          </button>
        </div>
      </div>
      <ng-content />
    </section>
  `,
})
export class AgentModelsSectionComponent {
  private readonly modelConfig = inject(ModelConfigStateService);

  readonly heading = input('Models');
  readonly headingId = input.required<string>();
  readonly refs = input<ReadonlyArray<AgentModelRef | null | undefined>>([]);
  readonly disabled = input(false);

  readonly busy = this.modelConfig.catalogBusy;
  readonly message = signal('');
  readonly failed = signal(false);
  private readonly providers = computed(() =>
    Array.from(new Set(this.refs().flatMap((ref) => (ref?.provider ? [ref.provider] : [])))),
  );

  async refresh(): Promise<void> {
    this.message.set('');
    const failures = Object.entries(await this.modelConfig.refreshCatalogs(this.providers()));
    this.failed.set(failures.length > 0);
    this.message.set(
      failures.length ? failures.map(([provider, error]) => `${provider}: ${error}`).join(' · ') : 'Models updated.',
    );
  }
}
