// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, input, output } from '@angular/core';

import { AgentModelRef } from '../../core/models/agent-types';
import { ReasoningLevel } from '../../core/models/types';
import { ModelCatalog, ProviderOption } from './agent-form';

let nextFieldId = 0;
const REASONING: readonly ReasoningLevel[] = ['off', 'low', 'medium', 'high'];

/** Provider + model (+ optional reasoning override) for one agent role. */
@Component({
  selector: 'app-agent-model-field',
  standalone: true,
  styleUrl: './agent-sections.scss',
  template: `
    <div class="model-field">
      @if (optional()) {
        <label class="check">
          <input type="checkbox" [checked]="!!ref()" (change)="toggle($any($event.target).checked)" />
          {{ label() }}
        </label>
      } @else {
        <strong>{{ label() }} <span aria-hidden="true">*</span></strong>
      }
      @if (ref(); as current) {
        <div class="field-grid">
          <label class="field">Provider
            <select (change)="patch({ provider: $any($event.target).value })">
              @for (option of providers(); track option.value) {
                <option [value]="option.value" [selected]="option.value === current.provider">{{ option.label }}</option>
              }
            </select>
          </label>
          <label class="field">Model
            <input type="text" [value]="current.model" [attr.list]="listId" autocomplete="off"
              (input)="patch({ model: $any($event.target).value })" />
            <datalist [id]="listId">
              @for (model of modelsFor(current.provider); track model) { <option [value]="model"></option> }
            </datalist>
          </label>
          <label class="field">Reasoning
            <select (change)="setReasoning($any($event.target).value)">
              <option value="" [selected]="!current.reasoning_level">Inherit agent level</option>
              @for (level of levels; track level) {
                <option [value]="level" [selected]="current.reasoning_level === level">{{ level }}</option>
              }
            </select>
          </label>
        </div>
      }
    </div>
  `,
})
export class AgentModelFieldComponent {
  readonly label = input.required<string>();
  readonly ref = input<AgentModelRef | null | undefined>(null);
  readonly optional = input(false);
  /** Seed used when an optional override is switched on. */
  readonly fallback = input<AgentModelRef | null>(null);
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly refChange = output<AgentModelRef | null>();

  readonly listId = `agent-model-list-${++nextFieldId}`;
  readonly levels = REASONING;
  readonly providers = computed(() => {
    const options = this.providerOptions();
    const current = this.ref()?.provider;
    return current && !options.some((o) => o.value === current)
      ? [...options, { value: current, label: current }]
      : options;
  });

  modelsFor(provider: string): readonly string[] {
    return this.catalog()[provider] ?? [];
  }

  toggle(enabled: boolean): void {
    const base = this.fallback();
    this.refChange.emit(enabled ? { provider: base?.provider ?? 'ollama', model: base?.model ?? '' } : null);
  }

  patch(change: Partial<AgentModelRef>): void {
    const current = this.ref();
    if (!current) return;
    this.refChange.emit({ ...current, ...change });
  }

  setReasoning(value: string): void {
    this.patch({ reasoning_level: (value || null) as ReasoningLevel | null });
  }
}
