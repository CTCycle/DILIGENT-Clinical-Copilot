// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, inject, input, output } from '@angular/core';

import { AgentModelRef } from '../../core/models/agent-types';
import { ReasoningLevel } from '../../core/models/types';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { ModelCatalog, ProviderOption } from './agent-form';

const REASONING: readonly ReasoningLevel[] = ['off', 'low', 'medium', 'high'];

let nextFieldId = 0;

/**
 * Provider, model and reasoning selectors for one agent role.
 * `layout="row"` aligns them on one row; `layout="stack"` stacks them for side-by-side columns.
 * Catalog notes render on their own line below the controls, so a warning never shifts or
 * misaligns neighbouring selects; in stack layout that line keeps its height when empty so
 * adjacent columns stay aligned.
 */
@Component({
  selector: 'app-agent-model-field',
  standalone: true,
  styleUrl: './agent-sections.scss',
  template: `
    @if (ref(); as current) {
      <div class="model-field" [class.is-stack]="layout() === 'stack'">
        <div class="model-row">
          <label class="field">Provider
            <select [attr.aria-label]="label() + ' provider'" (change)="setProvider($any($event.target).value)">
              @for (option of providers(); track option.value) {
                <option [value]="option.value" [selected]="option.value === current.provider">{{ option.label }}</option>
              }
            </select>
          </label>
          <label class="field">Model
            <select [attr.aria-label]="label() + ' model'" [attr.aria-describedby]="modelNote() ? noteId : null"
              (change)="patch({ model: $any($event.target).value })" [disabled]="loading()">
              @if (loading()) {
                <option value="" selected>Loading models…</option>
              } @else {
                @if (!current.model) {
                  <option value="" selected disabled>Select a model</option>
                }
                @for (model of models(); track model) {
                  <option [value]="model" [selected]="model === current.model">{{ model }}</option>
                }
                @if (current.model && !models().includes(current.model)) {
                  <option [value]="current.model" selected>{{ current.model }} (not in catalog)</option>
                }
              }
            </select>
          </label>
          <label class="field">Reasoning
            <select [attr.aria-label]="label() + ' reasoning'" (change)="setReasoning($any($event.target).value)">
              <option value="" [selected]="!current.reasoning_level">Model default</option>
              @for (level of levelsFor(current); track level) {
                <option [value]="level" [selected]="current.reasoning_level === level">{{ level }}</option>
              }
            </select>
          </label>
        </div>
        <p class="model-note" [id]="noteId" [class.is-warning]="modelNote()?.warn" aria-live="polite">
          {{ modelNote()?.text ?? '' }}
        </p>
      </div>
    }
  `,
})
export class AgentModelFieldComponent {
  private readonly modelConfig = inject(ModelConfigStateService);

  readonly label = input.required<string>();
  readonly ref = input<AgentModelRef | null | undefined>(null);
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly layout = input<'row' | 'stack'>('row');
  readonly refChange = output<AgentModelRef>();

  readonly noteId = `agent-model-note-${nextFieldId++}`;
  readonly providers = computed(() => {
    const options = this.providerOptions();
    const current = this.ref()?.provider;
    return current && !options.some((o) => o.value === current)
      ? [...options, { value: current, label: current }]
      : options;
  });
  readonly models = computed<readonly string[]>(() => this.catalog()[this.ref()?.provider ?? ''] ?? []);

  /** True until the first model-config response, or while a never-loaded catalog is being fetched. */
  readonly loading = computed(() => {
    if (this.models().length > 0) return false;
    const status = this.modelConfig.status();
    if (status === 'idle' || status === 'loading') return !this.modelConfig.data();
    const info = this.modelConfig.catalogInfo(this.ref()?.provider ?? '');
    return this.modelConfig.catalogBusy() && info?.status === 'not_loaded';
  });

  /** Explains an empty or stale model list; unavailable and not-yet-loaded are reported differently. */
  readonly modelNote = computed<{ text: string; warn: boolean } | null>(() => {
    const current = this.ref();
    if (!current || this.loading()) return null;
    const info = this.modelConfig.catalogInfo(current.provider);
    if (!info && this.modelConfig.status() === 'error') {
      return { text: 'The model catalog could not be loaded. Try Refresh.', warn: true };
    }
    if (this.models().length === 0) {
      switch (info?.status) {
        case 'not_loaded':
          return { text: 'Models have not been loaded yet. Use Refresh.', warn: false };
        case 'authentication_required':
          return { text: info.message || 'An access key is required to list models.', warn: true };
        case 'unavailable':
          return { text: info.message || 'The provider could not be reached.', warn: true };
        default:
          return { text: 'The provider returned no models.', warn: true };
      }
    }
    if (current.model && !this.models().includes(current.model)) {
      return { text: 'The saved model is not in the provider’s current list.', warn: true };
    }
    return null;
  });

  /** Reasoning levels the selected model supports, per catalog capability metadata when known. */
  levelsFor(current: AgentModelRef): readonly ReasoningLevel[] {
    const provider = this.modelConfig.data()?.cloud_providers.find((item) => item.id === current.provider);
    const capabilities = provider?.models.find((model) => model.id === current.model)?.model_capabilities;
    let levels = REASONING;
    if (capabilities?.supports_reasoning === false) {
      levels = ['off'];
    } else if (capabilities?.reasoning_levels?.length) {
      levels = REASONING.filter((level) => level === 'off' || capabilities.reasoning_levels.includes(level));
    }
    return current.reasoning_level && !levels.includes(current.reasoning_level)
      ? [...levels, current.reasoning_level]
      : levels;
  }

  patch(change: Partial<AgentModelRef>): void {
    const current = this.ref();
    if (!current) return;
    this.refChange.emit({ ...current, ...change });
  }

  /** Switching provider keeps the model only when the new provider offers it; it never picks another. */
  setProvider(provider: string): void {
    const current = this.ref();
    if (!current) return;
    const available = this.catalog()[provider] ?? [];
    this.patch({ provider, model: available.includes(current.model) ? current.model : '' });
  }

  setReasoning(value: string): void {
    this.patch({ reasoning_level: (value || null) as ReasoningLevel | null });
  }
}
