// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, inject, input, output, signal } from '@angular/core';

import { AgentModelRef } from '../../core/models/agent-types';
import { ReasoningLevel } from '../../core/models/types';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { ModelCatalog, ProviderOption } from './agent-form';

const REASONING: readonly ReasoningLevel[] = ['off', 'low', 'medium', 'high'];

/**
 * Provider, model and reasoning selectors for one agent role.
 * `layout="row"` aligns them on one row; `layout="stack"` stacks them for column cards.
 */
@Component({
  selector: 'app-agent-model-field',
  standalone: true,
  styleUrl: './agent-sections.scss',
  template: `
    <div class="model-field">
      @if (headed()) {
        <div class="model-field-header">
          @if (optional()) {
            <label class="check">
              <input type="checkbox" [checked]="!!ref()" (change)="toggle($any($event.target).checked)" />
              {{ label() }}
            </label>
          } @else {
            <strong>{{ label() }} <span aria-hidden="true">*</span></strong>
          }
          @if (refreshable() && ref(); as current) {
            <button type="button" class="btn" [disabled]="refreshing()"
              [attr.aria-label]="'Refresh models for ' + label()" (click)="refresh(current.provider)">
              {{ refreshing() ? 'Refreshing…' : 'Refresh models' }}
            </button>
          }
        </div>
      }
      @if (ref(); as current) {
        <div class="model-row" [class.is-stack]="layout() === 'stack'">
          <label class="field">Provider
            <select (change)="setProvider($any($event.target).value)">
              @for (option of providers(); track option.value) {
                <option [value]="option.value" [selected]="option.value === current.provider">{{ option.label }}</option>
              }
            </select>
          </label>
          <label class="field">Model
            <select (change)="patch({ model: $any($event.target).value })" [disabled]="loading()">
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
            @if (modelNote(); as note) {
              <span class="field-hint" [class.is-warning]="note.warn">{{ note.text }}</span>
            }
          </label>
          <label class="field">Reasoning
            <select (change)="setReasoning($any($event.target).value)">
              <option value="" [selected]="!current.reasoning_level">Inherit agent level</option>
              @for (level of levelsFor(current); track level) {
                <option [value]="level" [selected]="current.reasoning_level === level">{{ level }}</option>
              }
            </select>
          </label>
        </div>
        @if (refreshError()) {
          <p class="field-error" role="alert">{{ refreshError() }}</p>
        }
      }
    </div>
  `,
})
export class AgentModelFieldComponent {
  private readonly modelConfig = inject(ModelConfigStateService);

  readonly label = input.required<string>();
  readonly ref = input<AgentModelRef | null | undefined>(null);
  readonly optional = input(false);
  /** Seed used when an optional override is switched on. */
  readonly fallback = input<AgentModelRef | null>(null);
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly layout = input<'row' | 'stack'>('row');
  /** When false the field shows no refresh button; the parent offers one refresh for all fields. */
  readonly refreshable = input(true);
  /** When false the label/toggle header is omitted (the parent renders its own). */
  readonly headed = input(true);
  readonly refChange = output<AgentModelRef | null>();

  readonly refreshing = signal(false);
  readonly refreshError = signal('');
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

  toggle(enabled: boolean): void {
    const base = this.fallback();
    this.refChange.emit(enabled ? { provider: base?.provider ?? 'ollama', model: base?.model ?? '' } : null);
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

  async refresh(provider: string): Promise<void> {
    if (this.refreshing()) return;
    this.refreshing.set(true);
    this.refreshError.set('');
    const failures = await this.modelConfig.refreshCatalogs([provider]);
    this.refreshError.set(failures[provider] ?? '');
    this.refreshing.set(false);
  }
}
