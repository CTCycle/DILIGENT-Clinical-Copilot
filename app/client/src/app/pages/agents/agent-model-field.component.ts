// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, inject, input, output, signal } from '@angular/core';

import { AgentModelRef } from '../../core/models/agent-types';
import { ReasoningLevel } from '../../core/models/types';
import { refreshModelCatalog } from '../../core/services/model-config-api';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { formatUnknownError } from '../../core/utils';
import { ModelCatalog, ProviderOption } from './agent-form';

const REASONING: readonly ReasoningLevel[] = ['off', 'low', 'medium', 'high'];

/** Provider, model and reasoning selectors for one agent role, laid out on one aligned row. */
@Component({
  selector: 'app-agent-model-field',
  standalone: true,
  styleUrl: './agent-sections.scss',
  template: `
    <div class="model-field">
      <div class="model-field-header">
        @if (optional()) {
          <label class="check">
            <input type="checkbox" [checked]="!!ref()" (change)="toggle($any($event.target).checked)" />
            {{ label() }}
          </label>
        } @else {
          <strong>{{ label() }} <span aria-hidden="true">*</span></strong>
        }
        @if (ref(); as current) {
          <button type="button" class="btn" [disabled]="refreshing()"
            [attr.aria-label]="'Refresh models for ' + label()" (click)="refresh(current.provider)">
            {{ refreshing() ? 'Refreshing…' : 'Refresh models' }}
          </button>
        }
      </div>
      @if (ref(); as current) {
        <div class="model-row">
          <label class="field">Provider
            <select (change)="setProvider($any($event.target).value)">
              @for (option of providers(); track option.value) {
                <option [value]="option.value" [selected]="option.value === current.provider">{{ option.label }}</option>
              }
            </select>
          </label>
          <label class="field">Model
            <select (change)="patch({ model: $any($event.target).value })">
              @if (!current.model) {
                <option value="" selected disabled>Select a model</option>
              }
              @for (model of models(); track model) {
                <option [value]="model" [selected]="model === current.model">{{ model }}</option>
              }
              @if (current.model && !models().includes(current.model)) {
                <option [value]="current.model" selected>{{ current.model }} (not available)</option>
              }
            </select>
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
  readonly refChange = output<AgentModelRef | null>();

  readonly levels = REASONING;
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

  toggle(enabled: boolean): void {
    const base = this.fallback();
    this.refChange.emit(enabled ? { provider: base?.provider ?? 'ollama', model: base?.model ?? '' } : null);
  }

  patch(change: Partial<AgentModelRef>): void {
    const current = this.ref();
    if (!current) return;
    this.refChange.emit({ ...current, ...change });
  }

  /** Switching provider keeps the model only when the new provider offers it. */
  setProvider(provider: string): void {
    const current = this.ref();
    if (!current) return;
    const available = this.catalog()[provider] ?? [];
    this.patch({ provider, model: available.includes(current.model) ? current.model : (available[0] ?? '') });
  }

  setReasoning(value: string): void {
    this.patch({ reasoning_level: (value || null) as ReasoningLevel | null });
  }

  async refresh(provider: string): Promise<void> {
    if (this.refreshing()) return;
    this.refreshing.set(true);
    this.refreshError.set('');
    try {
      const result = await refreshModelCatalog(provider);
      this.modelConfig.setFromApiState(result.state);
      if (result.outcome === 'failed') {
        this.refreshError.set(result.error || 'Unable to refresh the model catalog.');
      }
    } catch (error) {
      this.refreshError.set(formatUnknownError(error, 'Unable to refresh the model catalog.'));
    } finally {
      this.refreshing.set(false);
    }
  }
}
