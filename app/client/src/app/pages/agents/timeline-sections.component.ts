// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output } from '@angular/core';

import { AgentModelRef, TimelineConfig } from '../../core/models/agent-types';
import { ReasoningLevel } from '../../core/models/types';
import { AgentModelFieldComponent } from './agent-model-field.component';
import { INSTRUCTION_MAX_LENGTH, ModelCatalog, ProviderOption } from './agent-form';

const REASONING: readonly ReasoningLevel[] = ['off', 'low', 'medium', 'high'];

/** Instructions, model and parameters of the Timeline capability. */
@Component({
  selector: 'app-timeline-sections',
  standalone: true,
  imports: [AgentModelFieldComponent],
  styleUrl: './agent-sections.scss',
  template: `
    <fieldset class="plain" [disabled]="disabled()">
      <section class="subsection" aria-labelledby="timeline-instructions-heading">
        <h3 id="timeline-instructions-heading">Instructions</h3>
        <p class="agent-section-note">
          Guidance is added to the timeline extraction prompt. It cannot override source grounding,
          date handling or the required output structure.
        </p>
        <label class="field">
          Timeline focus
          <textarea rows="4" [value]="config().instructions.timeline_focus"
            (input)="setFocus($any($event.target).value)"></textarea>
          <span class="field-counter" [class.is-over]="config().instructions.timeline_focus.length > limit">
            {{ config().instructions.timeline_focus.length }} / {{ limit }}
          </span>
        </label>
      </section>

      <section class="subsection" aria-labelledby="timeline-model-heading">
        <h3 id="timeline-model-heading">Model</h3>
        <app-agent-model-field label="Timeline model" [ref]="config().model"
          [providerOptions]="providerOptions()" [catalog]="catalog()" (refChange)="setModel($event)" />
        <div class="field-grid">
          <label class="field">Reasoning level
            <select (change)="setReasoning($any($event.target).value)">
              @for (level of reasoningLevels; track level) {
                <option [value]="level" [selected]="config().reasoning_level === level">{{ level }}</option>
              }
            </select>
          </label>
          <label class="field">Ollama seed
            <input type="number" min="0" step="1" [value]="config().ollama_seed ?? ''"
              (input)="setSeed($any($event.target).value)" />
          </label>
        </div>
      </section>
    </fieldset>
  `,
})
export class TimelineSectionsComponent {
  readonly config = input.required<TimelineConfig>();
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly disabled = input(false);
  readonly configChange = output<TimelineConfig>();

  readonly reasoningLevels = REASONING;
  readonly limit = INSTRUCTION_MAX_LENGTH;

  private update(change: (draft: TimelineConfig) => void): void {
    const draft = structuredClone(this.config());
    change(draft);
    this.configChange.emit(draft);
  }

  setFocus(value: string): void {
    this.update((d) => (d.instructions.timeline_focus = value));
  }

  setModel(ref: AgentModelRef | null): void {
    if (ref) this.update((d) => (d.model = ref));
  }

  setReasoning(level: string): void {
    this.update((d) => (d.reasoning_level = level as ReasoningLevel));
  }

  setSeed(value: string): void {
    this.update((d) => (d.ollama_seed = value.trim() === '' ? null : Number(value)));
  }
}
