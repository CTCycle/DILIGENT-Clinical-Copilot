// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output } from '@angular/core';

import { AgentModelRef, RevisionConfig } from '../../core/models/agent-types';
import { AgentModelFieldComponent } from './agent-model-field.component';
import { AgentModelsSectionComponent } from './agent-models-section.component';
import { INSTRUCTION_MAX_LENGTH, ModelCatalog, ProviderOption } from './agent-form';

/** Custom instructions and model of the DILI Report Review capability; the pipeline itself is fixed. */
@Component({
  selector: 'app-revision-sections',
  standalone: true,
  imports: [AgentModelFieldComponent, AgentModelsSectionComponent],
  styleUrl: './agent-sections.scss',
  templateUrl: './revision-sections.component.html',
})
export class RevisionSectionsComponent {
  readonly config = input.required<RevisionConfig>();
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly disabled = input(false);
  readonly configChange = output<RevisionConfig>();

  readonly limit = INSTRUCTION_MAX_LENGTH;

  private update(change: (draft: RevisionConfig) => void): void {
    const draft = structuredClone(this.config());
    change(draft);
    this.configChange.emit(draft);
  }

  setInstructions(value: string): void {
    this.update((d) => (d.instructions.custom_instructions = value));
  }

  setModel(ref: AgentModelRef): void {
    this.update((d) => (d.models.planning = ref));
  }
}
