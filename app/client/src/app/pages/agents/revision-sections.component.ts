// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output } from '@angular/core';

import {
  AgentModelRef,
  RevisionConfig,
  RevisionInstructions,
  RevisionStage,
} from '../../core/models/agent-types';
import { ReasoningLevel } from '../../core/models/types';
import { AgentModelFieldComponent } from './agent-model-field.component';
import {
  INSTRUCTION_MAX_LENGTH,
  ModelCatalog,
  ProviderOption,
  REVISION_TOOL_NAMES,
} from './agent-form';

const INSTRUCTION_FIELDS: ReadonlyArray<{ key: keyof RevisionInstructions; label: string }> = [
  { key: 'review_objectives', label: 'Review objectives' },
  { key: 'editing_style', label: 'Editing style' },
];
const STAGES: ReadonlyArray<{ stage: RevisionStage; label: string }> = [
  { stage: 'planning', label: 'Planning' },
  { stage: 'tool_selection', label: 'Tool selection' },
  { stage: 'editing', label: 'Editing' },
  { stage: 'qa', label: 'QA' },
];
const REASONING: readonly ReasoningLevel[] = ['off', 'low', 'medium', 'high'];

/** Instructions, models and execution limits of the DILI Revision capability. */
@Component({
  selector: 'app-revision-sections',
  standalone: true,
  imports: [AgentModelFieldComponent],
  styleUrl: './agent-sections.scss',
  templateUrl: './revision-sections.component.html',
})
export class RevisionSectionsComponent {
  readonly config = input.required<RevisionConfig>();
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly disabled = input(false);
  readonly configChange = output<RevisionConfig>();

  readonly instructionFields = INSTRUCTION_FIELDS;
  readonly overrideStages = STAGES.filter((item) => item.stage !== 'planning');
  readonly reasoningLevels = REASONING;
  readonly toolNames = REVISION_TOOL_NAMES;
  readonly limit = INSTRUCTION_MAX_LENGTH;

  private update(change: (draft: RevisionConfig) => void): void {
    const draft = structuredClone(this.config());
    change(draft);
    this.configChange.emit(draft);
  }

  setInstruction(key: keyof RevisionInstructions, value: string): void {
    this.update((d) => (d.instructions[key] = value));
  }

  setPlanning(ref: AgentModelRef | null): void {
    if (ref) this.update((d) => (d.models.planning = ref));
  }

  setOverride(stage: RevisionStage, ref: AgentModelRef | null): void {
    this.update((d) => {
      if (ref) d.models.stage_overrides[stage] = ref;
      else delete d.models.stage_overrides[stage];
    });
  }

  setReasoning(level: string): void {
    this.update((d) => (d.reasoning_level = level as ReasoningLevel));
  }

  setSeed(value: string): void {
    this.update((d) => (d.ollama_seed = value.trim() === '' ? null : Number(value)));
  }

  setLimit(key: 'max_tasks' | 'max_tool_iterations', value: string): void {
    this.update((d) => (d.limits[key] = Number(value)));
  }

  allowsAllTools(): boolean {
    return this.config().limits.allowed_tools === null;
  }

  setAllTools(all: boolean): void {
    this.update((d) => (d.limits.allowed_tools = all ? null : [...REVISION_TOOL_NAMES]));
  }

  toolAllowed(tool: string): boolean {
    const allowed = this.config().limits.allowed_tools;
    return allowed === null || allowed.includes(tool);
  }

  toggleTool(tool: string, checked: boolean): void {
    this.update((d) => {
      const rest = (d.limits.allowed_tools ?? []).filter((name) => name !== tool);
      d.limits.allowed_tools = checked ? [...rest, tool] : rest;
    });
  }
}
