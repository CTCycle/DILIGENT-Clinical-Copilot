// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output } from '@angular/core';

import {
  AgentModelRef,
  AssessmentConfig,
  AssessmentInstructions,
  ExtractionStage,
} from '../../core/models/agent-types';
import { EvidenceSourceId, ReasoningLevel } from '../../core/models/types';
import { AgentModelFieldComponent } from './agent-model-field.component';
import { INSTRUCTION_MAX_LENGTH, ModelCatalog, ProviderOption } from './agent-form';

const INSTRUCTION_FIELDS: ReadonlyArray<{ key: keyof AssessmentInstructions; label: string }> = [
  { key: 'assessment_focus', label: 'Assessment focus' },
  { key: 'report_style', label: 'Report style' },
  { key: 'extraction_guidance', label: 'Extraction guidance' },
];
const EXTRACTION_STAGES: ReadonlyArray<{ stage: ExtractionStage; label: string }> = [
  { stage: 'therapy_drugs', label: 'Therapy drugs' },
  { stage: 'anamnesis_drugs', label: 'Anamnesis drugs' },
  { stage: 'diseases', label: 'Diseases' },
  { stage: 'laboratory', label: 'Laboratory' },
];
const EVIDENCE_SOURCES: ReadonlyArray<{ id: EvidenceSourceId; label: string }> = [
  { id: 'livertox', label: 'LiverTox' },
  { id: 'dilirank', label: 'DILIrank' },
  { id: 'documents', label: 'Documents' },
];
const REASONING: readonly ReasoningLevel[] = ['off', 'low', 'medium', 'high'];

type NumericRetrievalKey =
  | 'retrieval_candidate_count'
  | 'retrieval_selected_count'
  | 'hybrid_vector_weight'
  | 'hybrid_text_weight';

/** Instructions, models, retrieval and parameters of the DILI Assessment capability. */
@Component({
  selector: 'app-assessment-sections',
  standalone: true,
  imports: [AgentModelFieldComponent],
  styleUrl: './agent-sections.scss',
  templateUrl: './assessment-sections.component.html',
})
export class AssessmentSectionsComponent {
  readonly config = input.required<AssessmentConfig>();
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly disabled = input(false);
  readonly configChange = output<AssessmentConfig>();

  readonly instructionFields = INSTRUCTION_FIELDS;
  readonly extractionStages = EXTRACTION_STAGES;
  readonly evidenceSources = EVIDENCE_SOURCES;
  readonly reasoningLevels = REASONING;
  readonly limit = INSTRUCTION_MAX_LENGTH;

  private update(change: (draft: AssessmentConfig) => void): void {
    const draft = structuredClone(this.config());
    change(draft);
    this.configChange.emit(draft);
  }

  setInstruction(key: keyof AssessmentInstructions, value: string): void {
    this.update((d) => (d.instructions[key] = value));
  }

  setModel(role: 'extraction' | 'clinical', ref: AgentModelRef | null): void {
    if (ref) this.update((d) => (d.models[role] = ref));
  }

  setOverride(stage: ExtractionStage, ref: AgentModelRef | null): void {
    this.update((d) => {
      if (ref) d.models.extraction_overrides[stage] = ref;
      else delete d.models.extraction_overrides[stage];
    });
  }

  setReasoning(level: string): void {
    this.update((d) => (d.reasoning_level = level as ReasoningLevel));
  }

  setSeed(value: string): void {
    this.update((d) => (d.ollama_seed = value.trim() === '' ? null : Number(value)));
  }

  toggleSource(id: EvidenceSourceId, checked: boolean): void {
    this.update((d) => {
      const rest = d.retrieval.evidence_sources.filter((source) => source !== id);
      d.retrieval.evidence_sources = checked ? [...rest, id] : rest;
    });
  }

  setFlag(key: 'use_hybrid_search' | 'use_reranking', value: boolean): void {
    this.update((d) => (d.retrieval[key] = value));
  }

  setReranker(value: string): void {
    this.update((d) => (d.retrieval.reranker_model = value));
  }

  setNumber(key: NumericRetrievalKey, value: string): void {
    this.update((d) => (d.retrieval[key] = Number(value)));
  }
}
