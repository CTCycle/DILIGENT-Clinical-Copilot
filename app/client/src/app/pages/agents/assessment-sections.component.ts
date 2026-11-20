// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, input, output } from '@angular/core';

import {
  AgentModelRef,
  AssessmentConfig,
  ReportTemplate,
} from '../../core/models/agent-types';
import { EvidenceSourceId } from '../../core/models/types';
import { AgentModelFieldComponent } from './agent-model-field.component';
import { AgentModelsSectionComponent } from './agent-models-section.component';
import { INSTRUCTION_MAX_LENGTH, ModelCatalog, ProviderOption } from './agent-form';
import { ReportTemplateFieldComponent } from './report-template-field.component';

const EVIDENCE_SOURCES: ReadonlyArray<{ id: EvidenceSourceId; label: string; hint: string }> = [
  { id: 'livertox', label: 'LiverTox', hint: 'NIH drug-induced liver injury monographs.' },
  { id: 'dilirank', label: 'DILIrank', hint: 'FDA DILI concern rankings.' },
  { id: 'documents', label: 'Documents', hint: 'Your uploaded reference documents.' },
];

type NumericRetrievalKey =
  | 'retrieval_candidate_count'
  | 'retrieval_selected_count'
  | 'hybrid_vector_weight'
  | 'hybrid_text_weight';

/** Instructions, models, retrieval and parameters of the DILI Assessment capability. */
@Component({
  selector: 'app-assessment-sections',
  standalone: true,
  imports: [AgentModelFieldComponent, AgentModelsSectionComponent, ReportTemplateFieldComponent],
  styleUrl: './agent-sections.scss',
  templateUrl: './assessment-sections.component.html',
})
export class AssessmentSectionsComponent {
  readonly config = input.required<AssessmentConfig>();
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly disabled = input(false);
  readonly configChange = output<AssessmentConfig>();

  readonly evidenceSources = EVIDENCE_SOURCES;
  readonly limit = INSTRUCTION_MAX_LENGTH;

  readonly weightSum = computed(() => {
    const { hybrid_vector_weight: vector, hybrid_text_weight: text } = this.config().retrieval;
    return (Number(vector) + Number(text)).toFixed(2);
  });

  private update(change: (draft: AssessmentConfig) => void): void {
    const draft = structuredClone(this.config());
    change(draft);
    // Every extraction stage is mandatory in the pipeline and always uses the extraction model.
    draft.models.extraction_overrides = {};
    this.configChange.emit(draft);
  }

  setInstructions(value: string): void {
    this.update((d) => (d.instructions.custom_instructions = value));
  }

  setTemplate(template: ReportTemplate | null): void {
    this.update((d) => (d.report_template = template));
  }

  setModel(role: 'extraction' | 'clinical', ref: AgentModelRef): void {
    this.update((d) => (d.models[role] = ref));
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
