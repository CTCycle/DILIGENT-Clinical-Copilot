// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, inject, input, output, signal } from '@angular/core';

import {
  AgentModelRef,
  AssessmentConfig,
  ExtractionStage,
  ReportTemplate,
} from '../../core/models/agent-types';
import { EvidenceSourceId, ReasoningLevel } from '../../core/models/types';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { AgentModelFieldComponent } from './agent-model-field.component';
import { INSTRUCTION_MAX_LENGTH, ModelCatalog, ProviderOption } from './agent-form';
import { ReportTemplateFieldComponent } from './report-template-field.component';

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
  imports: [AgentModelFieldComponent, ReportTemplateFieldComponent],
  styleUrl: './agent-sections.scss',
  templateUrl: './assessment-sections.component.html',
})
export class AssessmentSectionsComponent {
  private readonly modelConfig = inject(ModelConfigStateService);

  readonly config = input.required<AssessmentConfig>();
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly disabled = input(false);
  readonly configChange = output<AssessmentConfig>();

  readonly extractionStages = EXTRACTION_STAGES;
  readonly evidenceSources = EVIDENCE_SOURCES;
  readonly reasoningLevels = REASONING;
  readonly limit = INSTRUCTION_MAX_LENGTH;

  readonly busy = this.modelConfig.catalogBusy;
  readonly refreshMessage = signal('');
  readonly refreshFailed = signal(false);
  readonly weightSum = computed(() => {
    const { hybrid_vector_weight: vector, hybrid_text_weight: text } = this.config().retrieval;
    return (Number(vector) + Number(text)).toFixed(2);
  });

  /** One refresh for every provider this tab's models use. */
  async refreshModels(): Promise<void> {
    const { models } = this.config();
    const providers = [models.clinical, models.extraction, ...Object.values(models.extraction_overrides)]
      .map((ref) => ref?.provider)
      .filter((provider): provider is string => !!provider);
    this.refreshMessage.set('');
    const failures = await this.modelConfig.refreshCatalogs(providers);
    const failed = Object.entries(failures);
    this.refreshFailed.set(failed.length > 0);
    this.refreshMessage.set(
      failed.length ? failed.map(([provider, error]) => `${provider}: ${error}`).join(' · ') : 'Models updated.',
    );
  }

  private update(change: (draft: AssessmentConfig) => void): void {
    const draft = structuredClone(this.config());
    change(draft);
    this.configChange.emit(draft);
  }

  setInstructions(value: string): void {
    this.update((d) => (d.instructions.custom_instructions = value));
  }

  setTemplate(template: ReportTemplate | null): void {
    this.update((d) => (d.report_template = template));
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
