// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { CommonModule } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';

import {
  AdvancedRuntimeSettings,
  DataRuntimeSettings,
  GeneralRuntimeSettings,
  IntegrationRuntimeSettings,
  MatchingRuntimeSettings,
  RuntimeSettingsCategory,
  RuntimeSettingsUpdateRequest,
  RuntimeSettingsValues,
} from '../../../core/models/settings';
import { RuntimeSettingsStateService } from '../../../core/state/settings-state.service';
import {
  StatusMessageComponent,
  resolveStatusTone,
} from '../../../components/status-message/status-message.component';
import { formatUnknownError } from '../../../core/utils';
import { formatAppDateTime } from '../../../core/utils/date-formatting';

type RuntimeCategorySettings =
  | GeneralRuntimeSettings
  | DataRuntimeSettings
  | IntegrationRuntimeSettings
  | MatchingRuntimeSettings
  | AdvancedRuntimeSettings;

type SettingFieldKind = 'number' | 'boolean' | 'email' | 'text' | 'select';

type SettingOption = {
  value: string;
  label: string;
};

type SettingFieldDescriptor = {
  key: string;
  label: string;
  help: string;
  scope: string;
  kind: SettingFieldKind;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: readonly SettingOption[];
};

type SettingsSectionDefinition = {
  title: string;
  description: string;
  fields: readonly SettingFieldDescriptor[];
};

export const SETTINGS_SECTION_DEFINITIONS: Record<
  RuntimeSettingsCategory,
  SettingsSectionDefinition
> = {
  general: {
    title: 'General',
    description: 'Application-wide runtime behavior applied to new background work.',
    fields: [
      {
        key: 'polling_interval',
        label: 'Job polling interval',
        help: 'How often newly started background jobs ask clients to poll for status.',
        scope: 'Jobs',
        kind: 'number',
        min: 0.25,
        max: 60,
        step: 0.25,
        unit: 'seconds',
      },
    ],
  },
  data: {
    title: 'Data Processing',
    description: 'Ingestion, pipeline concurrency, and clinical language detection thresholds.',
    fields: [
      { key: 'drug_name_min_length', label: 'Minimum drug-name length', help: 'Shortest normalized drug name accepted during ingestion.', scope: 'Ingestion', kind: 'number', min: 1, max: 200, step: 1, unit: 'characters' },
      { key: 'drug_name_max_length', label: 'Maximum drug-name length', help: 'Longest normalized drug name accepted during ingestion.', scope: 'Ingestion', kind: 'number', min: 1, max: 1000, step: 1, unit: 'characters' },
      { key: 'drug_name_max_tokens', label: 'Maximum drug-name tokens', help: 'Maximum whitespace-delimited tokens in a drug-name candidate.', scope: 'Ingestion', kind: 'number', min: 1, max: 64, step: 1, unit: 'tokens' },
      { key: 'text_extraction_batch_size', label: 'Text extraction batch size', help: 'Number of extraction items processed together.', scope: 'Session pipeline', kind: 'number', min: 1, max: 128, step: 1, unit: 'items' },
      { key: 'text_extraction_max_concurrency', label: 'Text extraction concurrency', help: 'Maximum extraction tasks running at the same time.', scope: 'Session pipeline', kind: 'number', min: 1, max: 64, step: 1, unit: 'tasks' },
      { key: 'retrieval_batch_size', label: 'Retrieval batch size', help: 'Number of retrieval items processed together.', scope: 'Session pipeline', kind: 'number', min: 1, max: 128, step: 1, unit: 'items' },
      { key: 'retrieval_max_concurrency', label: 'Retrieval concurrency', help: 'Maximum retrieval tasks running at the same time.', scope: 'Session pipeline', kind: 'number', min: 1, max: 64, step: 1, unit: 'tasks' },
      { key: 'clinical_assessment_batch_size', label: 'Clinical assessment batch size', help: 'Number of clinical assessment items processed together.', scope: 'Session pipeline', kind: 'number', min: 1, max: 128, step: 1, unit: 'items' },
      { key: 'clinical_assessment_max_concurrency', label: 'Clinical assessment concurrency', help: 'Maximum clinical assessment tasks running at the same time.', scope: 'Session pipeline', kind: 'number', min: 1, max: 64, step: 1, unit: 'tasks' },
      { key: 'min_best_score', label: 'Minimum language score', help: 'Minimum score required before a language candidate is considered.', scope: 'Language detection', kind: 'number', min: 0, max: 100, step: 0.1, unit: 'score' },
      { key: 'high_confidence_min_score', label: 'High-confidence score', help: 'Minimum score for a high-confidence language result.', scope: 'Language detection', kind: 'number', min: 0, max: 100, step: 0.1, unit: 'score' },
      { key: 'high_confidence_min_margin', label: 'High-confidence margin', help: 'Minimum lead over the second-ranked language for high confidence.', scope: 'Language detection', kind: 'number', min: 0, max: 100, step: 0.1, unit: 'score' },
      { key: 'moderate_confidence_min_score', label: 'Moderate-confidence score', help: 'Minimum score for a moderate-confidence language result.', scope: 'Language detection', kind: 'number', min: 0, max: 100, step: 0.1, unit: 'score' },
      { key: 'moderate_confidence_min_margin', label: 'Moderate-confidence margin', help: 'Minimum lead over the second-ranked language for moderate confidence.', scope: 'Language detection', kind: 'number', min: 0, max: 100, step: 0.1, unit: 'score' },
    ],
  },
  integrations: {
    title: 'Integrations',
    description: 'Non-secret runtime controls for LiverTox and RxNav operations.',
    fields: [
      { key: 'ncbi_contact_email', label: 'NCBI developer contact email', help: 'Contact address sent with NCBI automated requests. Use a valid developer or organization email registered with NCBI for production use. The bundled .local value is only a compatibility fallback.', scope: 'NCBI / LiverTox', kind: 'email' },
      { key: 'livertox_download_timeout', label: 'LiverTox download timeout', help: 'Maximum wait for a LiverTox download request.', scope: 'LiverTox', kind: 'number', min: 1, max: 3600, step: 1, unit: 'seconds' },
      { key: 'livertox_archive', label: 'LiverTox archive name', help: 'Archive filename used for the LiverTox source bundle.', scope: 'LiverTox', kind: 'text' },
      { key: 'livertox_yield_interval', label: 'LiverTox yield interval', help: 'Progress interval used while processing LiverTox records.', scope: 'LiverTox', kind: 'number', min: 1, max: 10000, step: 1, unit: 'records' },
      { key: 'livertox_skip_deterministic_ratio', label: 'Deterministic-skip ratio', help: 'Share of records allowed to skip model-assisted processing.', scope: 'LiverTox', kind: 'number', min: 0, max: 1, step: 0.01, unit: 'ratio' },
      { key: 'livertox_monograph_max_workers', label: 'LiverTox worker count', help: 'Maximum workers used for monograph processing.', scope: 'LiverTox', kind: 'number', min: 1, max: 64, step: 1, unit: 'workers' },
      { key: 'rxnav_request_timeout', label: 'RxNav request timeout', help: 'Maximum wait for an individual RxNav request.', scope: 'RxNav', kind: 'number', min: 1, max: 120, step: 1, unit: 'seconds' },
      { key: 'rxnav_max_concurrency', label: 'RxNav maximum concurrency', help: 'Maximum number of concurrent RxNav requests.', scope: 'RxNav', kind: 'number', min: 1, max: 64, step: 1, unit: 'requests' },
    ],
  },
  matching: {
    title: 'Drug Matching',
    description: 'Confidence thresholds, cache sizes, and spelling behavior for new matching work.',
    fields: [
      { key: 'direct_confidence', label: 'Direct-match confidence', help: 'Confidence assigned to an exact catalog match.', scope: 'Matcher', kind: 'number', min: 0, max: 1, step: 0.01, unit: 'score' },
      { key: 'master_confidence', label: 'Master-list confidence', help: 'Confidence assigned to a trusted master-list match.', scope: 'Matcher', kind: 'number', min: 0, max: 1, step: 0.01, unit: 'score' },
      { key: 'synonym_confidence', label: 'Synonym confidence', help: 'Confidence assigned to a synonym match.', scope: 'Matcher', kind: 'number', min: 0, max: 1, step: 0.01, unit: 'score' },
      { key: 'min_confidence', label: 'Minimum accepted confidence', help: 'Lowest match confidence accepted by repository queries.', scope: 'Matcher', kind: 'number', min: 0, max: 1, step: 0.01, unit: 'score' },
      { key: 'normalization_cache_limit', label: 'Normalization cache size', help: 'Maximum normalized-name entries retained in memory.', scope: 'Matcher', kind: 'number', min: 1, max: 2000000, step: 1, unit: 'entries' },
      { key: 'match_cache_limit', label: 'Match cache size', help: 'Maximum resolved match entries retained in memory.', scope: 'Matcher', kind: 'number', min: 1, max: 2000000, step: 1, unit: 'entries' },
      { key: 'alias_cache_limit', label: 'Alias cache size', help: 'Maximum alias entries retained in memory.', scope: 'Matcher', kind: 'number', min: 1, max: 2000000, step: 1, unit: 'entries' },
      { key: 'token_min_length', label: 'Minimum token length', help: 'Shortest token considered during name matching.', scope: 'Matcher', kind: 'number', min: 1, max: 128, step: 1, unit: 'characters' },
      { key: 'catalog_index_limit', label: 'Catalog index limit', help: 'Maximum catalog entries loaded into the matching index.', scope: 'Matcher', kind: 'number', min: 1, max: 2000000, step: 1, unit: 'entries' },
      { key: 'spelling_confidence', label: 'Spelling confidence', help: 'Confidence assigned to an authoritative spelling candidate.', scope: 'Spelling', kind: 'number', min: 0, max: 1, step: 0.01, unit: 'score' },
      { key: 'spelling_min_query_length', label: 'Minimum spelling query length', help: 'Shortest query eligible for spelling correction.', scope: 'Spelling', kind: 'number', min: 1, max: 128, step: 1, unit: 'characters' },
      { key: 'spelling_short_name_length', label: 'Short-name threshold', help: 'Names at or below this length use the short-name distance rule.', scope: 'Spelling', kind: 'number', min: 1, max: 128, step: 1, unit: 'characters' },
      { key: 'spelling_short_max_distance', label: 'Short-name edit distance', help: 'Maximum edit distance for short names.', scope: 'Spelling', kind: 'number', min: 0, max: 16, step: 1, unit: 'edits' },
      { key: 'spelling_long_max_distance', label: 'Long-name edit distance', help: 'Maximum edit distance for long names.', scope: 'Spelling', kind: 'number', min: 0, max: 16, step: 1, unit: 'edits' },
    ],
  },
  advanced: {
    title: 'Advanced',
    description: 'Technical runtime budgets resolved dynamically for newly created model and evidence work.',
    fields: [
      { key: 'default_llm_timeout', label: 'Default LLM timeout', help: 'Base timeout used when a more specific budget does not apply.', scope: 'LLM runtime', kind: 'number', min: 1, max: 86400, step: 1, unit: 'seconds' },
      { key: 'parser_llm_timeout', label: 'Parser LLM timeout', help: 'Timeout budget used by new parser work.', scope: 'LLM runtime', kind: 'number', min: 1, max: 86400, step: 1, unit: 'seconds' },
      { key: 'disease_llm_timeout', label: 'Disease LLM timeout', help: 'Timeout budget used by new disease extraction work.', scope: 'LLM runtime', kind: 'number', min: 1, max: 86400, step: 1, unit: 'seconds' },
      { key: 'clinical_llm_timeout', label: 'Clinical LLM timeout', help: 'Timeout budget used by new clinical reasoning work.', scope: 'LLM runtime', kind: 'number', min: 1, max: 86400, step: 1, unit: 'seconds' },
      { key: 'livertox_llm_timeout', label: 'LiverTox LLM timeout', help: 'Timeout budget used by new LiverTox-assisted extraction work.', scope: 'LLM runtime', kind: 'number', min: 1, max: 86400, step: 1, unit: 'seconds' },
      { key: 'minimum_llm_timeout', label: 'Minimum LLM timeout', help: 'Lower bound applied when runtime timeout budgets are resolved.', scope: 'LLM runtime', kind: 'number', min: 1, max: 3600, step: 1, unit: 'seconds' },
      { key: 'cloud_llm_timeout_cap', label: 'Cloud LLM timeout cap', help: 'Upper timeout budget applied to new cloud-model operations.', scope: 'LLM runtime', kind: 'number', min: 1, max: 86400, step: 1, unit: 'seconds' },
      { key: 'local_llm_timeout_cap', label: 'Local LLM timeout cap', help: 'Upper timeout budget applied to new local-model operations.', scope: 'LLM runtime', kind: 'number', min: 1, max: 86400, step: 1, unit: 'seconds' },
      { key: 'ollama_server_start_timeout', label: 'Ollama startup timeout', help: 'Maximum wait for a newly started Ollama server.', scope: 'Ollama runtime', kind: 'number', min: 1, max: 3600, step: 1, unit: 'seconds' },
      { key: 'max_excerpt_length', label: 'Maximum evidence excerpt length', help: 'Maximum excerpt size used by runtime evidence processing.', scope: 'Evidence', kind: 'number', min: 500, max: 100000, step: 100, unit: 'characters' },
    ],
  },
};

function isRuntimeSettingsCategory(value: unknown): value is RuntimeSettingsCategory {
  return value === 'general'
    || value === 'data'
    || value === 'integrations'
    || value === 'matching'
    || value === 'advanced';
}

function categoryRecord(
  values: RuntimeSettingsValues,
  category: RuntimeSettingsCategory,
): Record<string, number | string | boolean> {
  return values[category] as unknown as Record<string, number | string | boolean>;
}

function isValidContactEmail(value: unknown): boolean {
  if (typeof value !== 'string') return false;
  const normalized = value.trim();
  if (normalized.length === 0 || normalized.length > 254 || /\s/.test(normalized)) return false;
  if (normalized.indexOf('@') !== normalized.lastIndexOf('@')) return false;
  const separator = normalized.indexOf('@');
  return separator > 0 && separator < normalized.length - 1;
}

function cloneValues(values: RuntimeSettingsValues): RuntimeSettingsValues {
  return {
    general: { ...values.general },
    data: { ...values.data },
    integrations: { ...values.integrations },
    matching: { ...values.matching },
    advanced: { ...values.advanced },
  };
}

export function validateRuntimeSettingsSection(
  category: RuntimeSettingsCategory,
  values: RuntimeSettingsValues,
): string {
  const record = categoryRecord(values, category);
  for (const field of SETTINGS_SECTION_DEFINITIONS[category].fields) {
    const value = record[field.key];
    if (field.kind === 'number') {
      if (typeof value !== 'number' || !Number.isFinite(value)) return `${field.label} must be a number.`;
      if ((field.min !== undefined && value < field.min) || (field.max !== undefined && value > field.max)) {
        return `${field.label} is outside its allowed range.`;
      }
    } else if (field.kind === 'email' && !isValidContactEmail(value)) {
      return `${field.label} must be a valid email address.`;
    } else if (field.kind === 'text' && (typeof value !== 'string' || !value.trim())) {
      return `${field.label} cannot be empty.`;
    }
  }

  if (category === 'data' && values.data.drug_name_max_length < values.data.drug_name_min_length) {
    return 'Maximum drug-name length cannot be smaller than minimum drug-name length.';
  }
  if (category === 'data' && values.data.high_confidence_min_score < values.data.moderate_confidence_min_score) {
    return 'High-confidence score cannot be smaller than moderate-confidence score.';
  }
  if (category === 'data' && values.data.high_confidence_min_margin < values.data.moderate_confidence_min_margin) {
    return 'High-confidence margin cannot be smaller than moderate-confidence margin.';
  }
  if (category === 'advanced' && values.advanced.cloud_llm_timeout_cap < values.advanced.minimum_llm_timeout) {
    return 'Cloud LLM timeout cap cannot be smaller than the minimum timeout.';
  }
  if (category === 'advanced' && values.advanced.local_llm_timeout_cap < values.advanced.minimum_llm_timeout) {
    return 'Local LLM timeout cap cannot be smaller than the minimum timeout.';
  }
  return '';
}

@Component({
  selector: 'app-operational-settings-page',
  standalone: true,
  imports: [CommonModule, FormsModule, StatusMessageComponent],
  templateUrl: './operational-settings-page.component.html',
  styleUrl: './operational-settings-page.component.scss',
})
export class OperationalSettingsPageComponent implements OnInit {
  readonly settingsState = inject(RuntimeSettingsStateService);
  private readonly route = inject(ActivatedRoute);

  readonly section = signal<RuntimeSettingsCategory>('general');
  readonly draft = signal<RuntimeSettingsValues | null>(null);
  readonly isSaving = signal(false);
  readonly isResetting = signal(false);
  readonly statusMessage = signal('');
  readonly statusTone = computed(() => resolveStatusTone(this.statusMessage()));
  readonly definition = computed(() => SETTINGS_SECTION_DEFINITIONS[this.section()]);
  readonly fields = computed(() => this.definition().fields);
  readonly validationMessage = computed(() => {
    const values = this.draft();
    return values ? validateRuntimeSettingsSection(this.section(), values) : '';
  });
  readonly isDirty = computed(() => {
    const saved = this.settingsState.data()?.values;
    const draft = this.draft();
    if (!saved || !draft) return false;
    return JSON.stringify(saved[this.section()]) !== JSON.stringify(draft[this.section()]);
  });
  readonly canSave = computed(
    () => this.isDirty() && !this.isSaving() && !this.isResetting() && !this.validationMessage(),
  );
  readonly updatedAtLabel = computed(() => {
    const updatedAt = this.settingsState.data()?.updated_at;
    return updatedAt ? formatAppDateTime(updatedAt, updatedAt) : 'Not available';
  });

  constructor() {
    this.route.data
      .pipe(takeUntilDestroyed())
      .subscribe((data) => {
        const nextSection = data['settingsSection'];
        if (!isRuntimeSettingsCategory(nextSection)) return;
        this.section.set(nextSection);
        this.statusMessage.set('');
        this.syncDraft();
      });
  }

  async ngOnInit(): Promise<void> {
    await this.loadSettings(false);
  }

  fieldValue(field: SettingFieldDescriptor): number | string | boolean | null {
    const values = this.draft();
    if (!values) return null;
    return categoryRecord(values, this.section())[field.key] ?? null;
  }

  updateField(field: SettingFieldDescriptor, rawValue: number | string | boolean | null): void {
    const current = this.draft();
    if (!current) return;
    let value: number | string | boolean;
    if (field.kind === 'number') {
      value = rawValue === null || rawValue === '' ? Number.NaN : Number(rawValue);
    } else if (field.kind === 'boolean') {
      value = Boolean(rawValue);
    } else {
      value = String(rawValue ?? '');
    }
    const category = this.section();
    const nextCategory = {
      ...categoryRecord(current, category),
      [field.key]: value,
    } as unknown as RuntimeCategorySettings;
    this.draft.set({ ...current, [category]: nextCategory } as RuntimeSettingsValues);
    this.statusMessage.set('');
  }

  discardChanges(): void {
    this.syncDraft();
    this.statusMessage.set('Changes discarded.');
  }

  async retryLoad(): Promise<void> {
    await this.loadSettings(true);
  }

  async save(): Promise<void> {
    const values = this.draft();
    if (!values || !this.canSave()) return;
    const category = this.section();
    const payload = { [category]: { ...values[category] } } as RuntimeSettingsUpdateRequest;

    this.isSaving.set(true);
    try {
      await this.settingsState.update(payload);
      this.syncDraft();
      this.statusMessage.set('Settings saved and applied to new runtime work.');
    } catch (error) {
      this.statusMessage.set(formatUnknownError(error, 'Unable to save settings.'));
    } finally {
      this.isSaving.set(false);
    }
  }

  async resetToDefaults(): Promise<void> {
    if (this.isSaving() || this.isResetting()) return;
    this.isResetting.set(true);
    try {
      await this.settingsState.reset(this.section());
      this.syncDraft();
      this.statusMessage.set('Section reset to application defaults.');
    } catch (error) {
      this.statusMessage.set(formatUnknownError(error, 'Unable to reset settings.'));
    } finally {
      this.isResetting.set(false);
    }
  }

  private async loadSettings(force: boolean): Promise<void> {
    try {
      await this.settingsState.load(force);
      this.syncDraft();
      this.statusMessage.set('');
    } catch (error) {
      this.statusMessage.set(formatUnknownError(error, 'Unable to load settings.'));
    }
  }

  private syncDraft(): void {
    const values = this.settingsState.data()?.values;
    if (values) this.draft.set(cloneValues(values));
  }
}
