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
  | AdvancedRuntimeSettings;

type SettingFieldDescriptor = {
  key: string;
  label: string;
  help: string;
  jsonPath: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
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
    description: 'Application-wide runtime behavior that can be changed safely while DILIGENT is running.',
    fields: [
      {
        key: 'polling_interval',
        label: 'Job polling interval',
        help: 'How often newly started background jobs ask clients to poll for status.',
        jsonPath: 'jobs.polling_interval',
        min: 0.25,
        max: 60,
        step: 0.25,
        unit: 'seconds',
      },
    ],
  },
  data: {
    title: 'Data Processing',
    description: 'Validation limits applied to drug names accepted by the runtime knowledge repositories.',
    fields: [
      {
        key: 'drug_name_min_length',
        label: 'Minimum drug-name length',
        help: 'Reject shorter normalized drug names during ingestion and runtime vocabulary updates.',
        jsonPath: 'ingestion.drug_name_min_length',
        min: 1,
        max: 200,
        step: 1,
        unit: 'characters',
      },
      {
        key: 'drug_name_max_length',
        label: 'Maximum drug-name length',
        help: 'Reject longer normalized drug names during ingestion and runtime vocabulary updates.',
        jsonPath: 'ingestion.drug_name_max_length',
        min: 1,
        max: 1000,
        step: 1,
        unit: 'characters',
      },
      {
        key: 'drug_name_max_tokens',
        label: 'Maximum drug-name tokens',
        help: 'Limits whitespace-delimited tokens in accepted drug-name candidates.',
        jsonPath: 'ingestion.drug_name_max_tokens',
        min: 1,
        max: 64,
        step: 1,
        unit: 'tokens',
      },
    ],
  },
  integrations: {
    title: 'Integrations',
    description: 'Non-secret runtime controls for LiverTox and RxNav data operations.',
    fields: [
      {
        key: 'livertox_download_timeout',
        label: 'LiverTox download timeout',
        help: 'Maximum wait for a LiverTox download request before it is treated as failed.',
        jsonPath: 'runtime.livertox_download_timeout',
        min: 1,
        max: 3600,
        step: 1,
        unit: 'seconds',
      },
      {
        key: 'rxnav_request_timeout',
        label: 'RxNav request timeout',
        help: 'Maximum wait for an individual RxNav request.',
        jsonPath: 'runtime.rxnav_request_timeout',
        min: 1,
        max: 120,
        step: 1,
        unit: 'seconds',
      },
      {
        key: 'rxnav_max_concurrency',
        label: 'RxNav maximum concurrency',
        help: 'Maximum number of concurrent RxNav requests.',
        jsonPath: 'runtime.rxnav_max_concurrency',
        min: 1,
        max: 64,
        step: 1,
        unit: 'requests',
      },
    ],
  },
  advanced: {
    title: 'Advanced',
    description: 'Technical runtime limits that are resolved dynamically for new model and evidence work.',
    fields: [
      {
        key: 'default_llm_timeout',
        label: 'Default LLM timeout',
        help: 'Base timeout used by runtime model clients when a more specific budget does not apply.',
        jsonPath: 'runtime.default_llm_timeout',
        min: 1,
        max: 86400,
        step: 1,
        unit: 'seconds',
      },
      {
        key: 'clinical_llm_timeout',
        label: 'Clinical LLM timeout',
        help: 'Timeout budget used by clinical reasoning services created after this value is saved.',
        jsonPath: 'runtime.clinical_llm_timeout',
        min: 1,
        max: 86400,
        step: 1,
        unit: 'seconds',
      },
      {
        key: 'livertox_llm_timeout',
        label: 'LiverTox LLM timeout',
        help: 'Timeout budget used by new LiverTox-assisted extraction work.',
        jsonPath: 'runtime.livertox_llm_timeout',
        min: 1,
        max: 86400,
        step: 1,
        unit: 'seconds',
      },
      {
        key: 'minimum_llm_timeout',
        label: 'Minimum LLM timeout',
        help: 'Lower bound applied when runtime timeout budgets are resolved.',
        jsonPath: 'runtime.minimum_llm_timeout',
        min: 1,
        max: 3600,
        step: 1,
        unit: 'seconds',
      },
      {
        key: 'cloud_llm_timeout_cap',
        label: 'Cloud LLM timeout cap',
        help: 'Upper timeout budget applied to new cloud-model operations.',
        jsonPath: 'runtime.cloud_llm_timeout_cap',
        min: 1,
        max: 86400,
        step: 1,
        unit: 'seconds',
      },
      {
        key: 'local_llm_timeout_cap',
        label: 'Local LLM timeout cap',
        help: 'Upper timeout budget applied to new local-model operations.',
        jsonPath: 'runtime.local_llm_timeout_cap',
        min: 1,
        max: 86400,
        step: 1,
        unit: 'seconds',
      },
      {
        key: 'max_excerpt_length',
        label: 'Maximum evidence excerpt length',
        help: 'Maximum excerpt size used by runtime evidence processing.',
        jsonPath: 'runtime.max_excerpt_length',
        min: 500,
        max: 100000,
        step: 100,
        unit: 'characters',
      },
    ],
  },
};

function isRuntimeSettingsCategory(value: unknown): value is RuntimeSettingsCategory {
  return value === 'general' || value === 'data' || value === 'integrations' || value === 'advanced';
}

function categoryRecord(
  values: RuntimeSettingsValues,
  category: RuntimeSettingsCategory,
): Record<string, number> {
  return values[category] as unknown as Record<string, number>;
}

function cloneValues(values: RuntimeSettingsValues): RuntimeSettingsValues {
  return {
    general: { ...values.general },
    data: { ...values.data },
    integrations: { ...values.integrations },
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
    if (!Number.isFinite(value)) return `${field.label} must be a number.`;
    if (value < field.min || value > field.max) {
      return `${field.label} must be between ${field.min} and ${field.max}.`;
    }
  }

  if (category === 'data' && values.data.drug_name_max_length < values.data.drug_name_min_length) {
    return 'Maximum drug-name length cannot be smaller than minimum drug-name length.';
  }
  if (
    category === 'advanced'
    && values.advanced.cloud_llm_timeout_cap < values.advanced.minimum_llm_timeout
  ) {
    return 'Cloud LLM timeout cap cannot be smaller than the minimum timeout.';
  }
  if (
    category === 'advanced'
    && values.advanced.local_llm_timeout_cap < values.advanced.minimum_llm_timeout
  ) {
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

  fieldValue(field: SettingFieldDescriptor): number | null {
    const values = this.draft();
    if (!values) return null;
    return categoryRecord(values, this.section())[field.key] ?? null;
  }

  updateField(field: SettingFieldDescriptor, rawValue: number | string | null): void {
    const current = this.draft();
    if (!current) return;
    const numericValue = rawValue === null || rawValue === '' ? Number.NaN : Number(rawValue);
    const category = this.section();
    const nextCategory = {
      ...categoryRecord(current, category),
      [field.key]: numericValue,
    } as unknown as RuntimeCategorySettings;
    this.draft.set({
      ...current,
      [category]: nextCategory,
    } as RuntimeSettingsValues);
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
    const payload = {
      [category]: { ...values[category] },
    } as RuntimeSettingsUpdateRequest;

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
