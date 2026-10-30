// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, effect, inject, input, output, signal, untracked } from '@angular/core';

import { ModalShellComponent } from '../../../../components/modal-shell/modal-shell.component';
import {
  DEFAULT_EVIDENCE_SOURCES,
  EVIDENCE_SOURCE_OPTIONS,
  describeEvidenceSources,
} from '../../../../core/evidence-sources';
import {
  ClinicalInputPreflightIssue,
  ClinicalInputPreflightResponse,
  ClinicalRequestPayload,
  EvidenceSourceId,
  EvidenceSourceStatus,
  RuntimeSettings,
  SessionModelOverrides,
} from '../../../../core/models/types';
import { validateClinicalInput } from '../../../../core/services/clinical-api';
import { DiliJobTrackerService } from '../../../../core/services/dili-job-tracker.service';
import { AppStateService } from '../../../../core/state/app-state.service';
import { ModelConfigStateService } from '../../../../core/state/model-config-state.service';
import { buildClinicalPayload } from '../../../../core/utils';

export type WizardStepId = 'preflight' | 'sources' | 'models' | 'review';
type ModelRole = 'text_extraction_model' | 'clinical_model';
type ModelOverrideDraft = Record<ModelRole, string | null>;

export const WIZARD_STEPS: ReadonlyArray<{ id: WizardStepId; label: string }> = [
  { id: 'preflight', label: 'Preflight' },
  { id: 'sources', label: 'Evidence Sources' },
  { id: 'models', label: 'Models' },
  { id: 'review', label: 'Review' },
];

export const LIVERTOX_WARNING =
  'Excluding LiverTox removes drug-level hepatotoxicity evidence from the causality assessment and may reduce accuracy.';

const NO_OVERRIDES: ModelOverrideDraft = { text_extraction_model: null, clinical_model: null };

type PreflightRecord = {
  result: ClinicalInputPreflightResponse;
  formKey: string;
  configKey: string;
};

function preflightUnavailable(): ClinicalInputPreflightResponse {
  const description =
    'The application could not complete the safety checks. Verify backend connectivity and runtime configuration, then retry.';
  return {
    ready: false,
    blocking_issues: [
      {
        severity: 'blocking',
        code: 'preflight_request_failed',
        message: description,
        field: null,
        title: 'Pre-flight checks are unavailable',
        description,
        affected_section: 'Analysis readiness',
        consequence: 'The analysis cannot start until the safety checks complete successfully.',
        continuation_allowed: false,
      },
    ],
    non_blocking_issues: [],
    runtime_settings: {},
    extraction_quality: {},
    deterministic_diagnostics: {},
    rag_readiness: null,
  };
}

@Component({
  selector: 'app-analysis-wizard',
  imports: [ModalShellComponent],
  templateUrl: './analysis-wizard.component.html',
  styleUrl: './analysis-wizard.component.scss',
})
export class AnalysisWizardComponent {
  readonly isOpen = input(false);
  readonly cancelled = output<void>();
  readonly started = output<void>();
  readonly fixInput = output<string | null>();

  private readonly stateService = inject(AppStateService);
  private readonly modelConfigState = inject(ModelConfigStateService);
  private readonly tracker = inject(DiliJobTrackerService);

  readonly steps = WIZARD_STEPS;
  readonly sourceOptions = EVIDENCE_SOURCE_OPTIONS;
  readonly livertoxWarning = LIVERTOX_WARNING;

  readonly step = signal<WizardStepId>('preflight');
  readonly sources = signal<readonly EvidenceSourceId[]>(DEFAULT_EVIDENCE_SOURCES);
  readonly overrides = signal<ModelOverrideDraft>(NO_OVERRIDES);
  readonly record = signal<PreflightRecord | null>(null);
  readonly checking = signal(false);
  readonly starting = signal(false);
  readonly startError = signal<string | null>(null);

  private checkAttempt = 0;

  readonly form = computed(() => this.stateService.state().diliAgent.form);
  readonly settings = computed(() => this.modelConfigState.settings());
  readonly stepIndex = computed(() => this.steps.findIndex((item) => item.id === this.step()));
  readonly formKey = computed(() => JSON.stringify(this.form()));
  readonly modelOverrides = computed<SessionModelOverrides | null>(() => {
    const draft = this.overrides();
    const active: SessionModelOverrides = {};
    if (draft.text_extraction_model) active.text_extraction_model = draft.text_extraction_model;
    if (draft.clinical_model) active.clinical_model = draft.clinical_model;
    return Object.keys(active).length ? active : null;
  });
  readonly configKey = computed(() =>
    JSON.stringify([[...this.sources()].sort(), this.modelOverrides()]),
  );

  private readonly formCurrent = computed(() => this.record()?.formKey === this.formKey());
  readonly result = computed(() => (this.formCurrent() ? this.record()!.result : null));
  readonly issues = computed<ClinicalInputPreflightIssue[]>(() => {
    const result = this.result();
    return result ? [...result.blocking_issues, ...result.non_blocking_issues] : [];
  });
  readonly blockingCount = computed(() => this.result()?.blocking_issues.length ?? 0);
  readonly warningCount = computed(() => this.result()?.non_blocking_issues.length ?? 0);
  readonly inputReady = computed(() => this.result() !== null && this.blockingCount() === 0);
  private readonly onlyConfigurationIssues = computed(() => {
    const blocking = this.result()?.blocking_issues ?? [];
    return blocking.length > 0 && blocking.every((issue) => issue.field && issue.field in CONFIGURATION_STEPS);
  });
  readonly checkCurrent = computed(
    () => this.formCurrent() && this.record()?.configKey === this.configKey(),
  );
  readonly canStart = computed(
    () => this.inputReady() && this.checkCurrent() && !this.checking() && !this.starting(),
  );
  readonly livertoxExcluded = computed(() => !this.sources().includes('livertox'));
  readonly sourcesSummary = computed(() => describeEvidenceSources(this.sources()));
  readonly sourceStatuses = computed(() => {
    const entries = this.result()?.evidence_sources ?? [];
    return new Map<EvidenceSourceId, EvidenceSourceStatus>(entries.map((entry) => [entry.source, entry]));
  });

  readonly provider = computed(() => {
    const settings = this.settings();
    if (!settings) return '';
    return settings.useCloudServices ? settings.provider : 'Ollama (local)';
  });
  readonly modelRoles = computed(() => {
    const settings = this.settings();
    const draft = this.overrides();
    const options = this.modelOptions(settings);
    const rows: Array<{ role: ModelRole; label: string }> = [
      { role: 'text_extraction_model', label: 'Text extraction model' },
      { role: 'clinical_model', label: 'Clinical model' },
    ];
    return rows.map(({ role, label }) => {
      const defaultModel = this.defaultModel(settings, role);
      const effective = draft[role] ?? defaultModel;
      return {
        role,
        label,
        defaultModel,
        effective,
        overridden: draft[role] !== null,
        options: options.includes(effective) ? options : [effective, ...options].filter(Boolean),
      };
    });
  });
  readonly hasOverrides = computed(() => this.modelOverrides() !== null);

  constructor() {
    effect(() => {
      if (this.isOpen()) {
        untracked(() => void this.modelConfigState.load().catch(() => undefined));
      }
    });
    effect(() => {
      const needsCheck = this.isOpen()
        && !this.checkCurrent()
        && !this.checking()
        && (this.step() === 'preflight' || this.step() === 'review')
        && this.settings() !== null;
      if (needsCheck) {
        untracked(() => void this.runPreflight());
      }
    });
  }

  reset(): void {
    this.checkAttempt += 1;
    this.sources.set(DEFAULT_EVIDENCE_SOURCES);
    this.overrides.set(NO_OVERRIDES);
    this.record.set(null);
    this.checking.set(false);
    this.starting.set(false);
    this.startError.set(null);
    this.step.set('preflight');
  }

  canVisit(index: number): boolean {
    return index <= this.stepIndex() || this.inputReady() || this.onlyConfigurationIssues();
  }

  goTo(index: number): void {
    if (this.canVisit(index)) {
      this.step.set(this.steps[index].id);
    }
  }

  next(): void {
    this.goTo(Math.min(this.stepIndex() + 1, this.steps.length - 1));
  }

  back(): void {
    this.goTo(Math.max(this.stepIndex() - 1, 0));
  }

  toggleSource(source: EvidenceSourceId, checked: boolean): void {
    this.startError.set(null);
    this.sources.update((current) => {
      const next = current.filter((item) => item !== source);
      return checked ? [...next, source] : next;
    });
  }

  setModel(role: ModelRole, value: string): void {
    const defaultModel = this.defaultModel(this.settings(), role);
    this.overrides.update((current) => ({ ...current, [role]: value && value !== defaultModel ? value : null }));
  }

  restoreModelDefaults(): void {
    this.overrides.set(NO_OVERRIDES);
  }

  async runPreflight(): Promise<void> {
    const attempt = ++this.checkAttempt;
    const formKey = this.formKey();
    const configKey = this.configKey();
    this.checking.set(true);
    let result: ClinicalInputPreflightResponse;
    try {
      const payload = await this.buildPayload();
      result = await validateClinicalInput(payload);
    } catch (error) {
      console.error('Clinical pre-flight request failed.', error);
      result = preflightUnavailable();
    }
    if (attempt !== this.checkAttempt) {
      return;
    }
    this.record.set({ result, formKey, configKey });
    this.checking.set(false);
  }

  fixInInput(): void {
    const fields = this.issues().map((issue) => issue.field ?? '');
    const configurationField = fields.find((field) => field in CONFIGURATION_STEPS);
    if (configurationField) {
      this.step.set(CONFIGURATION_STEPS[configurationField]);
      return;
    }
    this.fixInput.emit(fields.map((field) => PREFLIGHT_FIELD_TARGETS[field]).find(Boolean) ?? null);
  }

  async start(): Promise<void> {
    if (!this.canStart()) {
      return;
    }
    this.starting.set(true);
    this.startError.set(null);
    try {
      await this.tracker.startSession(await this.buildPayload(), null);
    } catch (error) {
      this.starting.set(false);
      this.startError.set(error instanceof Error ? error.message : 'The analysis could not be started.');
      return;
    }
    const vm = this.stateService.state().diliAgent;
    if (vm.isRunning) {
      this.reset();
      this.started.emit();
      return;
    }
    this.starting.set(false);
    this.startError.set(vm.message || 'The analysis could not be started.');
  }

  private async buildPayload(): Promise<ClinicalRequestPayload> {
    let settings = this.settings();
    if (!settings) {
      await this.modelConfigState.load();
      settings = this.settings();
    }
    if (!settings) {
      throw new Error('Persisted model configuration is unavailable.');
    }
    return buildClinicalPayload(this.form(), settings, this.sources(), this.modelOverrides());
  }

  private defaultModel(settings: RuntimeSettings | null, role: ModelRole): string {
    if (!settings) return '';
    return role === 'text_extraction_model' ? settings.textExtractionModel : settings.clinicalModel;
  }

  private modelOptions(settings: RuntimeSettings | null): string[] {
    const data = this.modelConfigState.data();
    if (!settings || !data) return [];
    const names = settings.useCloudServices
      ? (data.cloud_providers.find((provider) => provider.id === settings.provider)?.models ?? []).map((model) => model.id)
      : data.local_models.filter((model) => model.available_in_ollama).map((model) => model.name);
    return [...new Set(names)];
  }
}

export const PREFLIGHT_FIELD_TARGETS: Record<string, string> = {
  anamnesis: 'clinical-input',
  clinical_input: 'clinical-input',
  drugs: 'clinical-input',
  laboratory_analysis: 'clinical-input',
  selected_model_providers: 'run-analysis-button',
  visit_date: 'visit-date',
};

const CONFIGURATION_STEPS: Record<string, WizardStepId> = {
  evidence_sources: 'sources',
  model_overrides: 'models',
};
