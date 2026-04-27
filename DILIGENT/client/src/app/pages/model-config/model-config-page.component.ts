import { CommonModule } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ProviderAccessCardComponent } from '../../components/provider-access-card/provider-access-card.component';
import {
  StatusMessageComponent,
  resolveStatusTone,
} from '../../components/status-message/status-message.component';
import { AppStateService } from '../../core/state/app-state.service';
import { formatUnknownError } from '../../core/utils';
import { CLOUD_MODEL_CHOICES } from '../../core/constants';
import {
  buildRuntimeSettingsFromConfig,
  resolveCloudChoices,
  resolveCloudModel,
  resolveProvider,
} from '../../core/model-config';
import {
  AccessKeyProvider,
  CloudProvider,
  JobStatus,
  ModelConfigStateResponse,
  ModelConfigUpdateRequest,
  RuntimeSettings,
} from '../../core/models/types';
import {
  fetchModelConfigState,
  fetchModelPullJobStatus,
  startModelPullJob,
  updateModelConfigState,
} from '../../core/services/model-config-api';
import { JobPollingService } from '../../core/services/job-polling.service';
import { AccessKeyModalComponent } from './components/access-key-modal.component';
import { ModelRoleActionButtonComponent } from './components/model-role-action-button.component';
import {
  MODEL_FILTERS,
  modelMatchesFilters,
  resolveDraftFromSettings,
} from './model-catalog';
import {
  DraftRuntimeConfig,
  ModelFilterKey,
  ModelPullProgressState,
  ModelRole,
} from './model-config.types';

const DEFAULT_CLOUD_PROVIDERS: readonly CloudProvider[] = ['openai', 'gemini'];

const PROVIDER_LABELS: Record<AccessKeyProvider, string> = {
  openai: 'OpenAI',
  gemini: 'Gemini',
  tavily: 'Tavily',
};

const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ['completed', 'failed', 'cancelled'];

function isCloudProvider(provider: string): provider is CloudProvider {
  return provider === 'openai' || provider === 'gemini';
}

function resolveProviderLabel(provider: string): string {
  if (provider === 'openai' || provider === 'gemini' || provider === 'tavily') {
    return PROVIDER_LABELS[provider];
  }
  return provider;
}

@Component({
  selector: 'app-model-config-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ProviderAccessCardComponent,
    StatusMessageComponent,
    AccessKeyModalComponent,
    ModelRoleActionButtonComponent,
  ],
  templateUrl: './model-config-page.component.html',
  styleUrl: './model-config-page.component.scss',
})
export class ModelConfigPageComponent implements OnInit {
  readonly appState = inject(AppStateService);
  private readonly jobPolling = inject(JobPollingService);

  readonly modelFilters = MODEL_FILTERS;

  readonly isLoading = signal(true);
  readonly hasLoadedConfig = signal(false);
  readonly isSaving = signal(false);
  readonly localModels = signal<ModelConfigStateResponse['local_models']>([]);
  readonly cloudChoices = signal(CLOUD_MODEL_CHOICES);
  readonly modelSearchQuery = signal('');
  readonly statusMessage = signal('');
  readonly openProviderModal = signal<AccessKeyProvider | null>(null);
  readonly modelPullProgress = signal<Record<string, ModelPullProgressState>>({});
  readonly activeFilters = signal<Record<ModelFilterKey, boolean>>({
    installed: false,
    reasoning: false,
    small: false,
    extraction: false,
  });
  readonly draftConfig = signal<DraftRuntimeConfig>(resolveDraftFromSettings(this.appState.state().diliAgent.settings));

  readonly filteredLocalModels = computed(() => {
    const query = this.modelSearchQuery().trim().toLowerCase();
    return this.localModels().filter((model) =>
      modelMatchesFilters(model, query, this.activeFilters()),
    );
  });

  readonly availableLocalModelCount = computed(
    () => this.localModels().filter((model) => model.available_in_ollama).length,
  );

  readonly draftProvider = computed(() =>
    resolveProvider(this.draftConfig().provider, this.cloudChoices()),
  );

  readonly draftCloudModel = computed(() =>
    resolveCloudModel(this.draftProvider(), this.draftConfig().cloudModel, this.cloudChoices()),
  );

  readonly activeCloudModels = computed(() => this.cloudChoices()[this.draftProvider()] || []);

  readonly runtimeLabel = computed(() =>
    this.draftConfig().useCloudServices
      ? `Cloud (${resolveProviderLabel(this.draftProvider())})`
      : 'Local (Ollama)',
  );

  readonly statusTone = computed(() => resolveStatusTone(this.statusMessage()));

  readonly cloudProviderOptions = computed<CloudProvider[]>(() => {
    const values = Object.keys(this.cloudChoices());
    const providers = values.filter(isCloudProvider);
    if (providers.length) {
      return providers;
    }
    return [...DEFAULT_CLOUD_PROVIDERS];
  });

  readonly missingRequiredModels = computed(() => {
    const draft = this.draftConfig();
    if (draft.useCloudServices) return [];
    const modelMap = new Map(this.localModels().map((model) => [model.name, model]));
    const missing = new Set<string>();
    for (const modelName of [draft.clinicalModel, draft.textExtractionModel]) {
      const candidate = modelName.trim();
      if (!candidate) continue;
      const localModel = modelMap.get(candidate);
      if (!localModel || !localModel.available_in_ollama) missing.add(candidate);
    }
    return Array.from(missing);
  });

  readonly noLocalModelMessage = computed(() => {
    if (this.isLoading()) return 'Loading local model catalog...';
    if (!this.localModels().length) return 'No local model catalog entries available.';
    if (!this.filteredLocalModels().length) {
      const q = this.modelSearchQuery().trim();
      return q ? `No models match "${q}".` : 'No models match the active filters.';
    }
    return '';
  });

  readonly saveDisabled = computed(() => {
    const settings = this.appState.state().diliAgent.settings;
    const draft = this.draftConfig();
    const savedProvider = resolveProvider(settings.provider, this.cloudChoices());
    const savedCloudModel = resolveCloudModel(savedProvider, settings.cloudModel, this.cloudChoices());
    const hasPendingChanges =
      draft.useCloudServices !== settings.useCloudServices ||
      this.draftProvider() !== savedProvider ||
      (this.draftCloudModel() || '') !== (savedCloudModel || '') ||
      draft.clinicalModel !== settings.clinicalModel ||
      draft.textExtractionModel !== settings.textExtractionModel ||
      draft.temperature !== settings.temperature;

    return (
      this.isLoading() ||
      this.isSaving() ||
      !hasPendingChanges ||
      (draft.useCloudServices && !this.draftCloudModel())
    );
  });

  async ngOnInit(): Promise<void> {
    await this.loadModelConfig();
  }

  async loadModelConfig(syncDraft = true, includeLocalAvailability?: boolean): Promise<void> {
    this.isLoading.set(true);
    try {
      const payload = await fetchModelConfigState(includeLocalAvailability);
      this.applyConfigToState(payload, syncDraft);
      this.statusMessage.set('');
    } catch (error) {
      this.statusMessage.set(formatUnknownError(error, 'Unable to load model settings.'));
    } finally {
      this.isLoading.set(false);
      this.hasLoadedConfig.set(true);
    }
  }

  private applyConfigToState(payload: ModelConfigStateResponse, syncDraft: boolean): void {
    this.localModels.set(payload.local_models || []);
    const choices = resolveCloudChoices(payload.cloud_model_choices);
    this.cloudChoices.set(choices);

    const current = this.appState.state().diliAgent.settings;
    const nextSettings: RuntimeSettings = buildRuntimeSettingsFromConfig(payload, current);
    this.appState.updateDiliAgent({ settings: nextSettings });

    if (syncDraft) {
      this.draftConfig.set(resolveDraftFromSettings(nextSettings));
    }
  }

  async persistConfigPatch(patch: ModelConfigUpdateRequest, successMessage = '', syncDraft = true): Promise<void> {
    this.isSaving.set(true);
    try {
      const payload = await updateModelConfigState(patch);
      this.applyConfigToState(payload, syncDraft);
      this.statusMessage.set(successMessage);
    } catch (error) {
      this.statusMessage.set(formatUnknownError(error, 'Unable to save model settings.'));
    } finally {
      this.isSaving.set(false);
    }
  }

  toggleFilter(key: ModelFilterKey): void {
    this.activeFilters.update((current) => ({ ...current, [key]: !current[key] }));
  }

  setModelSearchQuery(value: string): void {
    this.modelSearchQuery.set(value);
  }

  handleRoleSelection(role: ModelRole, modelName: string): void {
    this.draftConfig.update((previous) => ({
      ...previous,
      clinicalModel: role === 'clinical' ? modelName : previous.clinicalModel,
      textExtractionModel: role === 'text_extraction' ? modelName : previous.textExtractionModel,
    }));
  }

  handleCloudSwitchChange(value: boolean): void {
    this.draftConfig.update((previous) => ({ ...previous, useCloudServices: value }));
    if (!value) {
      void this.loadModelConfig(false, true);
    }
  }

  handleProviderChange(provider: CloudProvider): void {
    const resolvedProvider = resolveProvider(provider, this.cloudChoices());
    const cloudModel = resolveCloudModel(resolvedProvider, null, this.cloudChoices());
    this.draftConfig.update((previous) => ({
      ...previous,
      provider: resolvedProvider,
      cloudModel,
    }));
  }

  handleCloudModelChange(modelName: string): void {
    this.draftConfig.update((previous) => ({ ...previous, cloudModel: modelName || null }));
  }

  async handleReasoningChange(enabled: boolean): Promise<void> {
    await this.persistConfigPatch({ ollama_reasoning: enabled }, 'Extra parameters saved.', false);
  }

  setTemperature(value: string): void {
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed)) return;
    const bounded = Math.max(0, Math.min(2, parsed));
    this.draftConfig.update((previous) => ({
      ...previous,
      temperature: bounded,
    }));
  }

  async handleSaveConfiguration(): Promise<void> {
    const draft = this.draftConfig();
    const patch: ModelConfigUpdateRequest = {
      use_cloud_services: draft.useCloudServices,
      llm_provider: this.draftProvider(),
      cloud_model: this.draftCloudModel(),
      ollama_temperature: draft.temperature,
      cloud_temperature: draft.temperature,
    };
    if (!draft.useCloudServices) {
      patch.clinical_model = draft.clinicalModel || null;
      patch.text_extraction_model = draft.textExtractionModel || null;
    }
    await this.persistConfigPatch(
      patch,
      'Configuration saved.',
      true,
    );
  }

  openKeys(provider: AccessKeyProvider): void {
    this.openProviderModal.set(provider);
  }

  closeKeys(): void {
    this.openProviderModal.set(null);
  }

  async pullModelByName(modelName: string): Promise<void> {
    const candidate = modelName.trim();
    if (!candidate) {
      this.statusMessage.set('[ERROR] Enter a model name to pull from Ollama.');
      return;
    }
    await this.runPull([candidate], `[INFO] Pulling '${candidate}' from Ollama...`);
  }

  async installRequiredModels(modelNames: readonly string[]): Promise<void> {
    if (!modelNames.length) return;
    await this.runPull(modelNames, `[INFO] Installing required models: ${modelNames.join(', ')}.`);
  }

  private async runPull(models: readonly string[], startMessage: string): Promise<void> {
    const requested = Array.from(new Set(models.map((m) => m.trim()).filter((m) => !!m)));
    if (!requested.length) return;

    this.appState.updateDiliAgent({ isPulling: true });
    this.statusMessage.set(startMessage);
    let failureMessage = '';
    const completed: string[] = [];

    try {
      for (const modelName of requested) {
        this.updateModelPullProgress(modelName, {
          progress: 1,
          status: 'pending',
          message: `Starting pull for '${modelName}'...`,
        });

        try {
          const start = await startModelPullJob(modelName);
          const intervalMs = Math.max(250, Math.round(start.poll_interval * 1000));
          await this.pollPullJob(modelName, start.job_id, intervalMs);
          completed.push(modelName);
        } catch (error) {
          const description = error instanceof Error ? error.message : `Failed to pull '${modelName}'.`;
          failureMessage = description.startsWith('[ERROR]') ? description : `[ERROR] ${description}`;
          break;
        } finally {
          this.updateModelPullProgress(modelName, null);
        }
      }

      if (!failureMessage) {
        this.statusMessage.set(
          completed.length === 1
            ? `[INFO] Model available locally: ${completed[0]}.`
            : `[INFO] Models available locally: ${completed.join(', ')}.`,
        );
      }
    } finally {
      await this.loadModelConfig(false);
      if (failureMessage) {
        this.statusMessage.set(failureMessage);
      }
      this.appState.updateDiliAgent({ isPulling: false });
    }
  }

  private async pollPullJob(modelName: string, jobId: string, intervalMs: number): Promise<void> {
    const safeIntervalMs = Math.max(250, intervalMs);
    const requestTimeoutSeconds = Math.min(30, Math.max(5, Math.ceil((safeIntervalMs / 1000) * 4)));

    await this.jobPolling.run({
      intervalMs: safeIntervalMs,
      pollStep: async () => {
        const payload = await fetchModelPullJobStatus(jobId, requestTimeoutSeconds);
        const progress = Math.max(0, Math.min(100, payload.progress));
        const message =
          typeof payload.result?.progress_message === 'string' && payload.result.progress_message.trim()
            ? payload.result.progress_message
            : payload.status === 'completed'
              ? `Model '${modelName}' is available locally.`
              : payload.status === 'cancelled'
                ? `Pull cancelled for '${modelName}'.`
                : payload.status === 'failed'
                  ? `Pull failed for '${modelName}'.`
                  : `Pulling '${modelName}' from Ollama...`;

        this.updateModelPullProgress(modelName, {
          progress,
          status: payload.status,
          message,
        });

        if (!TERMINAL_JOB_STATUSES.includes(payload.status)) {
          return true;
        }
        if (payload.status === 'completed') {
          return false;
        }
        const errorMessage = payload.error?.trim() || message;
        throw new Error(`[ERROR] ${errorMessage}`);
      },
    });
  }

  private updateModelPullProgress(modelName: string, progress: ModelPullProgressState | null): void {
    this.modelPullProgress.update((current) => {
      if (progress === null) {
        if (!(modelName in current)) return current;
        const { [modelName]: removed, ...rest } = current;
        void removed;
        return rest;
      }
      return { ...current, [modelName]: progress };
    });
  }

  getProviderLabel(provider: AccessKeyProvider): string {
    return PROVIDER_LABELS[provider];
  }

  resolveProviderLabel(provider: string): string {
    return resolveProviderLabel(provider);
  }

  progressForModel(name: string): ModelPullProgressState | null {
    return this.modelPullProgress()[name] || null;
  }

  isFilterActive(key: ModelFilterKey): boolean {
    return this.activeFilters()[key];
  }
}

