import { CommonModule } from '@angular/common';
import { Component, HostListener, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  StatusMessageComponent,
  resolveStatusTone,
} from '../../components/status-message/status-message.component';
import { HelpPopoverComponent } from '../../core/guidance/help-popover.component';
import { AppStateService } from '../../core/state/app-state.service';
import { formatUnknownError } from '../../core/utils';
import { formatAppDateTime } from '../../core/utils/date-formatting';
import {
  resolveCloudChoices,
  resolveCloudModel,
  resolveProvider,
} from '../../core/model-config';
import {
  AccessKeyProvider,
  CatalogProvider,
  CloudProvider,
  ModelConfigStateResponse,
  ModelConfigPersistResponse,
  ModelConfigUpdateRequest,
  RagSettings,
  ReasoningLevel,
} from '../../core/models/types';
import {
  loadModelCatalog,
  refreshModelCatalog,
  updateModelConfigState,
} from '../../core/services/model-config-api';
import {
  ModelPullJobService,
  ModelPullProgressState,
} from '../../core/services/model-pull-job.service';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { AccessKeyModalComponent } from './components/access-key-modal.component';
import { ModelConfigToggleCardComponent } from './components/model-config-toggle-card.component';
import { ModelRoleActionButtonComponent } from './components/model-role-action-button.component';
import {
  MODEL_FILTERS,
  modelMatchesFilters,
  resolveDraftFromSettings,
} from './model-catalog';
import {
  DraftRuntimeConfig,
  DraftRagSettings,
  ModelFilterKey,
  ModelRole,
  RagSettingsSectionKey,
} from './model-config.types';
import { EMPTY_RUNTIME_SETTINGS } from '../../core/model-config';

const MODEL_BATCH_SIZE = 12;
const REASONING_LEVELS: readonly ReasoningLevel[] = ['off', 'low', 'medium', 'high'];
type SaveOperation = 'configuration' | 'reasoning' | 'rag';

const PROVIDER_LOGOS: Partial<Record<AccessKeyProvider, { src: string; alt: string }>> = {
  openai: { src: '/logos/openai-blossom-light.svg', alt: 'OpenAI logo' },
  gemini: { src: '/logos/google-g.svg', alt: 'Google logo' },
  deepseek: { src: '/logos/deepseek.svg', alt: 'DeepSeek logo' },
  anthropic: { src: '/logos/anthropic.svg', alt: 'Anthropic logo' },
  opencode: { src: '/logos/opencode.svg', alt: 'OpenCode logo' },
};
const DEFAULT_RAG_SETTINGS_SECTION: RagSettingsSectionKey = 'general';
const RERANKER_PROFILE_OPTIONS = [
  { value: 'lightweight-balanced-v1', label: 'Balanced' },
  { value: 'lightweight-lexical-v1', label: 'Lexical' },
  { value: 'lightweight-phrase-v1', label: 'Phrase' },
] as const;

const DEFAULT_RAG_SETTINGS: DraftRagSettings = {
  chunk_size: 1024,
  chunk_overlap: 128,
  embedding_batch_size: 64,
  use_hybrid_search: true,
  use_reranking: true,
  retrieval_candidate_count: 40,
  retrieval_selected_count: 6,
  reranker_model: 'lightweight-balanced-v1',
  hybrid_vector_weight: 0.7,
  hybrid_text_weight: 0.3,
  vector_stream_batch_size: 250,
  embedding_device: 'auto',
  embedding_offline_mode: false,
};

const EMPTY_LOCAL_CATALOG: ModelConfigStateResponse['local_catalog'] = {
  status: 'not_loaded',
  updated_at: null,
  message: null,
};

type CredentialProviderOption = {
  provider: AccessKeyProvider;
  label: string;
  description: string;
};

function resolveProviderLabel(provider: string, cloudProviders: ModelConfigStateResponse['cloud_providers']): string {
  const descriptor = cloudProviders.find((candidate) => candidate.id === provider);
  return descriptor?.display_name?.trim() || provider;
}

@Component({
  selector: 'app-model-config-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    StatusMessageComponent,
    AccessKeyModalComponent,
    ModelConfigToggleCardComponent,
    ModelRoleActionButtonComponent,
    HelpPopoverComponent,
  ],
  templateUrl: './model-config-page.component.html',
  styleUrl: './model-config-page.component.scss',
})
export class ModelConfigPageComponent implements OnInit, OnDestroy {
  readonly appState = inject(AppStateService);
  readonly modelConfigState = inject(ModelConfigStateService);
  private readonly modelPullJobs = inject(ModelPullJobService);

  readonly modelFilters = MODEL_FILTERS;
  readonly rerankerProfileOptions = RERANKER_PROFILE_OPTIONS;

  readonly isLoading = signal(true);
  readonly savingOperations = signal<ReadonlySet<SaveOperation>>(new Set());
  readonly isSaving = computed(() => this.savingOperations().size > 0);
  readonly isConfigurationSaving = computed(() => this.savingOperations().has('configuration'));
  readonly isReasoningSaving = computed(() => this.savingOperations().has('reasoning'));
  readonly isRagSaving = computed(() => this.savingOperations().has('rag'));
  readonly localModels = computed(() => this.modelConfigState.data()?.local_models ?? []);
  readonly localCatalog = computed(() => this.modelConfigState.data()?.local_catalog ?? EMPTY_LOCAL_CATALOG);
  readonly cloudProviders = computed(() => this.modelConfigState.data()?.cloud_providers ?? []);
  readonly cloudChoices = computed(() => resolveCloudChoices(Object.fromEntries(
    this.cloudProviders().map((provider) => [
      provider.id,
      provider.models.map((model) => model.id),
    ]),
  )));
  readonly modelSearchQuery = signal('');
  readonly visibleModelLimit = signal(MODEL_BATCH_SIZE);
  readonly statusMessage = signal('');
  readonly openProviderModal = signal<AccessKeyProvider | null>(null);
  readonly modelPullProgress = signal<Record<string, ModelPullProgressState>>({});
  readonly previewReasoningLevel = signal(0);
  readonly previewCloudModelOverrides = signal<Partial<Record<CloudProvider, string>>>({});
  readonly ragSettings = computed(() => this.normalizeRagSettings(this.modelConfigState.data()?.rag_settings));
  readonly draftRagSettings = signal<DraftRagSettings>({ ...DEFAULT_RAG_SETTINGS });
  readonly ragSettingsModalOpen = signal(false);
  readonly activeRagSettingsSection = signal<RagSettingsSectionKey>(DEFAULT_RAG_SETTINGS_SECTION);
  readonly embeddingRuntime = computed(() => this.modelConfigState.data()?.embedding_runtime ?? null);
  readonly embeddingIndex = computed(() => this.modelConfigState.data()?.embedding_index ?? null);
  readonly activeFilters = signal<Record<ModelFilterKey, boolean>>({
    installed: false,
    missing: false,
    small: false,
    large: false,
    quantized: false,
  });
  readonly draftConfig = signal<DraftRuntimeConfig>(resolveDraftFromSettings(EMPTY_RUNTIME_SETTINGS));
  readonly lastUpdatedAt = computed(() => this.modelConfigState.data()?.updated_at ?? null);
  readonly catalogProviderInFlight = signal<CatalogProvider | null>(null);
  private reasoningSaveTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly saveRevisions: Record<SaveOperation, number> = {
    configuration: 0,
    reasoning: 0,
    rag: 0,
  };

  @HostListener('document:keydown.escape')
  closeRagSettingsOnEscape(): void {
    if (this.ragSettingsModalOpen()) {
      this.closeRagSettingsPanel();
    }
  }

  readonly filteredLocalModels = computed(() => {
    const query = this.modelSearchQuery().trim().toLowerCase();
    return this.localModels().filter((model) =>
      modelMatchesFilters(model, query, this.activeFilters()),
    );
  });

  readonly availableLocalModelCount = computed(
    () => this.localModels().filter((model) => model.available_in_ollama).length,
  );

  readonly visibleLocalModels = computed(() => {
    return this.filteredLocalModels().slice(0, this.visibleModelLimit());
  });

  readonly filteredCloudModels = computed(() => {
    const query = this.modelSearchQuery().trim().toLowerCase();
    const models = this.cloudChoices()[this.draftProvider()] || [];
    if (!query) return models;
    return models.filter((name) => name.toLowerCase().includes(query));
  });

  readonly visibleCloudModels = computed(() => {
    return this.filteredCloudModels().slice(0, this.visibleModelLimit());
  });

  readonly canRevealMoreModels = computed(() =>
    this.draftConfig().useCloudServices
      ? this.visibleCloudModels().length < this.filteredCloudModels().length
      : this.visibleLocalModels().length < this.filteredLocalModels().length,
  );

  readonly visibleModelRangeStart = computed(() =>
    this.filteredModelCount() ? 1 : 0,
  );

  readonly visibleModelRangeEnd = computed(() =>
    this.draftConfig().useCloudServices ? this.visibleCloudModels().length : this.visibleLocalModels().length,
  );

  readonly filteredModelCount = computed(() =>
    this.draftConfig().useCloudServices ? this.filteredCloudModels().length : this.filteredLocalModels().length,
  );

  readonly draftProvider = computed(() =>
    resolveProvider(this.draftConfig().provider, this.cloudChoices()),
  );

  readonly draftCloudModel = computed(() =>
    resolveCloudModel(this.draftProvider(), this.draftConfig().cloudModel, this.cloudChoices(), true),
  );

  readonly activeCloudModels = computed(() => this.cloudChoices()[this.draftProvider()] || []);

  readonly selectedCloudProvider = computed(() =>
    this.cloudProviders().find((provider) => provider.id === this.draftProvider()) || null,
  );

  readonly cloudCatalogMessage = computed(() => {
    const provider = this.selectedCloudProvider();
    if (!provider) return '';
    if (provider.catalog_message) return provider.catalog_message;
    if (provider.catalog_status === 'available') return 'Live provider catalog loaded.';
    if (provider.catalog_status === 'unavailable') {
      return 'Provider catalog is temporarily unavailable; the configured model remains available.';
    }
    return 'No provider models were reported.';
  });

  readonly runtimeLabel = computed(() =>
    this.draftConfig().useCloudServices
      ? `Cloud (${resolveProviderLabel(this.draftProvider(), this.cloudProviders())})`
      : 'Local (Ollama)',
  );

  readonly statusTone = computed(() => resolveStatusTone(this.statusMessage()));

  readonly reasoningLevel = computed(() => this.previewReasoningLevel());
  readonly reasoningEnabled = computed(() => this.reasoningLevel() > 0);
  readonly currentRagModelLabel = computed(() => this.embeddingRuntime()?.model_display_name || 'Granite Embedding Small English R2');
  readonly ragSettingsValidationMessage = computed(() => {
    const draft = this.draftRagSettings();
    if (draft.retrieval_candidate_count < 1) {
      return 'Retrieved documents must be at least 1.';
    }
    if (draft.retrieval_selected_count < 1) {
      return 'Selected documents must be at least 1.';
    }
    if (draft.retrieval_selected_count > draft.retrieval_candidate_count) {
      return 'Selected documents cannot exceed retrieved documents.';
    }
    if (draft.chunk_overlap >= draft.chunk_size) {
      return 'Chunk overlap must be smaller than chunk size.';
    }
    return '';
  });

  readonly ragSettingsSaveDisabled = computed(
    () => this.isRagSaving() || this.isLoading() || !!this.ragSettingsValidationMessage(),
  );

  readonly lastSavedLabel = computed(() => {
    const updatedAt = this.lastUpdatedAt();
    if (this.isLoading() && !updatedAt) {
      return 'Loading...';
    }
    if (!updatedAt) {
      return 'Not saved in this session';
    }
    return formatAppDateTime(updatedAt, updatedAt);
  });

  readonly selectedModelDescription = computed(() => {
    if (this.isLoading() && !this.localModels().length) {
      return 'Loading model details...';
    }
    const draft = this.draftConfig();
    if (draft.useCloudServices) {
      const providerLabel = resolveProviderLabel(this.draftProvider(), this.cloudProviders());
      const modelName = this.draftCloudModel() || draft.cloudModel;
      return modelName
        ? `Cloud model selected: ${modelName} (${providerLabel}). Provider-supplied model details are not available.`
        : `No cloud model is selected for ${providerLabel}.`;
    }
    const selectedNames = [
      draft.clinicalModel,
      draft.textExtractionModel,
      draft.revisionModel,
      draft.timelineModel,
    ].filter((name) => !!name);
    const modelMap = new Map(this.localModels().map((model) => [model.name, model.description || '']));
    for (const modelName of selectedNames) {
      const description = modelMap.get(modelName);
      if (description?.trim()) {
        return description.trim();
      }
    }
    return 'Select an installed model to show catalog details here.';
  });

  readonly cloudProviderOptions = computed<CloudProvider[]>(() => {
    return this.cloudProviders().map((provider) => provider.id);
  });
  readonly cloudProviderSummary = computed(() => {
    const names = this.cloudProviders()
      .map((provider) => provider.display_name.trim())
      .filter((name) => !!name);
    return names.length ? names.join(', ') : 'Configured by the backend catalog';
  });
  readonly credentialProviderOptions = computed<CredentialProviderOption[]>(() => {
    const grouped = new Map<AccessKeyProvider, ModelConfigStateResponse['cloud_providers']>();
    for (const provider of this.cloudProviders()) {
      const current = grouped.get(provider.credential_scope) || [];
      grouped.set(provider.credential_scope, [...current, provider]);
    }
    return Array.from(grouped.entries()).map(([provider, descriptors]) => {
      const labels = descriptors
        .map((descriptor) => descriptor.display_name.trim())
        .filter((label) => !!label);
      const label = labels.length > 1 ? labels.join(' / ') : labels[0] || provider;
      const description = labels.length > 1
        ? `This credential is used by ${labels.join(' and ')} cloud models.`
        : `Manage the API keys used to access ${label} cloud models.`;
      return { provider, label, description };
    });
  });
  readonly selectedCredentialProvider = computed<AccessKeyProvider | null>(() =>
    this.cloudProviders().find((provider) => provider.id === this.draftProvider())?.credential_scope || null,
  );

  readonly missingRequiredModels = computed(() => {
    const draft = this.draftConfig();
    if (draft.useCloudServices) return [];
    const modelMap = new Map(this.localModels().map((model) => [model.name, model]));
    const missing = new Set<string>();
    for (const modelName of [
      draft.clinicalModel,
      draft.textExtractionModel,
      draft.revisionModel,
      draft.timelineModel,
    ]) {
      const candidate = modelName.trim();
      if (!candidate) continue;
      const localModel = modelMap.get(candidate);
      if (!localModel || !localModel.available_in_ollama) missing.add(candidate);
    }
    return Array.from(missing);
  });

  readonly catalogEmptyMessage = computed(() => {
    if (this.isLoading()) return 'Loading local model catalog...';
    if (this.draftConfig().useCloudServices) {
      if (!this.filteredCloudModels().length) {
        const q = this.modelSearchQuery().trim();
        return q ? `No cloud models match "${q}".` : 'No cloud models are available for this provider.';
      }
      return '';
    }
    if (!this.localModels().length) return 'No local model catalog entries available.';
    if (!this.filteredLocalModels().length) {
      const q = this.modelSearchQuery().trim();
      return q ? `No models match "${q}".` : 'No models match the active filters.';
    }
    return '';
  });

  readonly saveDisabled = computed(() => {
    const settings = this.modelConfigState.settings();
    if (!settings) return true;
    const draft = this.draftConfig();
    const savedProvider = resolveProvider(settings.provider, this.cloudChoices());
    const savedCloudModel = resolveCloudModel(savedProvider, settings.cloudModel, this.cloudChoices(), true);
    const hasAllRoleAssignments = [
      draft.clinicalModel,
      draft.textExtractionModel,
      draft.revisionModel,
      draft.timelineModel,
    ].every((modelName) => modelName.trim().length > 0);
    const hasPendingChanges =
      draft.useCloudServices !== settings.useCloudServices ||
      this.draftProvider() !== savedProvider ||
      (this.draftCloudModel() || '') !== (savedCloudModel || '') ||
      draft.clinicalModel !== settings.clinicalModel ||
      draft.textExtractionModel !== settings.textExtractionModel ||
      draft.revisionModel !== settings.revisionModel ||
      draft.timelineModel !== settings.timelineModel;

    return (
      this.isLoading() ||
      this.isConfigurationSaving() ||
      !hasPendingChanges ||
      !hasAllRoleAssignments ||
      (draft.useCloudServices && !this.draftCloudModel())
    );
  });

  async ngOnInit(): Promise<void> {
    await this.loadModelConfig(true, true);
    this.applyPreviewDefaultState();
    if (typeof window !== 'undefined' && window.location.hash === '#model-roles') {
      setTimeout(() => document.getElementById('model-roles')?.scrollIntoView({ block: 'start' }), 0);
    }
  }

  ngOnDestroy(): void {
    if (this.reasoningSaveTimer !== null) {
      clearTimeout(this.reasoningSaveTimer);
      this.reasoningSaveTimer = null;
    }
  }

  async loadModelConfig(
    syncDraft = true,
    initializeCatalog = false,
  ): Promise<void> {
    this.isLoading.set(true);
    try {
      const payload = await this.modelConfigState.load(true);
      this.applyConfigToState(payload, syncDraft);
      this.statusMessage.set('');
      if (initializeCatalog) {
        await this.ensureSelectedCatalog();
      }
    } catch (error) {
      this.statusMessage.set(formatUnknownError(error, 'Unable to load model settings.'));
    } finally {
      this.isLoading.set(false);
    }
  }

  private applyConfigToState(payload: ModelConfigStateResponse, syncDraft: boolean): void {
    this.modelConfigState.setFromApiState(payload);
    const nextRagSettings = this.ragSettings();
    if (!this.ragSettingsModalOpen()) {
      this.draftRagSettings.set({ ...nextRagSettings });
    }

    const nextSettings = this.modelConfigState.settings();
    if (!nextSettings) return;
    this.previewCloudModelOverrides.update((previous) => ({
      ...previous,
      [nextSettings.provider]: nextSettings.cloudModel || undefined,
    }));

    if (syncDraft) {
      this.draftConfig.set(resolveDraftFromSettings(nextSettings));
    }
  }

  private applyPreviewDefaultState(): void {
    const settings = this.modelConfigState.settings();
    this.previewReasoningLevel.set(
      Math.max(0, REASONING_LEVELS.indexOf(settings?.reasoning || 'off')),
    );
  }

  async persistConfigPatch(
    patch: ModelConfigUpdateRequest,
    successMessage = '',
    syncDraft = true,
    operation: SaveOperation = 'configuration',
  ): Promise<void> {
    const revision = ++this.saveRevisions[operation];
    this.savingOperations.update((current) => new Set(current).add(operation));
    try {
      const payload = await updateModelConfigState(patch);
      if (revision !== this.saveRevisions[operation]) return;
      this.applyPersistedConfigToState(payload, patch, syncDraft);
      this.statusMessage.set(successMessage);
    } catch (error) {
      this.statusMessage.set(formatUnknownError(error, 'Unable to save model settings.'));
    } finally {
      if (revision === this.saveRevisions[operation]) {
        this.savingOperations.update((current) => {
          const next = new Set(current);
          next.delete(operation);
          return next;
        });
      }
    }
  }

  private selectedCatalogProvider(): CatalogProvider {
    return this.draftConfig().useCloudServices ? this.draftProvider() : 'ollama';
  }

  private catalogStatus(provider: CatalogProvider): string {
    if (provider === 'ollama') return this.localCatalog().status;
    return this.cloudProviders().find((item) => item.id === provider)?.catalog_status || 'not_loaded';
  }

  private async ensureSelectedCatalog(): Promise<void> {
    const provider = this.selectedCatalogProvider();
    if (this.catalogStatus(provider) === 'not_loaded') {
      await this.runCatalogOperation(provider, false);
    }
  }

  async refreshSelectedCatalog(): Promise<void> {
    await this.runCatalogOperation(this.selectedCatalogProvider(), true);
  }

  private async runCatalogOperation(
    provider: CatalogProvider,
    forceRefresh: boolean,
  ): Promise<void> {
    if (this.catalogProviderInFlight()) return;
    this.catalogProviderInFlight.set(provider);
    try {
      const result = forceRefresh
        ? await refreshModelCatalog(provider)
        : await loadModelCatalog(provider);
      this.applyConfigToState(result.state, false);
      if (result.outcome === 'failed') {
        this.statusMessage.set(`[ERROR] ${result.error || 'Unable to refresh the model catalog.'}`);
      } else if (forceRefresh) {
        this.statusMessage.set('Model catalog refreshed.');
      }
    } catch (error) {
      this.statusMessage.set(formatUnknownError(error, 'Unable to refresh the model catalog.'));
    } finally {
      this.catalogProviderInFlight.set(null);
    }
  }

  private applyPersistedConfigToState(
    payload: ModelConfigPersistResponse,
    patch: ModelConfigUpdateRequest,
    syncDraft: boolean,
  ): void {
    this.modelConfigState.applyPersistedResponse(payload);
    const nextSettings = this.modelConfigState.settings();
    if (!nextSettings) return;
    if ('rag_settings' in patch) {
      const nextRagSettings = this.ragSettings();
      if (!this.ragSettingsModalOpen()) this.draftRagSettings.set({ ...nextRagSettings });
    }
    this.previewCloudModelOverrides.update((previous) => ({
      ...previous,
      [nextSettings.provider]: nextSettings.cloudModel || undefined,
    }));
    if (syncDraft) this.draftConfig.set(resolveDraftFromSettings(nextSettings));
  }

  private normalizeRagSettings(settings: Partial<RagSettings> | null | undefined): DraftRagSettings {
    const rerankerModel = typeof settings?.reranker_model === 'string'
      ? settings.reranker_model.trim()
      : '';
    return {
      ...DEFAULT_RAG_SETTINGS,
      ...(settings || {}),
      reranker_model: rerankerModel || DEFAULT_RAG_SETTINGS.reranker_model,
      retrieval_candidate_count: this.coercePositiveInteger(
        settings?.retrieval_candidate_count,
        DEFAULT_RAG_SETTINGS.retrieval_candidate_count,
      ),
      retrieval_selected_count: this.coercePositiveInteger(
        settings?.retrieval_selected_count,
        DEFAULT_RAG_SETTINGS.retrieval_selected_count,
      ),
      chunk_size: this.coercePositiveInteger(settings?.chunk_size, DEFAULT_RAG_SETTINGS.chunk_size),
      chunk_overlap: this.coercePositiveInteger(settings?.chunk_overlap, DEFAULT_RAG_SETTINGS.chunk_overlap),
      embedding_batch_size: this.coercePositiveInteger(
        settings?.embedding_batch_size,
        DEFAULT_RAG_SETTINGS.embedding_batch_size,
      ),
      vector_stream_batch_size: this.coercePositiveInteger(
        settings?.vector_stream_batch_size,
        DEFAULT_RAG_SETTINGS.vector_stream_batch_size,
      ),
    };
  }

  private coercePositiveInteger(value: number | null | undefined, fallback: number): number {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.max(1, Math.trunc(parsed));
  }

  toggleFilter(key: ModelFilterKey): void {
    this.activeFilters.update((current) => ({ ...current, [key]: !current[key] }));
    this.resetVisibleModelLimit();
  }

  clearFilters(): void {
    this.activeFilters.set({
      installed: false,
      missing: false,
      small: false,
      large: false,
      quantized: false,
    });
    this.resetVisibleModelLimit();
  }

  setModelSearchQuery(value: string): void {
    this.modelSearchQuery.set(value);
    this.resetVisibleModelLimit();
  }

  onModelCatalogScroll(event: Event): void {
    const shell = event.target instanceof HTMLElement ? event.target : null;
    if (!shell || !this.canRevealMoreModels()) return;
    const remainingScroll = shell.scrollHeight - shell.scrollTop - shell.clientHeight;
    if (remainingScroll <= 96) {
      this.revealMoreModels();
    }
  }

  private resetVisibleModelLimit(): void {
    this.visibleModelLimit.set(MODEL_BATCH_SIZE);
  }

  private revealMoreModels(): void {
    this.visibleModelLimit.update((current) =>
      Math.min(current + MODEL_BATCH_SIZE, this.filteredModelCount()),
    );
  }

  handleRoleSelection(role: ModelRole, modelName: string): void {
    this.draftConfig.update((previous) => ({
      ...previous,
      clinicalModel: role === 'clinical' ? modelName : previous.clinicalModel,
      textExtractionModel: role === 'text_extraction' ? modelName : previous.textExtractionModel,
      revisionModel: role === 'revision' ? modelName : previous.revisionModel,
      timelineModel: role === 'timeline' ? modelName : previous.timelineModel,
      cloudModel: previous.useCloudServices && !previous.cloudModel ? modelName : previous.cloudModel,
    }));
  }

  handleCloudSwitchChange(value: boolean): void {
    this.draftConfig.update((previous) => ({
      ...previous,
      useCloudServices: value,
      cloudModel: value ? previous.cloudModel : null,
      clinicalModel: '',
      textExtractionModel: '',
      revisionModel: '',
      timelineModel: '',
    }));
  }

  openRagSettingsPanel(): void {
    this.draftRagSettings.set({ ...this.ragSettings() });
    this.activeRagSettingsSection.set(DEFAULT_RAG_SETTINGS_SECTION);
    this.ragSettingsModalOpen.set(true);
  }

  closeRagSettingsPanel(): void {
    this.ragSettingsModalOpen.set(false);
    this.activeRagSettingsSection.set(DEFAULT_RAG_SETTINGS_SECTION);
    this.draftRagSettings.set({ ...this.ragSettings() });
  }

  setActiveRagSettingsSection(section: RagSettingsSectionKey): void {
    this.activeRagSettingsSection.set(section);
  }

  setDraftRagNumber(
    key: keyof Pick<
      DraftRagSettings,
      | 'chunk_size'
      | 'chunk_overlap'
      | 'embedding_batch_size'
      | 'retrieval_candidate_count'
      | 'retrieval_selected_count'
      | 'vector_stream_batch_size'
    >,
    value: string,
  ): void {
    const parsed = Number.parseInt(value, 10);
    this.draftRagSettings.update((previous) => ({
      ...previous,
      [key]: Number.isFinite(parsed) ? Math.max(0, parsed) : 0,
    }));
  }

  setDraftRagText(
    key: keyof Pick<
      DraftRagSettings,
      | 'reranker_model'
      | 'embedding_device'
    >,
    value: string,
  ): void {
    this.draftRagSettings.update((previous) => ({
      ...previous,
      [key]: value,
    }));
  }

  setDraftRagBoolean(
    key: keyof Pick<
      DraftRagSettings,
      'use_hybrid_search' | 'use_reranking' | 'embedding_offline_mode'
    >,
    value: boolean,
  ): void {
    this.draftRagSettings.update((previous) => ({
      ...previous,
      [key]: value,
    }));
  }

  async saveRagSettings(): Promise<void> {
    if (this.ragSettingsValidationMessage()) return;
    await this.persistConfigPatch(
      { rag_settings: { ...this.draftRagSettings() } },
      'RAG settings saved.',
      false,
      'rag',
    );
    this.ragSettingsModalOpen.set(false);
  }

  handleProviderChange(provider: CloudProvider): void {
    if (this.catalogProviderInFlight()) return;
    const resolvedProvider = resolveProvider(provider, this.cloudChoices());
    this.draftConfig.update((previous) => ({
      ...previous,
      provider: resolvedProvider,
      cloudModel: null,
      clinicalModel: '',
      textExtractionModel: '',
      revisionModel: '',
      timelineModel: '',
    }));
    this.modelSearchQuery.set('');
    this.resetVisibleModelLimit();
    void this.ensureSelectedCatalog();
  }

  handleCloudModelChange(modelName: string): void {
    this.previewCloudModelOverrides.update((previous) => ({
      ...previous,
      [this.draftProvider()]: modelName,
    }));
    this.draftConfig.update((previous) => ({ ...previous, cloudModel: modelName || null }));
  }

  handleReasoningLevelChange(level: number): void {
    const normalizedLevel = Math.max(0, Math.min(3, Math.round(level)));
    const reasoningLevel = REASONING_LEVELS[normalizedLevel];
    this.previewReasoningLevel.set(normalizedLevel);
    if (this.reasoningSaveTimer) {
      clearTimeout(this.reasoningSaveTimer);
    }
    this.reasoningSaveTimer = setTimeout(() => {
      void this.persistConfigPatch({ reasoning_level: reasoningLevel }, 'Reasoning preference saved.', false, 'reasoning');
      this.reasoningSaveTimer = null;
    }, 250);
  }

  async handleSaveConfiguration(): Promise<void> {
    const draft = this.draftConfig();
    if ([
      draft.clinicalModel,
      draft.textExtractionModel,
      draft.revisionModel,
      draft.timelineModel,
    ].some((modelName) => !modelName.trim())) {
      this.statusMessage.set('[ERROR] Select a model for every configured role before saving.');
      return;
    }
    const missingLocalModels = this.missingRequiredModels();
    if (!draft.useCloudServices && missingLocalModels.length) {
      this.statusMessage.set(`[ERROR] Install selected local models before saving: ${missingLocalModels.join(', ')}.`);
      return;
    }
    const patch: ModelConfigUpdateRequest = {
      use_cloud_services: draft.useCloudServices,
      llm_provider: this.draftProvider(),
      cloud_model: this.draftCloudModel(),
      clinical_model: draft.clinicalModel,
      text_extraction_model: draft.textExtractionModel,
      revision_model: draft.revisionModel,
      timeline_model: draft.timelineModel,
    };
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

    try {
      const { completedModels } = await this.modelPullJobs.pullModels(
        requested,
        (modelName, progress) => this.updateModelPullProgress(modelName, progress),
      );

      this.statusMessage.set(
        completedModels.length === 1
          ? `[INFO] Model available locally: ${completedModels[0]}.`
          : `[INFO] Models available locally: ${completedModels.join(', ')}.`,
      );
    } catch (error) {
      const description = error instanceof Error ? error.message : 'Failed to pull selected models.';
      failureMessage = description.startsWith('[ERROR]') ? description : `[ERROR] ${description}`;
    } finally {
      await this.loadModelConfig(false);
      if (failureMessage) {
        this.statusMessage.set(failureMessage);
      }
      this.appState.updateDiliAgent({ isPulling: false });
    }
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

  providerLogo(provider: AccessKeyProvider): { src: string; alt: string } | null {
    return PROVIDER_LOGOS[provider] || null;
  }

  resolveProviderLabel(provider: string): string {
    const cloudLabel = resolveProviderLabel(provider, this.cloudProviders());
    if (cloudLabel !== provider) return cloudLabel;
    return this.credentialProviderOptions().find((option) => option.provider === provider)?.label || provider;
  }

  progressForModel(name: string): ModelPullProgressState | null {
    return this.modelPullProgress()[name] || null;
  }

  modelSizeLabel(modelName: string): string {
    const match = modelName.match(/:(\d+(?:\.\d+)?)([mb])(?:$|[-_])/i);
    if (!match) {
      return 'Unknown';
    }
    return `${match[1]}${match[2].toUpperCase()}`;
  }

  modelContextLabel(modelName: string): string {
    const match = modelName.match(/(\d+)k/i);
    return match ? `${match[1]}K` : 'Default';
  }

  modelQuantLabel(modelName: string): string {
    const match = modelName.match(/q\d(?:_[a-z0-9]+)*/i);
    return match ? match[0].toUpperCase() : 'Default';
  }

  modelStatus(model: ModelConfigStateResponse['local_models'][number]): string {
    return model.available_in_ollama ? 'Installed' : 'Not installed';
  }

  localRecommendationLabel(model: ModelConfigStateResponse['local_models'][number]): string | null {
    if (!model.recommended_for_local_extraction) {
      return null;
    }
    if (model.recommended_rank === 0) {
      return 'Recommended fast extractor';
    }
    return 'Recommended backup extractor';
  }

  selectedClinicalModel(): string {
    return this.draftConfig().clinicalModel || 'Not set';
  }

  selectedExtractionModel(): string {
    return this.draftConfig().textExtractionModel || 'Not set';
  }

  selectedRevisionModel(): string {
    return this.draftConfig().revisionModel || 'Not set';
  }

  selectedTimelineModel(): string {
    return this.draftConfig().timelineModel || 'Not set';
  }

  isModelRoleSelectable(model: ModelConfigStateResponse['local_models'][number]): boolean {
    return model.available_in_ollama;
  }

  isFilterActive(key: ModelFilterKey): boolean {
    return this.activeFilters()[key];
  }
}

