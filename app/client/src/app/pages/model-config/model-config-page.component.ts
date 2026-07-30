import { CommonModule } from '@angular/common';
import { Component, HostListener, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  StatusMessageComponent,
  resolveStatusTone,
} from '../../components/status-message/status-message.component';
import { AppStateService } from '../../core/state/app-state.service';
import { formatUnknownError } from '../../core/utils';
import {
  buildRuntimeSettingsFromConfig,
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
  RuntimeSettings,
} from '../../core/models/types';
import {
  fetchModelConfigState,
  loadModelCatalog,
  refreshModelCatalog,
  updateModelConfigState,
} from '../../core/services/model-config-api';
import {
  ModelPullJobService,
  ModelPullProgressState,
} from '../../core/services/model-pull-job.service';
import { AccessKeyModalComponent } from './components/access-key-modal.component';
import { ModelConfigToggleCardComponent } from './components/model-config-toggle-card.component';
import { ModelRoleActionButtonComponent } from './components/model-role-action-button.component';
import {
  MODEL_FILTERS,
  modelMatchesFilters,
  normalizeDraftForLocalRuntime,
  resolveDraftFromSettings,
} from './model-catalog';
import {
  DraftRuntimeConfig,
  DraftRagSettings,
  ModelFilterKey,
  ModelRole,
  RagSettingsSectionKey,
} from './model-config.types';

const MODEL_BATCH_SIZE = 12;
type SaveOperation = 'configuration' | 'reasoning' | 'rag';

const PROVIDER_LABELS: Record<AccessKeyProvider, string> = {
  openai: 'OpenAI',
  gemini: 'Gemini',
  deepseek: 'DeepSeek',
  anthropic: 'Anthropic Claude',
  opencode: 'OpenCode',
  brave: 'Brave',
};
const PROVIDER_LOGOS: Partial<Record<AccessKeyProvider, { src: string; alt: string }>> = {
  openai: { src: '/logos/openai.svg', alt: 'OpenAI logo' },
  gemini: { src: '/logos/google-g.svg', alt: 'Google logo' },
  deepseek: { src: '/logos/deepseek.svg', alt: 'DeepSeek logo' },
  anthropic: { src: '/logos/anthropic.svg', alt: 'Anthropic logo' },
  opencode: { src: '/logos/opencode.svg', alt: 'OpenCode logo' },
};
const RUNTIME_PROVIDER_LABELS: Record<CloudProvider, string> = {
  openai: 'OpenAI',
  gemini: 'Google Gemini',
  deepseek: 'DeepSeek',
  anthropic: 'Anthropic Claude',
  opencode_zen: 'OpenCode Zen',
  opencode_go: 'OpenCode Go',
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

function isCloudProvider(provider: string): provider is CloudProvider {
  return ['openai', 'gemini', 'deepseek', 'anthropic', 'opencode_zen', 'opencode_go'].includes(provider);
}

function resolveProviderLabel(provider: string): string {
  if (isCloudProvider(provider)) {
    return RUNTIME_PROVIDER_LABELS[provider];
  }
  if (provider in PROVIDER_LABELS) {
    return PROVIDER_LABELS[provider as AccessKeyProvider];
  }
  return provider;
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
  ],
  templateUrl: './model-config-page.component.html',
  styleUrl: './model-config-page.component.scss',
})
export class ModelConfigPageComponent implements OnInit {
  readonly appState = inject(AppStateService);
  private readonly modelPullJobs = inject(ModelPullJobService);

  readonly modelFilters = MODEL_FILTERS;
  readonly rerankerProfileOptions = RERANKER_PROFILE_OPTIONS;

  readonly isLoading = signal(true);
  readonly savingOperations = signal<ReadonlySet<SaveOperation>>(new Set());
  readonly isSaving = computed(() => this.savingOperations().size > 0);
  readonly isConfigurationSaving = computed(() => this.savingOperations().has('configuration'));
  readonly isReasoningSaving = computed(() => this.savingOperations().has('reasoning'));
  readonly isRagSaving = computed(() => this.savingOperations().has('rag'));
  readonly localModels = signal<ModelConfigStateResponse['local_models']>([]);
  readonly localCatalog = signal<ModelConfigStateResponse['local_catalog']>({
    status: 'not_loaded',
    updated_at: null,
    message: null,
  });
  readonly cloudChoices = signal(resolveCloudChoices(undefined));
  readonly cloudProviders = signal<ModelConfigStateResponse['cloud_providers']>([]);
  readonly modelSearchQuery = signal('');
  readonly visibleModelLimit = signal(MODEL_BATCH_SIZE);
  readonly statusMessage = signal('');
  readonly openProviderModal = signal<AccessKeyProvider | null>(null);
  readonly modelPullProgress = signal<Record<string, ModelPullProgressState>>({});
  readonly previewReasoningLevel = signal(2);
  readonly previewCloudModelOverrides = signal<Partial<Record<CloudProvider, string>>>({});
  readonly ragSettings = signal<DraftRagSettings>({ ...DEFAULT_RAG_SETTINGS });
  readonly draftRagSettings = signal<DraftRagSettings>({ ...DEFAULT_RAG_SETTINGS });
  readonly ragSettingsModalOpen = signal(false);
  readonly activeRagSettingsSection = signal<RagSettingsSectionKey>(DEFAULT_RAG_SETTINGS_SECTION);
  readonly embeddingRuntime = signal<ModelConfigStateResponse['embedding_runtime'] | null>(null);
  readonly embeddingIndex = signal<ModelConfigStateResponse['embedding_index'] | null>(null);
  readonly activeFilters = signal<Record<ModelFilterKey, boolean>>({
    installed: false,
    missing: false,
    small: false,
    large: false,
    quantized: false,
  });
  readonly draftConfig = signal<DraftRuntimeConfig>(resolveDraftFromSettings(this.appState.state().diliAgent.settings));
  readonly lastUpdatedAt = signal<string | null>(null);
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
    resolveCloudModel(this.draftProvider(), this.draftConfig().cloudModel, this.cloudChoices()),
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
      ? `Cloud (${resolveProviderLabel(this.draftProvider())})`
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
    const parsed = new Date(updatedAt);
    if (Number.isNaN(parsed.getTime())) {
      return updatedAt;
    }
    return parsed.toLocaleString();
  });

  readonly selectedModelDescription = computed(() => {
    if (this.isLoading() && !this.localModels().length) {
      return 'Loading model details...';
    }
    const draft = this.draftConfig();
    if (draft.useCloudServices) {
      const providerLabel = resolveProviderLabel(this.draftProvider());
      const modelName = this.draftCloudModel() || draft.cloudModel;
      return modelName
        ? `Cloud model selected: ${modelName} (${providerLabel}). Provider-supplied model details are not available.`
        : `No cloud model is selected for ${providerLabel}.`;
    }
    const selectedNames = [draft.clinicalModel, draft.textExtractionModel].filter((name) => !!name);
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
  readonly credentialProviderOptions: readonly AccessKeyProvider[] = [
    'openai', 'gemini', 'deepseek', 'anthropic', 'opencode',
  ];

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
    const settings = this.appState.state().diliAgent.settings;
    const draft = this.draftConfig();
    const savedProvider = resolveProvider(settings.provider, this.cloudChoices());
    const savedCloudModel = resolveCloudModel(savedProvider, settings.cloudModel, this.cloudChoices());
    const hasPendingChanges =
      draft.useCloudServices !== settings.useCloudServices ||
      this.draftProvider() !== savedProvider ||
      (this.draftCloudModel() || '') !== (savedCloudModel || '') ||
      draft.clinicalModel !== settings.clinicalModel ||
      draft.textExtractionModel !== settings.textExtractionModel;

    return (
      this.isLoading() ||
      this.isConfigurationSaving() ||
      !hasPendingChanges ||
      (draft.useCloudServices && !this.draftCloudModel())
    );
  });

  async ngOnInit(): Promise<void> {
    await this.loadModelConfig(true, true);
    this.applyPreviewDefaultState();
  }

  async loadModelConfig(
    syncDraft = true,
    initializeCatalog = false,
  ): Promise<void> {
    this.isLoading.set(true);
    try {
      const payload = await fetchModelConfigState();
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
    this.localModels.set(payload.local_models || []);
    this.localCatalog.set(payload.local_catalog);
    this.cloudProviders.set(payload.cloud_providers || []);
    this.lastUpdatedAt.set(payload.updated_at);
    this.embeddingRuntime.set(payload.embedding_runtime);
    this.embeddingIndex.set(payload.embedding_index);
    const nextRagSettings = this.normalizeRagSettings(payload.rag_settings);
    this.ragSettings.set(nextRagSettings);
    if (!this.ragSettingsModalOpen()) {
      this.draftRagSettings.set({ ...nextRagSettings });
    }
    const choices = resolveCloudChoices(Object.fromEntries(
      (payload.cloud_providers || []).map((provider) => [provider.id, provider.models.map((model) => model.id)]),
    ));
    this.cloudChoices.set(choices);

    const current = this.appState.state().diliAgent.settings;
    const nextSettings: RuntimeSettings = buildRuntimeSettingsFromConfig(payload, current);
    this.appState.updateDiliAgent({ settings: nextSettings });
    this.previewCloudModelOverrides.update((previous) => ({
      ...previous,
      [nextSettings.provider]: nextSettings.cloudModel || undefined,
    }));

    if (syncDraft) {
      this.draftConfig.set(resolveDraftFromSettings(nextSettings));
    }
  }

  private applyPreviewDefaultState(): void {
    const state = this.appState.state().diliAgent;
    this.previewReasoningLevel.set(state.settings.reasoning ? 2 : 0);
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
    this.lastUpdatedAt.set(payload.updated_at);
    const current = this.appState.state().diliAgent.settings;
    const nextSettings: RuntimeSettings = { ...current };
    if ('use_cloud_services' in patch) nextSettings.useCloudServices = payload.use_cloud_services;
    if ('llm_provider' in patch) nextSettings.provider = payload.llm_provider;
    if ('cloud_model' in patch || 'llm_provider' in patch) nextSettings.cloudModel = payload.cloud_model;
    if ('clinical_model' in patch) nextSettings.clinicalModel = payload.clinical_model || '';
    if ('text_extraction_model' in patch) nextSettings.textExtractionModel = payload.text_extraction_model || '';
    if ('ollama_reasoning' in patch) nextSettings.reasoning = payload.ollama_reasoning;
    this.appState.updateDiliAgent({ settings: nextSettings });
    if ('rag_settings' in patch) {
      const nextRagSettings = this.normalizeRagSettings(payload.rag_settings);
      this.ragSettings.set(nextRagSettings);
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
      cloudModel: previous.useCloudServices ? modelName : previous.cloudModel,
    }));
  }

  handleCloudSwitchChange(value: boolean): void {
    if (!value) {
      this.draftConfig.update((previous) =>
        normalizeDraftForLocalRuntime(
          { ...previous, useCloudServices: false },
          this.localModels(),
        ),
      );
      return;
    }
    this.draftConfig.update((previous) => ({ ...previous, useCloudServices: true }));
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
    const enabled = normalizedLevel > 0;
    this.previewReasoningLevel.set(normalizedLevel);
    const currentSettings = this.appState.state().diliAgent.settings;
    this.appState.updateDiliAgent({
      settings: {
        ...currentSettings,
        reasoning: enabled,
      },
    });
    if (this.reasoningSaveTimer) {
      clearTimeout(this.reasoningSaveTimer);
    }
    this.reasoningSaveTimer = setTimeout(() => {
      void this.persistConfigPatch({ ollama_reasoning: enabled }, 'Extra parameters saved.', false, 'reasoning');
      this.reasoningSaveTimer = null;
    }, 250);
  }

  async handleSaveConfiguration(): Promise<void> {
    const draft = this.draftConfig();
    const missingLocalModels = this.missingRequiredModels();
    if (!draft.useCloudServices && missingLocalModels.length) {
      this.statusMessage.set(`[ERROR] Install selected local models before saving: ${missingLocalModels.join(', ')}.`);
      return;
    }
    const patch: ModelConfigUpdateRequest = {
      use_cloud_services: draft.useCloudServices,
      llm_provider: this.draftProvider(),
      cloud_model: this.draftCloudModel(),
      clinical_model: draft.clinicalModel || null,
      text_extraction_model: draft.textExtractionModel || null,
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

  getProviderLabel(provider: AccessKeyProvider): string {
    return PROVIDER_LABELS[provider];
  }

  providerLogo(provider: AccessKeyProvider): { src: string; alt: string } | null {
    return PROVIDER_LOGOS[provider] || null;
  }

  resolveProviderLabel(provider: string): string {
    return resolveProviderLabel(provider);
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

  isModelRoleSelectable(model: ModelConfigStateResponse['local_models'][number]): boolean {
    return model.available_in_ollama;
  }

  isFilterActive(key: ModelFilterKey): boolean {
    return this.activeFilters()[key];
  }
}

