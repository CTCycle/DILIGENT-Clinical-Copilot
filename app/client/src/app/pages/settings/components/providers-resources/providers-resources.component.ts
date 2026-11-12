// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, HostListener, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';

import {
  StatusMessageComponent,
  resolveStatusTone,
} from '../../../../components/status-message/status-message.component';
import {
  AccessKeyProvider,
  CatalogProvider,
  CloudModelDescriptor,
  ModelConfigStateResponse,
  RagSettings,
} from '../../../../core/models/types';
import {
  loadModelCatalog,
  refreshModelCatalog,
  updateModelConfigState,
} from '../../../../core/services/model-config-api';
import {
  ModelPullJobService,
  ModelPullProgressState,
} from '../../../../core/services/model-pull-job.service';
import { AppStateService } from '../../../../core/state/app-state.service';
import { ModelConfigStateService } from '../../../../core/state/model-config-state.service';
import { formatUnknownError } from '../../../../core/utils';
import { ProviderCatalogPanelComponent } from '../../../model-config/components/provider-catalog-panel.component';
import {
  CredentialProviderOption,
  ProviderCredentialsPanelComponent,
} from '../../../model-config/components/provider-credentials-panel.component';
import { RagResourcesPanelComponent } from '../../../model-config/components/rag-resources-panel.component';
import { modelMatchesFilters } from '../../../model-config/model-catalog';
import {
  DraftRagSettings,
  ModelFilterKey,
  RagBooleanKey,
  RagNumberKey,
  RagSettingsSectionKey,
  RagTextKey,
} from '../../../model-config/model-config.types';

const MODEL_BATCH_SIZE = 12;
const RAG_AUTOSAVE_DELAY_MS = 400;
const DEFAULT_RAG_SECTION: RagSettingsSectionKey = 'chunking';
const LOCAL_SOURCE: CatalogProvider = 'ollama';

const PROVIDER_LOGOS: Partial<Record<AccessKeyProvider, { src: string; alt: string }>> = {
  openai: { src: '/logos/openai-blossom-light.svg', alt: 'OpenAI logo' },
  gemini: { src: '/logos/google-g.svg', alt: 'Google logo' },
  deepseek: { src: '/logos/deepseek.svg', alt: 'DeepSeek logo' },
  anthropic: { src: '/logos/anthropic.svg', alt: 'Anthropic logo' },
  opencode: { src: '/logos/opencode.svg', alt: 'OpenCode logo' },
};

const DEFAULT_RAG_SETTINGS: DraftRagSettings = {
  allow_local_filesystem_access: true,
  vector_collection_name: 'documents',
  chunk_size: 512,
  chunk_overlap: 64,
  embedding_batch_size: 8,
  vector_index_metric: 'cosine',
  vector_index_type: 'IVF_FLAT',
  vector_stream_batch_size: 1024,
  embedding_offline_mode: false,
};

const EMPTY_LOCAL_CATALOG: ModelConfigStateResponse['local_catalog'] = {
  status: 'not_loaded',
  updated_at: null,
  message: null,
};

const NO_FILTERS: Record<ModelFilterKey, boolean> = {
  installed: false,
  missing: false,
  small: false,
  large: false,
  quantized: false,
};

function positiveInteger(value: number | null | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(1, Math.trunc(parsed)) : fallback;
}

/** Keeps only the shared index keys, so a PUT never carries anything else. */
function normalizeRagSettings(settings: Partial<RagSettings> | null | undefined): DraftRagSettings {
  const defaults = DEFAULT_RAG_SETTINGS;
  return {
    allow_local_filesystem_access: settings?.allow_local_filesystem_access ?? defaults.allow_local_filesystem_access,
    vector_collection_name: settings?.vector_collection_name?.trim() || defaults.vector_collection_name,
    chunk_size: positiveInteger(settings?.chunk_size, defaults.chunk_size),
    chunk_overlap: positiveInteger(settings?.chunk_overlap, defaults.chunk_overlap),
    embedding_batch_size: positiveInteger(settings?.embedding_batch_size, defaults.embedding_batch_size),
    vector_index_metric: settings?.vector_index_metric || defaults.vector_index_metric,
    vector_index_type: settings?.vector_index_type || defaults.vector_index_type,
    vector_stream_batch_size: positiveInteger(settings?.vector_stream_batch_size, defaults.vector_stream_batch_size),
    embedding_offline_mode: settings?.embedding_offline_mode ?? defaults.embedding_offline_mode,
  };
}

function formatTokenCount(tokens: number): string {
  if (tokens >= 1_000_000) return `${tokens / 1_000_000}M tokens`;
  if (tokens >= 1_000) return `${Math.round(tokens / 1_000)}K tokens`;
  return `${tokens} tokens`;
}

/**
 * Shared Providers & Models resources: the model catalog (read-only), provider credentials and the
 * shared document index settings. Per-agent model and retrieval choices live in the Agent Editor.
 */
@Component({
  selector: 'app-providers-resources',
  standalone: true,
  imports: [
    StatusMessageComponent,
    ProviderCatalogPanelComponent,
    ProviderCredentialsPanelComponent,
    RagResourcesPanelComponent,
  ],
  templateUrl: './providers-resources.component.html',
  styleUrl: './providers-resources.component.scss',
})
export class ProvidersResourcesComponent implements OnInit, OnDestroy {
  readonly appState = inject(AppStateService);
  private readonly modelConfigState = inject(ModelConfigStateService);
  private readonly modelPullJobs = inject(ModelPullJobService);

  readonly isLoading = signal(true);
  readonly statusMessage = signal('');
  readonly statusTone = computed(() => resolveStatusTone(this.statusMessage()));

  // Catalog
  readonly source = signal<CatalogProvider>(LOCAL_SOURCE);
  readonly modelSearchQuery = signal('');
  readonly visibleModelLimit = signal(MODEL_BATCH_SIZE);
  readonly activeFilters = signal<Record<ModelFilterKey, boolean>>({ ...NO_FILTERS });
  readonly catalogProviderInFlight = signal<CatalogProvider | null>(null);
  readonly modelPullProgress = signal<Record<string, ModelPullProgressState>>({});

  readonly localModels = computed(() => this.modelConfigState.data()?.local_models ?? []);
  readonly localCatalog = computed(() => this.modelConfigState.data()?.local_catalog ?? EMPTY_LOCAL_CATALOG);
  readonly cloudProviders = computed(() => this.modelConfigState.data()?.cloud_providers ?? []);
  readonly useCloud = computed(() => this.source() !== LOCAL_SOURCE);
  readonly selectedCloudProvider = computed(() =>
    this.cloudProviders().find((provider) => provider.id === this.source()) ?? null,
  );
  readonly sourceOptions = computed<{ value: CatalogProvider; label: string }[]>(() => [
    { value: LOCAL_SOURCE, label: 'Local (Ollama)' },
    ...this.cloudProviders().map((provider) => ({
      value: provider.id,
      label: provider.display_name?.trim() || provider.id,
    })),
  ]);

  readonly filteredLocalModels = computed(() => {
    const query = this.modelSearchQuery().trim().toLowerCase();
    return this.localModels().filter((model) => modelMatchesFilters(model, query, this.activeFilters()));
  });
  readonly filteredCloudModels = computed(() => {
    const query = this.modelSearchQuery().trim().toLowerCase();
    const names = this.selectedCloudProvider()?.models.map((model) => model.id) ?? [];
    return query ? names.filter((name) => name.toLowerCase().includes(query)) : names;
  });
  readonly visibleLocalModels = computed(() => this.filteredLocalModels().slice(0, this.visibleModelLimit()));
  readonly visibleCloudModels = computed(() => this.filteredCloudModels().slice(0, this.visibleModelLimit()));
  readonly filteredModelCount = computed(() =>
    this.useCloud() ? this.filteredCloudModels().length : this.filteredLocalModels().length,
  );
  readonly visibleModelRangeStart = computed(() => (this.filteredModelCount() ? 1 : 0));
  readonly visibleModelRangeEnd = computed(() =>
    this.useCloud() ? this.visibleCloudModels().length : this.visibleLocalModels().length,
  );
  readonly cloudCatalogMessage = computed(() => {
    const provider = this.selectedCloudProvider();
    if (!provider) return '';
    if (provider.catalog_message) return provider.catalog_message;
    if (provider.catalog_status === 'available') return 'Live provider catalog loaded.';
    if (provider.catalog_status === 'unavailable') {
      return 'Provider catalog is temporarily unavailable; configured agent models remain available.';
    }
    return 'No provider models were reported.';
  });
  readonly catalogEmptyMessage = computed(() => {
    const query = this.modelSearchQuery().trim();
    if (this.isLoading()) return 'Loading model catalog...';
    if (this.useCloud()) {
      if (this.filteredCloudModels().length) return '';
      return query ? `No cloud models match "${query}".` : 'No cloud models are available for this provider.';
    }
    if (!this.localModels().length) return 'No local model catalog entries available.';
    if (this.filteredLocalModels().length) return '';
    return query ? `No models match "${query}".` : 'No models match the active filters.';
  });

  // Credentials
  readonly credentialProviderOptions = computed<CredentialProviderOption[]>(() => {
    const grouped = new Map<AccessKeyProvider, ModelConfigStateResponse['cloud_providers']>();
    for (const provider of this.cloudProviders()) {
      grouped.set(provider.credential_scope, [...(grouped.get(provider.credential_scope) ?? []), provider]);
    }
    return Array.from(grouped.entries()).map(([provider, descriptors]) => {
      const labels = descriptors.map((descriptor) => descriptor.display_name.trim()).filter((label) => !!label);
      const label = labels.length > 1 ? labels.join(' / ') : labels[0] || provider;
      const description = labels.length > 1
        ? `This credential is used by ${labels.join(' and ')} cloud models.`
        : `Manage the API keys used to access ${label} cloud models.`;
      return { provider, label, description, logo: PROVIDER_LOGOS[provider] ?? null };
    });
  });
  readonly credentialProviderLabels = computed(() =>
    Object.fromEntries(this.credentialProviderOptions().map((option) => [option.provider, option.label])) as
      Partial<Record<AccessKeyProvider, string>>,
  );
  readonly selectedCredentialProvider = computed<AccessKeyProvider | null>(
    () => this.selectedCloudProvider()?.credential_scope ?? null,
  );

  // Shared RAG index settings
  readonly ragSettings = computed(() => normalizeRagSettings(this.modelConfigState.data()?.rag_settings));
  readonly draftRagSettings = signal<DraftRagSettings>({ ...DEFAULT_RAG_SETTINGS });
  readonly ragSettingsModalOpen = signal(false);
  readonly activeRagSettingsSection = signal<RagSettingsSectionKey>(DEFAULT_RAG_SECTION);
  readonly isRagSaving = signal(false);
  readonly ragSettingsSaveError = signal('');
  readonly embeddingRuntime = computed(() => this.modelConfigState.data()?.embedding_runtime ?? null);
  readonly embeddingIndex = computed(() => this.modelConfigState.data()?.embedding_index ?? null);
  readonly currentRagModelLabel = computed(
    () => this.embeddingRuntime()?.model_display_name || 'Granite Embedding Small English R2',
  );
  readonly isRagDirty = computed(
    () => JSON.stringify(this.ragSettings()) !== JSON.stringify(this.draftRagSettings()),
  );
  readonly ragSettingsValidationMessage = computed(() => {
    const draft = this.draftRagSettings();
    return draft.chunk_overlap >= draft.chunk_size ? 'Chunk overlap must be smaller than chunk size.' : '';
  });
  readonly ragSettingsStatus = computed(() => {
    if (this.ragSettingsValidationMessage()) return this.ragSettingsValidationMessage();
    if (this.ragSettingsSaveError()) return this.ragSettingsSaveError();
    if (this.isRagSaving()) return 'Saving changes...';
    return this.isRagDirty() ? 'Changes save automatically.' : 'All changes saved.';
  });

  private ragSaveTimer: ReturnType<typeof setTimeout> | null = null;

  @HostListener('document:keydown.escape')
  closeRagSettingsOnEscape(): void {
    if (this.ragSettingsModalOpen()) this.closeRagSettingsPanel();
  }

  async ngOnInit(): Promise<void> {
    await this.loadModelConfig();
    await this.ensureSelectedCatalog();
  }

  ngOnDestroy(): void {
    this.clearRagSaveTimer();
    if (this.isRagDirty() && !this.isRagSaving() && !this.ragSettingsValidationMessage()) {
      void this.saveRagSettings();
    }
  }

  async loadModelConfig(): Promise<void> {
    this.isLoading.set(true);
    try {
      this.applyState(await this.modelConfigState.load(true));
      this.statusMessage.set('');
    } catch (error) {
      this.statusMessage.set(formatUnknownError(error, 'Unable to load model settings.'));
    } finally {
      this.isLoading.set(false);
    }
  }

  private applyState(payload: ModelConfigStateResponse): void {
    this.modelConfigState.setFromApiState(payload);
    if (!this.ragSettingsModalOpen()) this.draftRagSettings.set({ ...this.ragSettings() });
  }

  // Catalog handlers

  private catalogStatus(provider: CatalogProvider): string {
    if (provider === LOCAL_SOURCE) return this.localCatalog().status;
    return this.cloudProviders().find((item) => item.id === provider)?.catalog_status || 'not_loaded';
  }

  private async ensureSelectedCatalog(): Promise<void> {
    if (this.catalogStatus(this.source()) === 'not_loaded') {
      await this.runCatalogOperation(this.source(), false);
    }
  }

  async refreshSelectedCatalog(): Promise<void> {
    await this.runCatalogOperation(this.source(), true);
  }

  private async runCatalogOperation(provider: CatalogProvider, forceRefresh: boolean): Promise<void> {
    if (this.catalogProviderInFlight()) return;
    this.catalogProviderInFlight.set(provider);
    try {
      const result = forceRefresh ? await refreshModelCatalog(provider) : await loadModelCatalog(provider);
      this.applyState(result.state);
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

  handleSourceChange(source: CatalogProvider): void {
    if (this.catalogProviderInFlight()) return;
    this.source.set(source);
    this.modelSearchQuery.set('');
    this.resetVisibleModelLimit();
    void this.ensureSelectedCatalog();
  }

  setModelSearchQuery(value: string): void {
    this.modelSearchQuery.set(value);
    this.resetVisibleModelLimit();
  }

  toggleFilter(key: ModelFilterKey): void {
    this.activeFilters.update((current) => ({ ...current, [key]: !current[key] }));
    this.resetVisibleModelLimit();
  }

  clearFilters(): void {
    this.activeFilters.set({ ...NO_FILTERS });
    this.resetVisibleModelLimit();
  }

  onModelCatalogScroll(event: Event): void {
    const shell = event.target instanceof HTMLElement ? event.target : null;
    if (!shell || this.visibleModelRangeEnd() >= this.filteredModelCount()) return;
    if (shell.scrollHeight - shell.scrollTop - shell.clientHeight <= 96) {
      this.visibleModelLimit.update((current) => Math.min(current + MODEL_BATCH_SIZE, this.filteredModelCount()));
    }
  }

  private resetVisibleModelLimit(): void {
    this.visibleModelLimit.set(MODEL_BATCH_SIZE);
  }

  // Model pull jobs

  async pullModelByName(modelName: string): Promise<void> {
    const candidate = modelName.trim();
    if (!candidate) return;

    this.appState.updateDiliAgent({ isPulling: true });
    this.statusMessage.set(`[INFO] Pulling '${candidate}' from Ollama...`);
    let failureMessage = '';
    try {
      const { completedModels } = await this.modelPullJobs.pullModels(
        [candidate],
        (name, progress) => this.updateModelPullProgress(name, progress),
      );
      this.statusMessage.set(`[INFO] Model available locally: ${completedModels[0] ?? candidate}.`);
    } catch (error) {
      const description = error instanceof Error ? error.message : 'Failed to pull selected models.';
      failureMessage = description.startsWith('[ERROR]') ? description : `[ERROR] ${description}`;
    } finally {
      await this.loadModelConfig();
      if (failureMessage) this.statusMessage.set(failureMessage);
      this.appState.updateDiliAgent({ isPulling: false });
    }
  }

  private updateModelPullProgress(modelName: string, progress: ModelPullProgressState | null): void {
    this.modelPullProgress.update((current) => {
      if (progress !== null) return { ...current, [modelName]: progress };
      const { [modelName]: removed, ...rest } = current;
      void removed;
      return rest;
    });
  }

  progressForModel(name: string): ModelPullProgressState | null {
    return this.modelPullProgress()[name] || null;
  }

  // Catalog row labels

  cloudModelContextLabel(modelName: string): string {
    const descriptor: CloudModelDescriptor | undefined = this.selectedCloudProvider()?.models.find(
      (model) => model.id === modelName,
    );
    const tokens = descriptor?.model_capabilities?.context_window_tokens;
    return tokens ? formatTokenCount(tokens) : 'Provider default';
  }

  modelSizeLabel(modelName: string): string {
    const match = modelName.match(/:(\d+(?:\.\d+)?)([mb])(?:$|[-_])/i);
    return match ? `${match[1]}${match[2].toUpperCase()}` : 'Unknown';
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
    if (!model.recommended_for_local_extraction) return null;
    return model.recommended_rank === 0 ? 'Recommended fast extractor' : 'Recommended backup extractor';
  }

  // Shared RAG settings

  openRagSettingsPanel(): void {
    this.draftRagSettings.set({ ...this.ragSettings() });
    this.ragSettingsSaveError.set('');
    this.activeRagSettingsSection.set(DEFAULT_RAG_SECTION);
    this.ragSettingsModalOpen.set(true);
  }

  closeRagSettingsPanel(): void {
    this.clearRagSaveTimer();
    if (this.isRagDirty() && !this.ragSettingsValidationMessage()) void this.saveRagSettings();
    this.ragSettingsModalOpen.set(false);
    this.activeRagSettingsSection.set(DEFAULT_RAG_SECTION);
  }

  setActiveRagSettingsSection(section: RagSettingsSectionKey): void {
    this.activeRagSettingsSection.set(section);
  }

  setDraftRagNumber(key: RagNumberKey, value: string): void {
    const parsed = Number.parseInt(value, 10);
    this.updateRagDraft({ [key]: Number.isFinite(parsed) ? Math.max(0, parsed) : 0 });
  }

  setDraftRagText(key: RagTextKey, value: string): void {
    this.updateRagDraft({ [key]: value });
  }

  setDraftRagBoolean(key: RagBooleanKey, value: boolean): void {
    this.updateRagDraft({ [key]: value });
  }

  private updateRagDraft(patch: Partial<DraftRagSettings>): void {
    this.draftRagSettings.update((previous) => ({ ...previous, ...patch }));
    this.ragSettingsSaveError.set('');
    this.scheduleRagSave();
  }

  async saveRagSettings(): Promise<void> {
    this.clearRagSaveTimer();
    if (!this.isRagDirty() || this.isRagSaving() || this.ragSettingsValidationMessage()) return;
    this.isRagSaving.set(true);
    try {
      const payload = await updateModelConfigState({ rag_settings: { ...this.draftRagSettings() } });
      this.modelConfigState.applyPersistedResponse(payload);
      if (!this.ragSettingsModalOpen() && !this.isRagDirty()) {
        this.draftRagSettings.set({ ...this.ragSettings() });
      }
      this.statusMessage.set('RAG settings saved.');
      this.ragSettingsSaveError.set('');
    } catch (error) {
      const message = formatUnknownError(error, 'Unable to save RAG settings.');
      this.statusMessage.set(message);
      this.ragSettingsSaveError.set(message);
    } finally {
      this.isRagSaving.set(false);
    }
    if (!this.ragSettingsSaveError() && this.isRagDirty()) this.scheduleRagSave();
  }

  private scheduleRagSave(): void {
    this.clearRagSaveTimer();
    if (!this.isRagDirty() || this.ragSettingsValidationMessage() || this.isLoading()) return;
    this.ragSaveTimer = setTimeout(() => {
      this.ragSaveTimer = null;
      void this.saveRagSettings();
    }, RAG_AUTOSAVE_DELAY_MS);
  }

  private clearRagSaveTimer(): void {
    if (this.ragSaveTimer === null) return;
    clearTimeout(this.ragSaveTimer);
    this.ragSaveTimer = null;
  }
}
