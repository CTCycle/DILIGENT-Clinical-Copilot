// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ModelConfigPageComponent } from './model-config-page.component';
import { AccessKeyRecord, ModelConfigPersistResponse, ModelConfigStateResponse } from '../../core/models/types';
import { AccessKeyModalComponent } from './components/access-key-modal.component';

describe('ModelConfigPageComponent', () => {
  let fixture: ComponentFixture<ModelConfigPageComponent>;
  let component: ModelConfigPageComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelConfigPageComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(ModelConfigPageComponent);
    component = fixture.componentInstance;
  });

  it('uses the supported OpenAI provider logo asset', () => {
    expect(component.providerLogo('openai')).toEqual({
      src: '/logos/openai-blossom-light.svg',
      alt: 'OpenAI logo',
    });
  });

  it('requires and saves explicit cloud role assignments after runtime toggle', async () => {
    const persistSpy = vi
      .spyOn(component as unknown as { persistConfigPatch: (...args: unknown[]) => Promise<boolean> }, 'persistConfigPatch')
      .mockResolvedValue(true);

    component.draftConfig.set({
      useCloudServices: false,
      provider: 'openai',
      cloudModel: 'gpt-4.1-mini',
      clinicalModel: 'gpt-oss:20b',
      textExtractionModel: 'qwen3:14b',
      revisionModel: 'gpt-oss:20b',
      timelineModel: 'qwen3:14b',
    });

    component.handleCloudSwitchChange(true);
    expect(component.draftConfig()).toMatchObject({
      clinicalModel: '',
      textExtractionModel: '',
      revisionModel: '',
      timelineModel: '',
    });
    component.draftConfig.update((previous) => ({
      ...previous,
      clinicalModel: 'gpt-4.1-mini',
      textExtractionModel: 'gpt-4.1-mini',
      revisionModel: 'gpt-4.1-mini',
      timelineModel: 'gpt-4.1-mini',
    }));

    await component.handleSaveConfiguration();

    expect(persistSpy).toHaveBeenCalledWith(
      {
        use_cloud_services: true,
        llm_provider: 'openai',
        cloud_model: 'gpt-4.1-mini',
        clinical_model: 'gpt-4.1-mini',
        text_extraction_model: 'gpt-4.1-mini',
        revision_model: 'gpt-4.1-mini',
        timeline_model: 'gpt-4.1-mini',
      },
      'Configuration saved.',
      true,
    );
  });

  it('switches providers without retaining stale cloud selections or search', () => {
    component.draftConfig.set({
      useCloudServices: true,
      provider: 'openai',
      cloudModel: 'gpt-4.1-mini',
      clinicalModel: 'gpt-4.1-mini',
      textExtractionModel: 'gpt-4.1-mini',
      revisionModel: 'gpt-4.1-mini',
      timelineModel: 'gpt-4.1-mini',
    });
    component.setModelSearchQuery('gpt');

    component.handleProviderChange('deepseek');

    expect(component.draftProvider()).toBe('deepseek');
    expect(component.draftCloudModel()).toBeNull();
    expect(component.draftConfig().clinicalModel).toBe('');
    expect(component.draftConfig().textExtractionModel).toBe('');
    expect(component.draftConfig().revisionModel).toBe('');
    expect(component.draftConfig().timelineModel).toBe('');
    expect(component.modelSearchQuery()).toBe('');
  });

  it('applies a persistence response without replacing catalog state', () => {
    const localCatalog = [{
      name: 'qwen3.5:2b',
      family: 'qwen',
      description: 'local',
      available_in_ollama: true,
      recommended_for_local_extraction: true,
      recommended_rank: 0,
    }];
    const cloudCatalog = component.cloudProviders();
    const payload: ModelConfigPersistResponse = {
      use_cloud_services: false,
      llm_provider: 'openai',
      cloud_model: null,
      clinical_model: 'qwen3.5:2b',
      text_extraction_model: 'qwen3.5:2b',
      revision_model: 'qwen3.5:2b',
      timeline_model: 'qwen3.5:2b',
      reasoning_level: 'medium',
      ollama_seed: 42,
      rag_settings: component.ragSettings(),
      updated_at: new Date().toISOString(),
    };
    const statePayload: ModelConfigStateResponse = {
      ...payload,
      local_models: localCatalog,
      cloud_providers: cloudCatalog,
      local_catalog: {
        status: 'available',
        updated_at: null,
        message: null,
      },
      embedding_runtime: {
        model_display_name: 'Embedding model',
        model_revision: 'test',
        device: 'cpu',
        cache_status: 'ready',
        loaded: false,
      },
      embedding_index: {
        status: 'ready',
        fingerprint: null,
        document_count: 0,
        chunk_count: 0,
        built_at: null,
      },
    };
    component.modelConfigState.setFromApiState(statePayload);
    const apply = (component as unknown as {
      applyPersistedConfigToState(
        response: ModelConfigPersistResponse,
        patch: { clinical_model: string },
        syncDraft: boolean,
      ): void;
    }).applyPersistedConfigToState.bind(component);

    apply(payload, { clinical_model: 'qwen3.5:2b' }, true);

    expect(component.localModels()).toBe(localCatalog);
    expect(component.cloudProviders()).toBe(cloudCatalog);
  });

  it('coalesces refresh clicks and preserves the refresh error state', async () => {
    component.draftConfig.set({
      useCloudServices: true,
      provider: 'openai',
      cloudModel: 'gpt-4.1-mini',
      clinicalModel: 'gpt-4.1-mini',
      textExtractionModel: 'gpt-4.1-mini',
      revisionModel: 'gpt-4.1-mini',
      timelineModel: 'gpt-4.1-mini',
    });
    vi.spyOn(component as unknown as { applyConfigToState: (...args: unknown[]) => void }, 'applyConfigToState')
      .mockImplementation(() => undefined);
    let resolveRefresh!: (value: Response) => void;
    const refreshPromise = new Promise<Response>((resolve) => { resolveRefresh = resolve; });
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockReturnValue(refreshPromise as never);

    const first = component.refreshSelectedCatalog();
    await Promise.resolve();
    const second = component.refreshSelectedCatalog();

    expect(component.catalogProviderInFlight()).toBe('openai');
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    resolveRefresh(new Response(JSON.stringify({
      outcome: 'failed',
      catalog_provider: 'openai',
      error: 'Provider unavailable',
      state: {},
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    await Promise.all([first, second]);
    fetchSpy.mockRestore();

    expect(component.catalogProviderInFlight()).toBeNull();
    expect(component.statusMessage()).toContain('Provider unavailable');
  });

  it('persists only backend-supported RAG fields and preserves hybrid weights', async () => {
    const persistSpy = vi
      .spyOn(component as unknown as { persistConfigPatch: (...args: unknown[]) => Promise<boolean> }, 'persistConfigPatch')
      .mockResolvedValue(true);

    const draftWithRuntimeOnlyDevice = {
      allow_local_filesystem_access: true,
      vector_collection_name: 'documents',
      chunk_size: 1024,
      chunk_overlap: 128,
      embedding_batch_size: 64,
      use_hybrid_search: true,
      use_reranking: true,
      retrieval_candidate_count: 40,
      retrieval_selected_count: 6,
      reranker_model: 'lightweight-balanced-v1',
      hybrid_vector_weight: 0.65,
      hybrid_text_weight: 0.35,
      vector_index_metric: 'cosine',
      vector_index_type: 'IVF_FLAT',
      vector_stream_batch_size: 250,
      embedding_offline_mode: false,
      embedding_device: 'cuda',
    };
    component.draftRagSettings.set(draftWithRuntimeOnlyDevice);
    component.ragSettingsModalOpen.set(true);

    await component.saveRagSettings();

    expect(persistSpy).toHaveBeenCalledWith(
      {
        rag_settings: {
          allow_local_filesystem_access: true,
          vector_collection_name: 'documents',
          chunk_size: 1024,
          chunk_overlap: 128,
          embedding_batch_size: 64,
          use_hybrid_search: true,
          use_reranking: true,
          retrieval_candidate_count: 40,
          retrieval_selected_count: 6,
          reranker_model: 'lightweight-balanced-v1',
          hybrid_vector_weight: 0.65,
          hybrid_text_weight: 0.35,
          vector_index_metric: 'cosine',
          vector_index_type: 'IVF_FLAT',
          vector_stream_batch_size: 250,
          embedding_offline_mode: false,
        },
      },
      'RAG settings saved.',
      false,
      'rag',
    );
    expect(component.ragSettingsModalOpen()).toBe(true);
    component.draftRagSettings.set(component.ragSettings());
  });

  it('validates hybrid weights while hybrid search is enabled', () => {
    component.draftRagSettings.update((previous) => ({
      ...previous,
      use_hybrid_search: true,
      hybrid_vector_weight: 0,
      hybrid_text_weight: 0,
    }));

    expect(component.ragSettingsValidationMessage()).toBe(
      'At least one hybrid search weight must be greater than 0.',
    );

    component.setDraftRagFloat('hybrid_vector_weight', '0.65');

    expect(component.draftRagSettings().hybrid_vector_weight).toBe(0.65);
    expect(component.ragSettingsValidationMessage()).toBe('');
  });

  it('persists a pending reasoning change immediately when the page is destroyed', () => {
    vi.useFakeTimers();
    const persistSpy = vi
      .spyOn(component as unknown as { persistConfigPatch: (...args: unknown[]) => Promise<boolean> }, 'persistConfigPatch')
      .mockResolvedValue(true);

    component.handleReasoningLevelChange(2);
    fixture.destroy();
    vi.advanceTimersByTime(300);

    expect(persistSpy).toHaveBeenCalledWith(
      { reasoning_level: 'medium' },
      'Reasoning preference saved.',
      false,
      'reasoning',
    );
    vi.useRealTimers();
  });
});

describe('AccessKeyModalComponent', () => {
  let fixture: ComponentFixture<AccessKeyModalComponent>;
  let component: AccessKeyModalComponent;
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  const key = (id: number, isActive: boolean): AccessKeyRecord => ({
    id,
    provider: 'openai',
    fingerprint: `fingerprint-${id}`,
    is_active: isActive,
    created_at: '2026-07-01T10:00:00Z',
    updated_at: '2026-07-01T10:00:00Z',
    last_used_at: null,
  });

  const jsonResponse = (body: unknown) => ({
    ok: true,
    status: 200,
    headers: new Headers({ 'content-type': 'application/json' }),
    text: async () => JSON.stringify(body),
  });

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AccessKeyModalComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(AccessKeyModalComponent);
    component = fixture.componentInstance;
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it('loads keys into signal state and keeps active keys first', async () => {
    fetchSpy.mockResolvedValue(jsonResponse([key(1, false), key(2, true)]) as Response);

    await component.loadKeys();

    expect(component.isLoading()).toBe(false);
    expect(component.hasKeys()).toBe(true);
    expect(component.sortedKeys().map((item) => item.id)).toEqual([2, 1]);
  });

  it('validates short key input without starting a save', async () => {
    component.newKeyValue.set('too-short');

    await component.addKey();

    expect(component.errorMessage()).toBe('Access keys must be at least 16 characters.');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('activates, deletes, and signal-updates key rows and visibility', async () => {
    const inactive = key(1, false);
    const active = key(2, true);
    component.keys.set([inactive, active]);
    const emitSpy = vi.spyOn(component.keysChanged, 'emit');
    fetchSpy
      .mockResolvedValueOnce(jsonResponse({ ...inactive, is_active: true, updated_at: '2026-07-02T10:00:00Z' }) as Response)
      .mockResolvedValueOnce(jsonResponse({ status: 'ok', deleted: true }) as Response);

    component.toggleVisibility(1);
    expect(component.fingerprintLabel(inactive)).toContain('fp:');

    await component.activateKey(1);
    expect(component.keys().find((item) => item.id === 1)?.is_active).toBe(true);
    expect(component.keys().find((item) => item.id === 2)?.is_active).toBe(false);

    await component.deleteKey(1);
    expect(component.keys().map((item) => item.id)).toEqual([2]);
    expect(component.isSaving()).toBe(false);
    expect(emitSpy).toHaveBeenCalledTimes(2);
  });
});
