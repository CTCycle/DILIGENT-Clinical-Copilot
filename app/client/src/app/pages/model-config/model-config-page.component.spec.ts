import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelConfigPageComponent } from './model-config-page.component';
import { ModelConfigPersistResponse, ModelConfigStateResponse } from '../../core/models/types';

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
      .spyOn(component as unknown as { persistConfigPatch: (...args: unknown[]) => Promise<void> }, 'persistConfigPatch')
      .mockResolvedValue();

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

  it('cancels pending reasoning persistence when the page is destroyed', () => {
    vi.useFakeTimers();
    const persistSpy = vi
      .spyOn(component as unknown as { persistConfigPatch: (...args: unknown[]) => Promise<void> }, 'persistConfigPatch')
      .mockResolvedValue();

    component.handleReasoningLevelChange(2);
    fixture.destroy();
    vi.advanceTimersByTime(300);

    expect(persistSpy).not.toHaveBeenCalled();
    vi.useRealTimers();
  });
});
