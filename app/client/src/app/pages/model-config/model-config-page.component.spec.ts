import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelConfigPageComponent } from './model-config-page.component';
import { ModelConfigPersistResponse } from '../../core/models/types';

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

  it('builds a cloud save patch with role assignments after runtime toggle', async () => {
    const persistSpy = vi
      .spyOn(component as unknown as { persistConfigPatch: (...args: unknown[]) => Promise<void> }, 'persistConfigPatch')
      .mockResolvedValue();

    component.cloudChoices.set({
      openai: ['gpt-4.1-mini'],
      gemini: ['gemini-2.5-pro'],
      deepseek: [],
      anthropic: [],
      opencode_zen: [],
      opencode_go: [],
    });
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
    component.cloudChoices.set({
      openai: ['gpt-4.1-mini'],
      gemini: ['gemini-3.5-flash'],
      deepseek: ['deepseek-v4-flash'],
      anthropic: [],
      opencode_zen: [],
      opencode_go: [],
    });
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
    component.localModels.set(localCatalog);
    component.cloudProviders.set(cloudCatalog);
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
