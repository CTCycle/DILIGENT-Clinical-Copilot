import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelConfigPageComponent } from './model-config-page.component';

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

  it('builds a cloud save patch with role assignments after runtime toggle and temperature update', async () => {
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
      temperature: 0.7,
    });

    component.handleCloudSwitchChange(true);
    component.setTemperature('0.64');

    await component.handleSaveConfiguration();

    expect(persistSpy).toHaveBeenCalledWith(
      {
        use_cloud_services: true,
        llm_provider: 'openai',
        cloud_model: 'gpt-4.1-mini',
        clinical_model: 'gpt-oss:20b',
        text_extraction_model: 'qwen3:14b',
        ollama_temperature: 0.64,
        cloud_temperature: 0.64,
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
      temperature: 0.7,
    });
    component.setModelSearchQuery('gpt');

    component.handleProviderChange('deepseek');

    expect(component.draftProvider()).toBe('deepseek');
    expect(component.draftCloudModel()).toBeNull();
    expect(component.draftConfig().clinicalModel).toBe('');
    expect(component.draftConfig().textExtractionModel).toBe('');
    expect(component.modelSearchQuery()).toBe('');
  });
});
