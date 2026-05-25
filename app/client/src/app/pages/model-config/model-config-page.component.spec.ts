import { ComponentFixture, TestBed } from '@angular/core/testing';
import { vi } from 'vitest';

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

  it('builds a cloud save patch after runtime toggle and temperature update', async () => {
    const persistSpy = vi
      .spyOn(component as unknown as { persistConfigPatch: (...args: unknown[]) => Promise<void> }, 'persistConfigPatch')
      .mockResolvedValue();

    component.cloudChoices.set({
      openai: ['gpt-4.1-mini'],
      gemini: ['gemini-2.5-pro'],
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
        ollama_temperature: 0.64,
        cloud_temperature: 0.64,
      },
      'Configuration saved.',
      true,
    );
  });
});
