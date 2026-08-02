import { buildClinicalPayload, isRecord } from './utils';
import { ClinicalFormState, RuntimeSettings } from './models/types';

describe('buildClinicalPayload', () => {
  it('uses the configured cloud provider when cloud runtime is enabled', () => {
    const form: ClinicalFormState = {
      patientName: 'John',
      visitDate: '2025-01-02',
      patientImageDataUrl: null,
      clinicalInput: 'test input',
      useRag: false,
    };
    const settings: RuntimeSettings = {
      useCloudServices: true,
      provider: 'openai',
      cloudModel: null,
      textExtractionModel: 'x',
      clinicalModel: 'y',
      reasoning: false,
    };

    const payload = buildClinicalPayload(form, settings);
    expect(payload.selected_model_providers).toEqual(['openai']);
  });

  it('uses ollama when local runtime is enabled', () => {
    const form: ClinicalFormState = {
      patientName: 'John',
      visitDate: '2025-01-02',
      patientImageDataUrl: null,
      clinicalInput: 'test input',
      useRag: false,
    };
    const settings: RuntimeSettings = {
      useCloudServices: false,
      provider: 'openai',
      cloudModel: null,
      textExtractionModel: 'qwen3.5:9b',
      clinicalModel: 'gpt-oss:20b',
      reasoning: false,
    };

    const payload = buildClinicalPayload(form, settings);
    expect(payload.selected_model_providers).toEqual(['ollama']);
  });
});

describe('isRecord', () => {
  it('narrows plain objects while excluding null and arrays', () => {
    expect(isRecord({ value: 1 })).toBe(true);
    expect(isRecord(null)).toBe(false);
    expect(isRecord(['value'])).toBe(false);
    expect(isRecord('value')).toBe(false);
  });
});
