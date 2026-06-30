import { buildClinicalPayload } from './utils';
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
      temperature: 0,
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
      temperature: 0,
      reasoning: false,
    };

    const payload = buildClinicalPayload(form, settings);
    expect(payload.selected_model_providers).toEqual(['ollama']);
  });
});
