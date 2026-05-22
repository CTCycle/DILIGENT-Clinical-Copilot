import { buildClinicalPayload } from './utils';
import { ClinicalFormState, RuntimeSettings } from './models/types';

describe('buildClinicalPayload', () => {
  it('includes selected_model_providers from settings.provider', () => {
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
});
