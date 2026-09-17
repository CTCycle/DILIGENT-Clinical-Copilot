import { describe, expect, it } from 'vitest';

import { RuntimeSettingsValues } from '../../../core/models/settings';
import {
  SETTINGS_SECTION_DEFINITIONS,
  validateRuntimeSettingsSection,
} from './operational-settings-page.component';

const VALUES: RuntimeSettingsValues = {
  general: { polling_interval: 1 },
  data: {
    drug_name_min_length: 3,
    drug_name_max_length: 200,
    drug_name_max_tokens: 8,
  },
  integrations: {
    livertox_download_timeout: 30,
    rxnav_request_timeout: 12,
    rxnav_max_concurrency: 10,
  },
  advanced: {
    default_llm_timeout: 3600,
    clinical_llm_timeout: 3600,
    livertox_llm_timeout: 3600,
    minimum_llm_timeout: 5,
    cloud_llm_timeout_cap: 1800,
    local_llm_timeout_cap: 45,
    max_excerpt_length: 8000,
  },
};

describe('operational settings definitions', () => {
  it('maps only the intended JSON-backed runtime categories', () => {
    expect(Object.keys(SETTINGS_SECTION_DEFINITIONS)).toEqual([
      'general',
      'data',
      'integrations',
      'advanced',
    ]);
    const paths = Object.values(SETTINGS_SECTION_DEFINITIONS)
      .flatMap((section) => section.fields.map((field) => field.jsonPath));
    expect(paths).toContain('jobs.polling_interval');
    expect(paths).toContain('ingestion.drug_name_max_tokens');
    expect(paths).toContain('runtime.rxnav_request_timeout');
    expect(paths).not.toContain('database.url');
    expect(paths.some((path) => path.includes('env'))).toBe(false);
  });

  it('enforces cross-field data and timeout constraints before save', () => {
    expect(validateRuntimeSettingsSection('general', VALUES)).toBe('');

    const invalidData: RuntimeSettingsValues = {
      ...VALUES,
      data: { ...VALUES.data, drug_name_min_length: 20, drug_name_max_length: 10 },
    };
    expect(validateRuntimeSettingsSection('data', invalidData)).toContain('cannot be smaller');

    const invalidAdvanced: RuntimeSettingsValues = {
      ...VALUES,
      advanced: {
        ...VALUES.advanced,
        minimum_llm_timeout: 60,
        local_llm_timeout_cap: 30,
      },
    };
    expect(validateRuntimeSettingsSection('advanced', invalidAdvanced)).toContain('Local LLM timeout cap');
  });
});
