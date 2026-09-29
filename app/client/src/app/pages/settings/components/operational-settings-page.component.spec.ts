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
    text_extraction_batch_size: 4,
    text_extraction_max_concurrency: 2,
    retrieval_batch_size: 8,
    retrieval_max_concurrency: 4,
    clinical_assessment_batch_size: 2,
    clinical_assessment_max_concurrency: 2,
    min_best_score: 2,
    high_confidence_min_score: 8,
    high_confidence_min_margin: 3,
    moderate_confidence_min_score: 4,
    moderate_confidence_min_margin: 1,
  },
  integrations: {
    ncbi_contact_email: 'clinical-copilot@pharmagent.local',
    livertox_download_timeout: 30,
    livertox_archive: 'livertox_NBK547852.tar.gz',
    livertox_yield_interval: 25,
    livertox_skip_deterministic_ratio: 0.8,
    livertox_monograph_max_workers: 4,
    rxnav_request_timeout: 12,
    rxnav_max_concurrency: 10,
  },
  matching: {
    direct_confidence: 1,
    master_confidence: 0.92,
    synonym_confidence: 0.9,
    normalization_cache_limit: 10000,
    match_cache_limit: 5000,
    alias_cache_limit: 2000,
    min_confidence: 0.9,
    token_min_length: 4,
    catalog_index_limit: 75000,
    spelling_confidence: 0.94,
    spelling_min_query_length: 6,
    spelling_short_name_length: 8,
    spelling_short_max_distance: 1,
    spelling_long_max_distance: 2,
  },
  advanced: {
    default_llm_timeout: 3600,
    parser_llm_timeout: 3600,
    disease_llm_timeout: 3600,
    clinical_llm_timeout: 3600,
    livertox_llm_timeout: 3600,
    minimum_llm_timeout: 5,
    cloud_llm_timeout_cap: 1800,
    local_llm_timeout_cap: 45,
    ollama_server_start_timeout: 15,
    max_excerpt_length: 8000,
  },
};

describe('operational settings definitions', () => {
  it('maps every database-backed runtime category and excludes environment settings', () => {
    expect(Object.keys(SETTINGS_SECTION_DEFINITIONS)).toEqual([
      'general',
      'data',
      'integrations',
      'matching',
      'advanced',
    ]);
    const fields = Object.values(SETTINGS_SECTION_DEFINITIONS)
      .flatMap((section) => section.fields);
    expect(fields.map((field) => field.key)).toContain('clinical_assessment_batch_size');
    expect(fields.map((field) => field.key)).toContain('spelling_long_max_distance');
    expect(fields.map((field) => field.kind)).toContain('text');
    expect(fields.map((field) => field.key)).toContain('ncbi_contact_email');
    const ncbiField = SETTINGS_SECTION_DEFINITIONS.integrations.fields.find(
      (field) => field.key === 'ncbi_contact_email',
    );
    expect(ncbiField?.kind).toBe('email');
    expect(ncbiField?.scope).toBe('NCBI / LiverTox');
    expect(fields.map((field) => field.kind)).not.toContain('boolean');
    expect(fields.every((field) => !field.scope.toLowerCase().includes('env'))).toBe(true);
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

    const invalidEmail: RuntimeSettingsValues = {
      ...VALUES,
      integrations: { ...VALUES.integrations, ncbi_contact_email: 'not-an-email' },
    };
    expect(validateRuntimeSettingsSection('integrations', invalidEmail)).toContain('valid email');

    const validEmail: RuntimeSettingsValues = {
      ...VALUES,
      integrations: { ...VALUES.integrations, ncbi_contact_email: 'developer@example.org' },
    };
    expect(validateRuntimeSettingsSection('integrations', validEmail)).toBe('');
  });
});
