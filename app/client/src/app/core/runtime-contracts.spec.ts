// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { afterEach, describe, expect, it } from 'vitest';
import { API_BASE_URL, configureDesktopApiOrigin } from './constants';
import { buildRuntimeSettingsFromConfig, resolveCloudChoices, resolveCloudModel, resolveProvider } from './model-config';
import { ClinicalFormState, ModelCapabilityMetadata, ModelConfigStateResponse, RuntimeSettings } from './models/types';
import { buildClinicalPayload, isRecord } from './utils';
import {
  apiBaseUrlDefine,
  buildApiProxyConfig,
  parseEnvText,
  resolveApiBaseUrl,
  resolveRuntimeNetworkSettings,
  rewriteApiRequestPath,
} from '../../../scripts/runtime-config.mjs';

describe('runtime network configuration', () => {
  it('keeps default development API routing same-origin', () => {
    const settings = resolveRuntimeNetworkSettings({}, {});

    expect(settings).toEqual({
      apiHost: '127.0.0.1',
      apiPort: 7690,
      uiHost: '127.0.0.1',
      uiPort: 9847,
      apiBaseUrl: '/api',
    });
    expect(buildApiProxyConfig(settings.apiBaseUrl, settings.apiHost, settings.apiPort)).toEqual({
      '/api': {
        target: 'http://127.0.0.1:7690',
        secure: false,
        changeOrigin: true,
      },
    });
  });

  it('applies environment overrides and rewrites only the configured API path', () => {
    const fileEnv = parseEnvText(
      "FASTAPI_HOST=127.0.0.2\nFASTAPI_PORT=7701\nUI_PORT=9850\nVITE_API_BASE_URL='/backend-api'",
    );
    const settings = resolveRuntimeNetworkSettings({ UI_PORT: '9851' }, fileEnv);

    expect(settings).toMatchObject({
      apiHost: '127.0.0.2',
      apiPort: 7701,
      uiPort: 9851,
      apiBaseUrl: '/backend-api',
    });
    expect(buildApiProxyConfig(settings.apiBaseUrl, settings.apiHost, settings.apiPort)[
      '/backend-api'
    ].pathRewrite?.['^/backend-api']).toBe('/api');
    expect(rewriteApiRequestPath('/backend-api/settings?mode=full', settings.apiBaseUrl)).toBe(
      '/api/settings?mode=full',
    );
    expect(rewriteApiRequestPath('/backend-apix/settings', settings.apiBaseUrl)).toBeNull();
  });

  it('bypasses the local proxy for absolute API URLs', () => {
    const apiBaseUrl = resolveApiBaseUrl('https://api.example.org/v1/');

    expect(apiBaseUrl).toBe('https://api.example.org/v1');
    expect(buildApiProxyConfig(apiBaseUrl, '127.0.0.1', 7690)).toEqual({});
    expect(rewriteApiRequestPath('/api/settings', apiBaseUrl)).toBeNull();
    expect(apiBaseUrlDefine(apiBaseUrl)).toBe(
      '__DILIGENT_API_BASE_URL__="https://api.example.org/v1"',
    );
  });

  it('pins desktop builds to the shell-managed /api route', () => {
    expect(resolveRuntimeNetworkSettings(
      { VITE_API_BASE_URL: 'https://api.example.org' },
      {},
      { desktopBuild: true },
    ).apiBaseUrl).toBe('/api');
  });

  it('rejects invalid ports and unsafe API URLs', () => {
    expect(() => resolveRuntimeNetworkSettings({ FASTAPI_PORT: '70000' }, {})).toThrow(
      /FASTAPI_PORT must be an integer/,
    );
    expect(() => resolveApiBaseUrl('//api.example.org/v1')).toThrow(/protocol-relative/);
    expect(() => resolveApiBaseUrl('ftp://api.example.org')).toThrow(/HTTP\(S\) URL/);
    expect(() => resolveApiBaseUrl('https://user:pass@api.example.org')).toThrow(/HTTP\(S\) URL/);
  });
});

describe('desktop API runtime configuration', () => {
  afterEach(() => {
    configureDesktopApiOrigin(null);
  });

  it('uses the shell supplied loopback origin for the packaged API', () => {
    configureDesktopApiOrigin('http://127.0.0.1:49152');
    expect(API_BASE_URL).toBe('http://127.0.0.1:49152/api');
  });

  it('rejects API origins outside the loopback HTTP boundary', () => {
    expect(() => configureDesktopApiOrigin('https://127.0.0.1:49152')).toThrow(
      /local loopback origin/,
    );
    expect(() => configureDesktopApiOrigin('http://example.org:49152')).toThrow(
      /local loopback origin/,
    );
    expect(() => configureDesktopApiOrigin('http://127.0.0.1:49152/api')).toThrow(
      /local loopback origin/,
    );
  });
});

function emptyCapabilityMetadata(): ModelCapabilityMetadata {
  return {
    semantic_model_id: null,
    aliases: [],
    endpoint_family: null,
    context_window_tokens: null,
    max_output_tokens: null,
    supports_chat: null,
    supports_streaming: null,
    supports_tools: null,
    tool_call_mode: 'structured',
    supports_tool_choice: null,
    supports_parallel_tool_calls: null,
    supports_structured_output: null,
    supports_json_mode: null,
    supports_native_json_schema: null,
    supports_reasoning: null,
    reasoning_levels: [],
    reasoning_parameter: '',
    reasoning_counts_toward_output_limit: null,
    reasoning_unsupported_parameters: [],
    requires_reasoning_content_for_tool_calls: null,
    requires_assistant_content_for_tool_calls: null,
    supports_temperature: null,
    supports_top_p: null,
    supports_usage_metadata: null,
    supports_finish_reason: null,
    evidence: 'fallback',
  };
}

function modelConfigPayload(): ModelConfigStateResponse {
  return {
    local_models: [],
    cloud_providers: [
      {
        id: 'backend-provider',
        display_name: 'Backend Provider',
        credential_scope: 'openai',
        capabilities: {
          chat: true,
          structured_output: true,
          reasoning: true,
          model_listing: true,
          embeddings: false,
          vision: false,
        },
        catalog_status: 'available',
        catalog_updated_at: null,
        catalog_message: null,
        models: [
          {
            id: 'backend-model',
            display_name: 'Backend Model',
            endpoint_family: null,
            capabilities: null,
            model_capabilities: emptyCapabilityMetadata(),
          },
        ],
      },
    ],
    local_catalog: { status: 'available', updated_at: null, message: null },
    use_cloud_services: true,
    llm_provider: 'backend-provider',
    cloud_model: 'backend-model',
    clinical_model: 'clinical-role-model',
    text_extraction_model: 'text-role-model',
    revision_model: 'revision-role-model',
    timeline_model: 'timeline-role-model',
    reasoning_level: 'high',
    ollama_seed: 42,
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
      reranker_model: 'balanced',
      hybrid_vector_weight: 0.7,
      hybrid_text_weight: 0.3,
      vector_index_metric: 'cosine',
      vector_index_type: 'IVF_FLAT',
      vector_stream_batch_size: 250,
      embedding_offline_mode: false,
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
    updated_at: null,
  };
}

describe('model configuration mapping', () => {
  it('preserves every backend role assignment independently', () => {
    const settings = buildRuntimeSettingsFromConfig(modelConfigPayload());

    expect(settings.provider).toBe('backend-provider');
    expect(settings.cloudModel).toBe('backend-model');
    expect(settings.clinicalModel).toBe('clinical-role-model');
    expect(settings.textExtractionModel).toBe('text-role-model');
    expect(settings.revisionModel).toBe('revision-role-model');
    expect(settings.timelineModel).toBe('timeline-role-model');
  });

  it('uses only provider and model choices supplied by the backend', () => {
    const choices = resolveCloudChoices({ 'backend-provider': ['backend-model'] });

    expect(choices).toEqual({ 'backend-provider': ['backend-model'] });
    expect(resolveProvider('unknown-provider', choices)).toBe('unknown-provider');
    expect(resolveProvider('', choices)).toBe('');
    expect(resolveCloudModel('backend-provider', 'backend-model', choices)).toBe('backend-model');
    expect(resolveCloudModel('backend-provider', 'unknown-model', choices)).toBeNull();
    expect(resolveCloudChoices(undefined)).toEqual({});
  });
});

describe('buildClinicalPayload', () => {
  it('uses the configured cloud provider when cloud runtime is enabled', () => {
    const form: ClinicalFormState = {
      patientName: 'John',
      visitDate: '2025-01-02',
      patientImageDataUrl: null,
      clinicalInput: 'test input',
    };
    const settings: RuntimeSettings = {
      useCloudServices: true,
      provider: 'openai',
      cloudModel: null,
      textExtractionModel: 'x',
      clinicalModel: 'y',
      revisionModel: 'y',
      timelineModel: 'x',
      reasoning: 'off',
    };

    const payload = buildClinicalPayload(form, settings, ['livertox'], null);
    expect(payload.selected_model_providers).toEqual(['openai']);
  });

  it('uses ollama when local runtime is enabled', () => {
    const form: ClinicalFormState = {
      patientName: 'John',
      visitDate: '2025-01-02',
      patientImageDataUrl: null,
      clinicalInput: 'test input',
    };
    const settings: RuntimeSettings = {
      useCloudServices: false,
      provider: 'openai',
      cloudModel: null,
      textExtractionModel: 'qwen3.5:9b',
      clinicalModel: 'gpt-oss:20b',
      revisionModel: 'gpt-oss:20b',
      timelineModel: 'qwen3.5:9b',
      reasoning: 'off',
    };

    const payload = buildClinicalPayload(form, settings, ['livertox'], null);
    expect(payload.selected_model_providers).toEqual(['ollama']);
  });

  it('carries the per-session evidence sources and model overrides, and no use_rag flag', () => {
    const form: ClinicalFormState = {
      patientName: 'John',
      visitDate: '2025-01-02',
      patientImageDataUrl: null,
      clinicalInput: 'test input',
    };
    const settings: RuntimeSettings = {
      useCloudServices: false,
      provider: '',
      cloudModel: null,
      textExtractionModel: 'qwen3.5:9b',
      clinicalModel: 'gpt-oss:20b',
      revisionModel: 'gpt-oss:20b',
      timelineModel: 'qwen3.5:9b',
      reasoning: 'off',
    };

    const llmOnly = buildClinicalPayload(form, settings, [], null);
    expect(llmOnly.evidence_sources).toEqual([]);
    expect(llmOnly.model_overrides).toBeNull();
    expect('use_rag' in llmOnly).toBe(false);

    const overridden = buildClinicalPayload(form, settings, ['dilirank', 'documents'], {
      clinical_model: 'other:7b',
    });
    expect(overridden.evidence_sources).toEqual(['dilirank', 'documents']);
    expect(overridden.model_overrides).toEqual({ clinical_model: 'other:7b' });
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
