import { describe, expect, it } from 'vitest';

import {
  buildRuntimeSettingsFromConfig,
  resolveCloudChoices,
  resolveCloudModel,
  resolveProvider,
} from './model-config';
import { ModelConfigStateResponse } from './models/types';

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
            input_token_limit: null,
            output_token_limit: null,
            supports_thinking: null,
            supports_temperature: null,
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
      vector_stream_batch_size: 250,
      embedding_device: 'auto',
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
