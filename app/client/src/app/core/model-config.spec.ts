// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { describe, expect, it } from 'vitest';

import {
  buildRuntimeSettingsFromConfig,
  resolveCloudChoices,
  resolveCloudModel,
  resolveProvider,
} from './model-config';
import { ModelCapabilityMetadata, ModelConfigStateResponse } from './models/types';

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
