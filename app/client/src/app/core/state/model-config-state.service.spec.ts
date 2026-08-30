import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelConfigStateResponse } from '../models/types';
import { ModelConfigStateService } from './model-config-state.service';

function modelConfigPayload(): ModelConfigStateResponse {
  return {
    local_models: [],
    cloud_providers: [],
    local_catalog: { status: 'available', updated_at: null, message: null },
    use_cloud_services: false,
    llm_provider: 'openai',
    cloud_model: null,
    clinical_model: 'clinical-model',
    text_extraction_model: 'text-model',
    revision_model: 'revision-model',
    timeline_model: 'timeline-model',
    reasoning_level: 'medium',
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

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('ModelConfigStateService', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    TestBed.configureTestingModule({});
  });

  it('hydrates exact backend roles and exposes the loaded resource', async () => {
    const payload = modelConfigPayload();
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(payload));
    const resource = TestBed.inject(ModelConfigStateService);

    await resource.load();

    expect(fetchSpy).toHaveBeenCalledWith(
      '/api/model-config',
      expect.objectContaining({ method: 'GET', cache: 'no-store' }),
    );
    expect(resource.status()).toBe('ready');
    expect(resource.settings()).toMatchObject({
      clinicalModel: 'clinical-model',
      textExtractionModel: 'text-model',
      revisionModel: 'revision-model',
      timelineModel: 'timeline-model',
    });
  });

  it('clears stale data and settings when a forced refresh fails', async () => {
    const payload = modelConfigPayload();
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(payload))
      .mockRejectedValueOnce(new Error('backend unavailable'));
    const resource = TestBed.inject(ModelConfigStateService);

    await resource.load();
    await expect(resource.load(true)).rejects.toThrow();

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(resource.status()).toBe('error');
    expect(resource.data()).toBeNull();
    expect(resource.settings()).toBeNull();
    expect(resource.error()).toContain('backend unavailable');
  });
});
