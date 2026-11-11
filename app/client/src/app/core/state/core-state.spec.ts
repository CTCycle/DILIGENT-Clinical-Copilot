// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AppStateService, resolvePageIdFromPath, resolvePathFromPage } from './app-state.service';
import { ModelConfigStateResponse } from '../models/types';
import { ModelConfigStateService } from './model-config-state.service';

describe('AppStateService', () => {
  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({});
  });

  it('starts with an empty form and no draft, and keeps editing state out of browser storage', () => {
    const service = TestBed.inject(AppStateService);

    expect(service.state().diliAgent).toMatchObject({
      draftSessionId: null,
      saveStatus: 'idle',
      isDirty: false,
    });
    expect(service.state().diliAgent.form.patientName).toBe('');

    service.updateDiliAgent({
      form: { ...service.state().diliAgent.form, patientName: 'Transient patient' },
      draftSessionId: 42,
      isDirty: true,
    });
    TestBed.flushEffects();

    expect(localStorage.length).toBe(0);
  });

  it('persists an explicit theme choice across service instances', () => {
    TestBed.inject(AppStateService).toggleTheme();
    const chosen = TestBed.inject(AppStateService).state().theme;

    TestBed.resetTestingModule();
    TestBed.configureTestingModule({});

    expect(TestBed.inject(AppStateService).state().theme).toBe(chosen);
  });

  it('maps configuration routes to Settings', () => {
    expect(resolvePageIdFromPath('/clinical-sessions?session=3&tab=preview')).toBe('clinical-sessions');
    expect(resolvePageIdFromPath('/settings/general')).toBe('settings');
    expect(resolvePageIdFromPath('/settings/advanced')).toBe('settings');
    expect(resolvePathFromPage('settings')).toBe('/settings/general');
  });
});

function modelConfigPayload(): ModelConfigStateResponse {
  return {
    local_models: [],
    cloud_providers: [],
    local_catalog: { status: 'available', updated_at: null, message: null },
    rag_settings: {
      allow_local_filesystem_access: true,
      vector_collection_name: 'documents',
      chunk_size: 1024,
      chunk_overlap: 128,
      embedding_batch_size: 64,
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

  it('hydrates the shared catalog and index state', async () => {
    const payload = modelConfigPayload();
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(payload));
    const resource = TestBed.inject(ModelConfigStateService);

    await resource.load();

    expect(fetchSpy).toHaveBeenCalledWith(
      '/api/model-config',
      expect.objectContaining({ method: 'GET', cache: 'no-store' }),
    );
    expect(resource.status()).toBe('ready');
    expect(resource.data()?.rag_settings.chunk_size).toBe(1024);
  });

  it('clears stale data when a forced refresh fails', async () => {
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
    expect(resource.error()).toContain('backend unavailable');
  });
});
