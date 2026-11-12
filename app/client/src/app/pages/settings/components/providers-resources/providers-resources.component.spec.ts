// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from 'vitest';

import { ModelConfigStateResponse } from '../../../core/models/types';
import { ProviderCatalogPanelComponent } from '../../model-config/components/provider-catalog-panel.component';
import { ProviderCredentialsPanelComponent } from '../../model-config/components/provider-credentials-panel.component';
import { RagResourcesPanelComponent } from '../../model-config/components/rag-resources-panel.component';
import { ProvidersResourcesComponent } from './providers-resources.component';

const SHARED_RAG_KEYS = [
  'allow_local_filesystem_access',
  'chunk_overlap',
  'chunk_size',
  'embedding_batch_size',
  'embedding_offline_mode',
  'vector_collection_name',
  'vector_index_metric',
  'vector_index_type',
  'vector_stream_batch_size',
];

function statePayload(): ModelConfigStateResponse {
  return {
    local_models: [{
      name: 'qwen3:8b',
      description: 'local',
      family: 'qwen',
      available_in_ollama: true,
      recommended_for_local_extraction: false,
      recommended_rank: null,
    }],
    cloud_providers: [],
    local_catalog: { status: 'available', updated_at: null, message: null },
    rag_settings: {
      allow_local_filesystem_access: true,
      vector_collection_name: 'documents',
      chunk_size: 512,
      chunk_overlap: 64,
      embedding_batch_size: 8,
      vector_index_metric: 'cosine',
      vector_index_type: 'IVF_FLAT',
      vector_stream_batch_size: 1024,
      embedding_offline_mode: false,
    },
    embedding_runtime: { model_display_name: 'Embedding', model_revision: 'test', device: 'cpu', cache_status: 'ready', loaded: false },
    embedding_index: { status: 'ready', fingerprint: null, document_count: 0, chunk_count: 0, built_at: null },
    updated_at: null,
  };
}

describe('ProvidersResourcesComponent', () => {
  let fixture: ComponentFixture<ProvidersResourcesComponent>;
  let component: ProvidersResourcesComponent;
  let fetchSpy: MockInstance<typeof fetch>;

  const calls = (method: string) =>
    fetchSpy.mock.calls.filter(([, init]) => (init?.method ?? 'GET') === method);

  beforeEach(async () => {
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
      const body = init?.method === 'PUT'
        ? { rag_settings: JSON.parse(String(init.body)).rag_settings, updated_at: null }
        : statePayload();
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
    });
    await TestBed.configureTestingModule({ imports: [ProvidersResourcesComponent] }).compileComponents();
    fixture = TestBed.createComponent(ProvidersResourcesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
    await vi.waitFor(() => expect(component.isLoading()).toBe(false));
    fixture.detectChanges();
  });

  afterEach(() => {
    fixture.destroy();
    vi.restoreAllMocks();
  });

  it('renders only the catalog, credentials and index panels', () => {
    const root = fixture.nativeElement as HTMLElement;
    expect(fixture.debugElement.query(By.directive(RagResourcesPanelComponent))).toBeTruthy();
    expect(fixture.debugElement.query(By.directive(ProviderCatalogPanelComponent))).toBeTruthy();
    expect(fixture.debugElement.query(By.directive(ProviderCredentialsPanelComponent))).toBeTruthy();
    expect(root.querySelector('main')).toBeNull();
    expect(root.textContent).toContain('qwen3:8b');
    expect(root.textContent).not.toContain('Runtime Source');
    expect(root.querySelector('app-model-role-action-button')).toBeNull();
  });

  it('saves a shared index change with only the shared keys', async () => {
    component.setDraftRagNumber('chunk_size', '1024');
    await component.saveRagSettings();

    const [[, init]] = calls('PUT');
    const body = JSON.parse(String(init?.body));
    expect(Object.keys(body)).toEqual(['rag_settings']);
    expect(Object.keys(body.rag_settings).sort()).toEqual(SHARED_RAG_KEYS);
    expect(body.rag_settings.chunk_size).toBe(1024);
    expect(component.ragSettings().chunk_size).toBe(1024);
    expect(component.isRagDirty()).toBe(false);
  });

  it('does not save an invalid chunk overlap', async () => {
    component.setDraftRagNumber('chunk_overlap', '600');
    await component.saveRagSettings();

    expect(calls('PUT')).toHaveLength(0);
    expect(component.ragSettingsStatus()).toBe('Chunk overlap must be smaller than chunk size.');
  });

  it('reloads the shared state when credentials change', async () => {
    const getsBefore = calls('GET').length;
    fixture.debugElement.query(By.directive(ProviderCredentialsPanelComponent))
      .componentInstance.keysChanged.emit();
    await vi.waitFor(() => expect(calls('GET').length).toBe(getsBefore + 1));
  });
});
