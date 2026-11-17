// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable, computed, signal } from '@angular/core';

import {
  CatalogStatus,
  ModelConfigPersistResponse,
  ModelConfigStateResponse,
} from '../models/types';
import {
  fetchModelConfigState,
  loadModelCatalog,
  refreshModelCatalog,
} from '../services/model-config-api';
import { formatUnknownError } from '../utils';

export type ModelConfigLoadStatus = 'idle' | 'loading' | 'ready' | 'error';

export type ModelConfigResource = {
  status: ModelConfigLoadStatus;
  data: ModelConfigStateResponse | null;
  error: string | null;
};

const INITIAL_RESOURCE: ModelConfigResource = {
  status: 'idle',
  data: null,
  error: null,
};

@Injectable({ providedIn: 'root' })
export class ModelConfigStateService {
  private readonly resourceState = signal<ModelConfigResource>(INITIAL_RESOURCE);
  private loadRequest: Promise<ModelConfigStateResponse> | null = null;

  readonly resource = this.resourceState.asReadonly();
  readonly status = computed(() => this.resourceState().status);
  readonly data = computed(() => this.resourceState().data);
  readonly error = computed(() => this.resourceState().error);
  /** True while any provider catalog is being loaded or refreshed. */
  readonly catalogBusy = signal(false);
  private readonly catalogLoads = new Map<string, Promise<void>>();

  async load(force = false): Promise<ModelConfigStateResponse> {
    const current = this.resourceState();
    if (!force && current.status === 'ready' && current.data) {
      return current.data;
    }
    if (this.loadRequest) {
      return this.loadRequest;
    }

    // Keep the previous data while reloading so dependent dropdowns do not blank out.
    this.resourceState.set({ status: 'loading', data: current.data, error: null });
    this.loadRequest = fetchModelConfigState()
      .then((payload) => {
        this.setFromApiState(payload);
        return payload;
      })
      .catch((error: unknown) => {
        this.resourceState.set({
          status: 'error',
          data: this.resourceState().data,
          error: this.errorMessage(error),
        });
        throw error;
      })
      .finally(() => {
        this.loadRequest = null;
      });
    return this.loadRequest;
  }

  setFromApiState(payload: ModelConfigStateResponse): void {
    this.resourceState.set({
      status: 'ready',
      data: payload,
      error: null,
    });
  }

  applyPersistedResponse(payload: ModelConfigPersistResponse): ModelConfigStateResponse {
    const current = this.data();
    if (!current) {
      throw new Error('Model configuration must be loaded before applying an update.');
    }
    const next = { ...current, ...payload } as ModelConfigStateResponse;
    this.setFromApiState(next);
    return next;
  }

  /** Catalog status and message of one provider as last reported by the server. */
  catalogInfo(provider: string): { status: CatalogStatus; message: string | null } | null {
    const data = this.data();
    if (!data) return null;
    if (provider === 'ollama') {
      const local = data.local_catalog;
      return local ? { status: local.status, message: local.message } : null;
    }
    const found = data.cloud_providers?.find((item) => item.id === provider);
    return found?.catalog_status ? { status: found.catalog_status, message: found.catalog_message ?? null } : null;
  }

  /** Loads the catalog of every listed provider that was never loaded; reads stay cache-only otherwise. */
  async ensureCatalogs(providers: readonly string[]): Promise<void> {
    const wanted = Array.from(new Set(providers)).filter(
      (provider) => this.catalogInfo(provider)?.status === 'not_loaded',
    );
    this.catalogBusy.set(true);
    try {
      await Promise.all(wanted.map((provider) => this.loadOnce(provider)));
    } finally {
      this.catalogBusy.set(this.catalogLoads.size > 0);
    }
  }

  private loadOnce(provider: string): Promise<void> {
    const existing = this.catalogLoads.get(provider);
    if (existing) return existing;
    const request = loadModelCatalog(provider)
      .then((result) => this.setFromApiState(result.state))
      .catch(() => undefined)
      .finally(() => this.catalogLoads.delete(provider));
    this.catalogLoads.set(provider, request);
    return request;
  }

  /** Force-refreshes the listed providers; resolves to an error message per failed provider. */
  async refreshCatalogs(providers: readonly string[]): Promise<Record<string, string>> {
    const failures: Record<string, string> = {};
    this.catalogBusy.set(true);
    try {
      await Promise.all(
        Array.from(new Set(providers)).map(async (provider) => {
          try {
            const result = await refreshModelCatalog(provider);
            this.setFromApiState(result.state);
            if (result.outcome === 'failed') {
              failures[provider] = result.error || 'Unable to refresh the model catalog.';
            }
          } catch (error) {
            failures[provider] = formatUnknownError(error, 'Unable to refresh the model catalog.');
          }
        }),
      );
    } finally {
      this.catalogBusy.set(false);
    }
    return failures;
  }

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
