import { Injectable, computed, signal } from '@angular/core';

import { buildRuntimeSettingsFromConfig } from '../model-config';
import {
  ModelConfigPersistResponse,
  ModelConfigStateResponse,
  RuntimeSettings,
} from '../models/types';
import { fetchModelConfigState } from '../services/model-config-api';

export type ModelConfigLoadStatus = 'idle' | 'loading' | 'ready' | 'error';

export type ModelConfigResource = {
  status: ModelConfigLoadStatus;
  data: ModelConfigStateResponse | null;
  settings: RuntimeSettings | null;
  error: string | null;
};

const INITIAL_RESOURCE: ModelConfigResource = {
  status: 'idle',
  data: null,
  settings: null,
  error: null,
};

@Injectable({ providedIn: 'root' })
export class ModelConfigStateService {
  private readonly resourceState = signal<ModelConfigResource>(INITIAL_RESOURCE);
  private loadRequest: Promise<ModelConfigStateResponse> | null = null;

  readonly resource = this.resourceState.asReadonly();
  readonly status = computed(() => this.resourceState().status);
  readonly data = computed(() => this.resourceState().data);
  readonly settings = computed(() => this.resourceState().settings);
  readonly error = computed(() => this.resourceState().error);

  async load(force = false): Promise<ModelConfigStateResponse> {
    const current = this.resourceState();
    if (!force && current.status === 'ready' && current.data) {
      return current.data;
    }
    if (this.loadRequest) {
      return this.loadRequest;
    }

    this.resourceState.set({
      status: 'loading',
      data: null,
      settings: null,
      error: null,
    });
    this.loadRequest = fetchModelConfigState()
      .then((payload) => {
        this.setFromApiState(payload);
        return payload;
      })
      .catch((error: unknown) => {
        this.resourceState.set({
          status: 'error',
          data: null,
          settings: null,
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
      settings: buildRuntimeSettingsFromConfig(payload),
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

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
