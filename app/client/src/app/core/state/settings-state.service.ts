import { Injectable, computed, signal } from '@angular/core';

import {
  RuntimeSettingsCategory,
  RuntimeSettingsStateResponse,
  RuntimeSettingsUpdateRequest,
} from '../models/settings';
import {
  fetchRuntimeSettings,
  resetRuntimeSettingsCategory,
  updateRuntimeSettings,
} from '../services/settings-api';

export type RuntimeSettingsLoadStatus = 'idle' | 'loading' | 'ready' | 'error';

@Injectable({ providedIn: 'root' })
export class RuntimeSettingsStateService {
  private readonly state = signal<RuntimeSettingsStateResponse | null>(null);
  private readonly loadStatus = signal<RuntimeSettingsLoadStatus>('idle');
  private readonly loadError = signal<string | null>(null);
  private loadRequest: Promise<RuntimeSettingsStateResponse> | null = null;

  readonly data = this.state.asReadonly();
  readonly status = this.loadStatus.asReadonly();
  readonly error = this.loadError.asReadonly();
  readonly isLoading = computed(() => this.loadStatus() === 'loading');

  async load(force = false): Promise<RuntimeSettingsStateResponse> {
    const current = this.state();
    if (!force && this.loadStatus() === 'ready' && current) {
      return current;
    }
    if (this.loadRequest) {
      return this.loadRequest;
    }

    this.loadStatus.set('loading');
    this.loadError.set(null);
    this.loadRequest = fetchRuntimeSettings()
      .then((payload) => {
        this.apply(payload);
        return payload;
      })
      .catch((error: unknown) => {
        this.loadStatus.set('error');
        this.loadError.set(this.errorMessage(error));
        throw error;
      })
      .finally(() => {
        this.loadRequest = null;
      });
    return this.loadRequest;
  }

  async update(payload: RuntimeSettingsUpdateRequest): Promise<RuntimeSettingsStateResponse> {
    const response = await updateRuntimeSettings(payload);
    this.apply(response);
    return response;
  }

  async reset(category: RuntimeSettingsCategory): Promise<RuntimeSettingsStateResponse> {
    const response = await resetRuntimeSettingsCategory(category);
    this.apply(response);
    return response;
  }

  private apply(payload: RuntimeSettingsStateResponse): void {
    this.state.set(payload);
    this.loadStatus.set('ready');
    this.loadError.set(null);
  }

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
