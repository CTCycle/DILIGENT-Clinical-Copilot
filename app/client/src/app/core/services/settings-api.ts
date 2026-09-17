import { API_BASE_URL } from '../constants';
import {
  RuntimeSettingsCategory,
  RuntimeSettingsStateResponse,
  RuntimeSettingsUpdateRequest,
} from '../models/settings';
import { requestJson } from './http-api';

export async function fetchRuntimeSettings(): Promise<RuntimeSettingsStateResponse> {
  return requestJson<RuntimeSettingsStateResponse>(`${API_BASE_URL}/settings`, {
    method: 'GET',
    cache: 'no-store',
    headers: {
      'Cache-Control': 'no-cache, no-store, max-age=0',
      Pragma: 'no-cache',
    },
  });
}

export async function updateRuntimeSettings(
  payload: RuntimeSettingsUpdateRequest,
): Promise<RuntimeSettingsStateResponse> {
  return requestJson<RuntimeSettingsStateResponse>(`${API_BASE_URL}/settings`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function resetRuntimeSettingsCategory(
  category: RuntimeSettingsCategory,
): Promise<RuntimeSettingsStateResponse> {
  return requestJson<RuntimeSettingsStateResponse>(
    `${API_BASE_URL}/settings/reset/${encodeURIComponent(category)}`,
    { method: 'POST' },
  );
}
