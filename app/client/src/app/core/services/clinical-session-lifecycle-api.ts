// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { API_BASE_URL } from '../constants';
import {
  ClinicalRequestPayload,
  ClinicalSessionInput,
  ClinicalSessionInputWrite,
  ClinicalSessionState,
} from '../models/types';
import { requestJson } from './http-api';

const JSON_HEADERS = { 'Content-Type': 'application/json' };

function sessionUrl(sessionId: number, suffix = ''): string {
  return `${API_BASE_URL}/clinical/sessions/${encodeURIComponent(String(sessionId))}${suffix}`;
}

export function createClinicalSession(
  payload: ClinicalSessionInputWrite,
): Promise<ClinicalSessionState> {
  return requestJson<ClinicalSessionState>(`${API_BASE_URL}/clinical/sessions`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export function fetchClinicalSessionInput(sessionId: number): Promise<ClinicalSessionInput> {
  return requestJson<ClinicalSessionInput>(sessionUrl(sessionId, '/input'), {
    method: 'GET',
    cache: 'no-store',
  });
}

export function updateClinicalSessionInput(
  sessionId: number,
  payload: ClinicalSessionInputWrite,
  expectedUpdatedAt: string,
): Promise<ClinicalSessionState> {
  return requestJson<ClinicalSessionState>(sessionUrl(sessionId, '/input'), {
    method: 'PUT',
    headers: JSON_HEADERS,
    body: JSON.stringify({ ...payload, expected_updated_at: expectedUpdatedAt }),
  });
}

export function launchClinicalSession(
  sessionId: number,
  payload: ClinicalRequestPayload,
): Promise<ClinicalSessionState> {
  return requestJson<ClinicalSessionState>(sessionUrl(sessionId, '/launch'), {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export function fetchClinicalSessionStatus(sessionId: number): Promise<ClinicalSessionState> {
  return requestJson<ClinicalSessionState>(sessionUrl(sessionId, '/status'), {
    method: 'GET',
    cache: 'no-store',
    headers: { 'Cache-Control': 'no-cache, no-store, max-age=0', Pragma: 'no-cache' },
  });
}

export function cancelClinicalSession(sessionId: number): Promise<ClinicalSessionState> {
  return requestJson<ClinicalSessionState>(sessionUrl(sessionId, '/cancel'), { method: 'POST' });
}
