// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { API_BASE_URL } from '../constants';
import {
  ClinicalDraftSnapshot,
  ClinicalDraftWrite,
} from '../models/types';
import { requestJson } from './http-api';

export function fetchClinicalDraft(): Promise<ClinicalDraftSnapshot> {
  return requestJson<ClinicalDraftSnapshot>(`${API_BASE_URL}/clinical/draft`, {
    method: 'GET',
    cache: 'no-store',
    headers: {
      'Cache-Control': 'no-cache, no-store, max-age=0',
      Pragma: 'no-cache',
    },
  });
}

export function saveClinicalDraft(
  payload: ClinicalDraftWrite,
): Promise<ClinicalDraftSnapshot> {
  return requestJson<ClinicalDraftSnapshot>(`${API_BASE_URL}/clinical/draft`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function discardClinicalDraft(
  expectedRevision: number,
  expectedDraftKey: string | null,
): Promise<ClinicalDraftSnapshot> {
  const query = new URLSearchParams({
    expected_revision: String(expectedRevision),
  });
  if (expectedDraftKey !== null) {
    query.set('expected_draft_key', expectedDraftKey);
  }
  return requestJson<ClinicalDraftSnapshot>(
    `${API_BASE_URL}/clinical/draft?${query.toString()}`,
    { method: 'DELETE', cache: 'no-store' },
  );
}
