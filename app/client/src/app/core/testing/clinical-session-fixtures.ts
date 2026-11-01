// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ClinicalSessionState } from '../models/types';

export type FetchCall = [RequestInfo | URL, RequestInit | undefined];

export function sessionState(overrides: Partial<ClinicalSessionState> = {}): ClinicalSessionState {
  return {
    session_id: 5,
    status: 'draft',
    patient_name: 'Ada',
    visit_date: '2026-10-01',
    created_at: '2026-10-01T08:00:00Z',
    updated_at: '2026-10-01T08:00:00Z',
    queued_at: null,
    started_at: null,
    finished_at: null,
    queue_position: null,
    progress: null,
    progress_stage: null,
    progress_message: null,
    stop_requested: false,
    error: null,
    has_report: false,
    version: 1,
    poll_interval: 0.25,
    ...overrides,
  };
}

export function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}
