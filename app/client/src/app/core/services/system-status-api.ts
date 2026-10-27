// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { API_BASE_URL } from '../constants';
import { requestJson } from './http-api';

export type MemoryUsage = {
  used_bytes: number;
  total_bytes: number;
};

export type SystemStatusResponse = {
  providers: { active: number; total: number };
  ollama: 'reachable' | 'unreachable';
  cpu_percent: number | null;
  ram: MemoryUsage | null;
  vram: MemoryUsage | null;
};

const STATUS_REQUEST_TIMEOUT_SECONDS = 4;

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  return requestJson<SystemStatusResponse>(
    `${API_BASE_URL}/system/status`,
    { method: 'GET', cache: 'no-store' },
    STATUS_REQUEST_TIMEOUT_SECONDS,
  );
}
