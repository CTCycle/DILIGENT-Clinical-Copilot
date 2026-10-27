// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { API_BASE_URL } from '../constants';
import { JobCancelResponse } from '../models/types';
import { requestJson } from './http-api';

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface JobSummary {
  job_id: string;
  job_type: string;
  status: JobStatus;
  progress: number | null;
  phase: string | null;
  message: string | null;
  error: string | null;
  started_at: number;
  finished_at: number | null;
  stop_requested: boolean;
  cancellable: boolean;
}

export interface JobSummaryListResponse {
  jobs: JobSummary[];
}

export function fetchJobs(): Promise<JobSummaryListResponse> {
  return requestJson<JobSummaryListResponse>(
    `${API_BASE_URL}/jobs`,
    { method: 'GET', headers: { 'Cache-Control': 'no-cache' } },
    10,
  );
}

export function cancelJob(jobId: string): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}`,
    { method: 'DELETE' },
  );
}
