import { API_BASE_URL } from "../constants";
import {
  InspectionLiverToxOverrideRequest,
  InspectionRagUpdateRequest,
  InspectionRxNavOverrideRequest,
  InspectionUpdateConfigResponse,
  InspectionUpdateJobListResponse,
  InspectionUpdateJobStatusResponse,
} from "../models/inspection-types";
import { JobCancelResponse, JobStartResponse } from "../models/types";
import { requestJson } from "./http-api";

const INSPECTION_JOB_STATUS_TIMEOUT_SECONDS = 20;

export async function fetchInspectionUpdateJobs(): Promise<InspectionUpdateJobListResponse> {
  return requestJson<InspectionUpdateJobListResponse>(
    `${API_BASE_URL}/inspection/jobs`,
    {
      method: "GET",
      headers: { "Cache-Control": "no-cache" },
    },
  );
}

export async function fetchInspectionRxNavUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(
    `${API_BASE_URL}/inspection/rxnav/update-config`,
    { method: "GET" },
  );
}

export async function startInspectionRxNavUpdateJob(
  payload: InspectionRxNavOverrideRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/rxnav/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionRxNavUpdateJobStatus(
  jobId: string,
  timeoutSeconds: number = INSPECTION_JOB_STATUS_TIMEOUT_SECONDS,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/rxnav/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
  );
}

export async function cancelInspectionRxNavUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/rxnav/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionLiverToxUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(
    `${API_BASE_URL}/inspection/livertox/update-config`,
    { method: "GET" },
  );
}

export async function startInspectionLiverToxUpdateJob(
  payload: InspectionLiverToxOverrideRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/livertox/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionLiverToxUpdateJobStatus(
  jobId: string,
  timeoutSeconds: number = INSPECTION_JOB_STATUS_TIMEOUT_SECONDS,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/livertox/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
  );
}

export async function cancelInspectionLiverToxUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/livertox/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionRagUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(
    `${API_BASE_URL}/inspection/rag/update-config`,
    { method: "GET" },
  );
}

export async function startInspectionRagUpdateJob(
  payload: InspectionRagUpdateRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/rag/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionRagUpdateJobStatus(
  jobId: string,
  timeoutSeconds: number = INSPECTION_JOB_STATUS_TIMEOUT_SECONDS,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/rag/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
  );
}

export async function cancelInspectionRagUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/rag/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}
