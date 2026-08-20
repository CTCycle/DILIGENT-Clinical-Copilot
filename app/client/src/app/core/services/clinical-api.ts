import { API_BASE_URL } from "../constants";
import {
  ClinicalInputPreflightResponse,
  ClinicalSectionTemplateResponse,
  ClinicalRequestPayload,
  JobCancelResponse,
  JobStartResponse,
  JobStatusResponse,
} from "../models/types";
import {
  HTTP_TIMEOUT,
  requestJson,
} from "./http-api";

export async function startClinicalJob(
  payload: ClinicalRequestPayload,
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/clinical/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchClinicalJobStatus(
  jobId: string,
  requestId: string,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<JobStatusResponse> {
  const query = new URLSearchParams({ _: requestId }).toString();
  return requestJson<JobStatusResponse>(
    `${API_BASE_URL}/clinical/jobs/${encodeURIComponent(jobId)}?${query}`,
    {
      method: "GET",
      cache: "no-store",
      headers: {
        "Cache-Control": "no-cache, no-store, max-age=0",
        Pragma: "no-cache",
      },
    },
    timeoutSeconds,
  );
}

export async function validateClinicalInput(
  payload: ClinicalRequestPayload,
): Promise<ClinicalInputPreflightResponse> {
  return requestJson<ClinicalInputPreflightResponse>(
    `${API_BASE_URL}/clinical/validate-input`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}

export async function fetchClinicalSectionTemplate(): Promise<ClinicalSectionTemplateResponse> {
  return requestJson<ClinicalSectionTemplateResponse>(
    `${API_BASE_URL}/clinical/section-template`,
    { method: "GET" },
  );
}

export async function cancelClinicalJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/clinical/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export function resolvePollIntervalMs(pollIntervalSeconds: number): number {
  if (!Number.isFinite(pollIntervalSeconds) || pollIntervalSeconds <= 0) {
    return 1000;
  }
  return Math.max(250, Math.round(pollIntervalSeconds * 1000));
}
