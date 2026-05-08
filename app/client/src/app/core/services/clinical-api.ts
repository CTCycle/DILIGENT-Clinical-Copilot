import { API_BASE_URL } from "../constants";
import {
  ApiResult,
  ClinicalRequestPayload,
  JobCancelResponse,
  JobStartResponse,
  JobStatusResponse,
} from "../models/types";
import {
  fetchWithTimeout,
  GENERIC_REQUEST_ERROR,
  HTTP_TIMEOUT,
  normalizeThrownError,
  parseApiResponse,
  requestJson,
} from "./http-api";

export async function runClinicalSession(
  payload: ClinicalRequestPayload,
): Promise<ApiResult> {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/clinical`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    return await parseApiResponse(response);
  } catch (error) {
    return {
      message: normalizeThrownError(error, GENERIC_REQUEST_ERROR),
      json: null,
    };
  }
}

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
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(
    `${API_BASE_URL}/clinical/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
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

export function pollClinicalJobStatus(
  jobId: string,
  intervalMs: number,
  onUpdate: (status: JobStatusResponse) => void,
  onError: (message: string) => void,
): { stop: () => void } {
  const safeIntervalMs = Math.max(intervalMs, 250);
  const requestTimeoutSeconds = Math.min(
    30,
    Math.max(5, Math.ceil((safeIntervalMs / 1000) * 4)),
  );
  const maxConsecutivePollErrors = 3;
  let timeoutId: ReturnType<typeof globalThis.setTimeout> | null = null;
  let stopped = false;
  let consecutivePollErrors = 0;

  const poll = async () => {
    if (stopped) return;
    try {
      const status = await fetchClinicalJobStatus(jobId, requestTimeoutSeconds);
      if (stopped) return;
      consecutivePollErrors = 0;
      onUpdate(status);
      if (
        status.status === "completed" ||
        status.status === "failed" ||
        status.status === "cancelled"
      ) {
        return;
      }
    } catch (error) {
      if (stopped) return;
      consecutivePollErrors += 1;
      if (consecutivePollErrors < maxConsecutivePollErrors) {
        timeoutId = globalThis.setTimeout(poll, safeIntervalMs);
        return;
      }
      const message =
        normalizeThrownError(
          error,
          "[ERROR] Polling could not continue. Please retry.",
        );
      onError(
        `Polling failed after ${maxConsecutivePollErrors} attempts. ${message}`,
      );
      return;
    }
    if (stopped) return;
    timeoutId = globalThis.setTimeout(poll, safeIntervalMs);
  };

  poll();

  return {
    stop: () => {
      stopped = true;
      if (timeoutId !== null) {
        globalThis.clearTimeout(timeoutId);
      }
    },
  };
}
