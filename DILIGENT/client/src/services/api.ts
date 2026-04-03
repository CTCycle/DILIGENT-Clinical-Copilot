import { API_BASE_URL, HTTP_TIMEOUT_SECONDS } from "../constants";
import {
  AccessKeyProvider,
  AccessKeyRecord,
  ApiResult,
  ClinicalRequestPayload,
  InspectionCatalogQuery,
  InspectionDeleteResponse,
  InspectionLiverToxOverrideRequest,
  InspectionDrugAliasesResponse,
  InspectionLiverToxCatalogResponse,
  InspectionLiverToxExcerptResponse,
  InspectionRagDocumentsResponse,
  InspectionRagOverrideRequest,
  InspectionRagVectorStoreSummary,
  InspectionRxNavOverrideRequest,
  InspectionUpdateConfigResponse,
  InspectionRxNavCatalogResponse,
  InspectionSessionCatalogResponse,
  InspectionSessionQuery,
  InspectionUpdateJobStatusResponse,
  InspectionSessionReportResponse,
  JobCancelResponse,
  JobStartResponse,
  JobStatusResponse,
  ModelConfigStateResponse,
  ModelConfigUpdateRequest,
  OllamaPullJobResult,
} from "../types";

const HTTP_TIMEOUT =
  Number.parseFloat(import.meta.env.VITE_HTTP_TIMEOUT ?? "") ||
  HTTP_TIMEOUT_SECONDS;
const GENERIC_REQUEST_ERROR = "[ERROR] Request could not be completed. Please try again.";
const TIMEOUT_REQUEST_ERROR = "[ERROR] Request timed out. Please retry.";
const NETWORK_REQUEST_ERROR =
  "[ERROR] Network is unavailable. Check your connection and retry.";

const SENSITIVE_ERROR_FRAGMENTS = [
  "traceback",
  "stack",
  "exception",
  "token",
  "secret",
  "password",
  "authorization",
  "api key",
  "access key",
  "sql",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafeUserFacingMessage(message: string): boolean {
  const candidate = message.trim();
  if (!candidate || candidate.length > 180) {
    return false;
  }
  const lowered = candidate.toLowerCase();
  return !SENSITIVE_ERROR_FRAGMENTS.some((fragment) => lowered.includes(fragment));
}

function extractServerDetail(data: unknown): string {
  if (!isRecord(data)) {
    return "";
  }
  const detail = data.detail;
  if (typeof detail === "string") {
    return detail.trim();
  }
  if (Array.isArray(detail)) {
    const firstIssue = detail.find((item) => isRecord(item));
    if (firstIssue && typeof firstIssue.msg === "string") {
      return firstIssue.msg.trim();
    }
  }
  return "";
}

function buildHttpErrorMessage(statusCode: number, detail: string): string {
  if (statusCode === 404) {
    return "[ERROR] Requested data was not found.";
  }
  if (statusCode === 408 || statusCode === 504) {
    return TIMEOUT_REQUEST_ERROR;
  }
  if (statusCode === 409) {
    return "[ERROR] Another operation is already running. Please wait and retry.";
  }
  if (statusCode === 422 || statusCode === 400) {
    if (isSafeUserFacingMessage(detail)) {
      return `[ERROR] ${detail}`;
    }
    return "[ERROR] Some inputs are invalid. Review the data and retry.";
  }
  if (statusCode === 429) {
    return "[ERROR] Service is busy. Wait a moment and retry.";
  }
  if (statusCode === 502 || statusCode === 503) {
    return "[ERROR] A required service is unavailable. Please retry shortly.";
  }
  if (statusCode >= 500) {
    return "[ERROR] Service is temporarily unavailable. Please retry.";
  }
  if (isSafeUserFacingMessage(detail)) {
    return `[ERROR] ${detail}`;
  }
  return GENERIC_REQUEST_ERROR;
}

function normalizeThrownError(error: unknown, fallback: string): string {
  if (error instanceof DOMException && error.name === "AbortError") {
    return TIMEOUT_REQUEST_ERROR;
  }
  if (error instanceof TypeError) {
    return NETWORK_REQUEST_ERROR;
  }
  if (error instanceof Error && isSafeUserFacingMessage(error.message)) {
    return error.message.startsWith("[ERROR]")
      ? error.message
      : `[ERROR] ${error.message}`;
  }
  return fallback;
}

function extractTextFromResult(data: unknown): string {
  if (!data) {
    return "";
  }

  if (typeof data === "string") {
    return data;
  }

  if (Array.isArray(data)) {
    try {
      return `\`\`\`json\n${JSON.stringify(data, null, 2)}\n\`\`\``;
    } catch {
      return `${data}`;
    }
  }

  if (isRecord(data)) {
    const candidates = ["output", "result", "text", "message", "response", "detail", "error"];
    for (const key of candidates) {
      const value = data[key];
      if (typeof value === "string" && value.trim()) {
        return value;
      }
    }
    try {
      return `\`\`\`json\n${JSON.stringify(data, null, 2)}\n\`\`\``;
    } catch {
      return `${data}`;
    }
  }

  return `${data}`;
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutMs = Math.max(timeoutSeconds, 1) * 1000;
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    throw new Error(normalizeThrownError(error, GENERIC_REQUEST_ERROR));
  } finally {
    globalThis.clearTimeout(timer);
  }
}

async function parseApiResponse(
  response: Response,
): Promise<ApiResult> {
  const contentType = response.headers.get("content-type") || "";
  const bodyText = await response.text();
  let parsed: unknown;
  let hasParsed = false;

  if (bodyText) {
    const shouldTryJson =
      contentType.includes("application/json") ||
      bodyText.trimStart().startsWith("{") ||
      bodyText.trimStart().startsWith("[");
    if (shouldTryJson) {
      try {
        parsed = JSON.parse(bodyText);
        hasParsed = true;
      } catch {
        hasParsed = false;
      }
    }
  }

  if (hasParsed) {
    const message = extractTextFromResult(parsed);
    const jsonPayload =
      typeof parsed === "object" || Array.isArray(parsed) ? parsed : null;

    if (response.ok) {
      return { message, json: jsonPayload };
    }

    const detail = extractServerDetail(parsed) || message;
    return {
      message: buildHttpErrorMessage(response.status, detail),
      json: jsonPayload,
    };
  }

  if (response.ok) {
    return { message: bodyText.trim(), json: null };
  }

  return {
    message: buildHttpErrorMessage(response.status, bodyText.trim()),
    json: null,
  };
}

async function requestJson<T>(
  url: string,
  options: RequestInit,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<T> {
  const response = await fetchWithTimeout(url, options, timeoutSeconds);
  const result = await parseApiResponse(response);
  if (!response.ok) {
    throw new Error(result.message || GENERIC_REQUEST_ERROR);
  }
  if (!result.json) {
    throw new Error("[ERROR] Service returned an invalid response. Please retry.");
  }
  return result.json as T;
}

function buildQueryString(
  params: Record<string, string | number | null | undefined>,
): string {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) {
      continue;
    }
    const serialized = `${value}`.trim();
    if (!serialized) {
      continue;
    }
    searchParams.set(key, serialized);
  }
  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : "";
}

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

export async function fetchInspectionSessions(
  query: InspectionSessionQuery,
): Promise<InspectionSessionCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    status: query.status,
    date_mode: query.date_mode,
    date: query.date,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionSessionCatalogResponse>(
    `${API_BASE_URL}/inspection/sessions${queryString}`,
    { method: "GET" },
  );
}

export async function fetchInspectionSessionReport(
  sessionId: number,
): Promise<InspectionSessionReportResponse> {
  return requestJson<InspectionSessionReportResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/report`,
    { method: "GET" },
  );
}

export async function deleteInspectionSession(
  sessionId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionRxNavCatalog(
  query: InspectionCatalogQuery,
): Promise<InspectionRxNavCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionRxNavCatalogResponse>(
    `${API_BASE_URL}/inspection/rxnav${queryString}`,
    { method: "GET" },
  );
}

export async function fetchInspectionRxNavAliases(
  drugId: number,
): Promise<InspectionDrugAliasesResponse> {
  return requestJson<InspectionDrugAliasesResponse>(
    `${API_BASE_URL}/inspection/rxnav/${encodeURIComponent(String(drugId))}/aliases`,
    { method: "GET" },
  );
}

export async function deleteInspectionRxNavDrug(
  drugId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${API_BASE_URL}/inspection/rxnav/${encodeURIComponent(String(drugId))}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionRxNavUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(`${API_BASE_URL}/inspection/rxnav/update-config`, {
    method: "GET",
  });
}

export async function startInspectionRxNavUpdateJob(
  payload: InspectionRxNavOverrideRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/rxnav/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionRxNavUpdateJobStatus(
  jobId: string,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/rxnav/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
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

export async function fetchInspectionLiverToxCatalog(
  query: InspectionCatalogQuery,
): Promise<InspectionLiverToxCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionLiverToxCatalogResponse>(
    `${API_BASE_URL}/inspection/livertox${queryString}`,
    { method: "GET" },
  );
}

export async function fetchInspectionLiverToxExcerpt(
  drugId: number,
): Promise<InspectionLiverToxExcerptResponse> {
  return requestJson<InspectionLiverToxExcerptResponse>(
    `${API_BASE_URL}/inspection/livertox/${encodeURIComponent(String(drugId))}/excerpt`,
    { method: "GET" },
  );
}

export async function deleteInspectionLiverToxDrug(
  drugId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${API_BASE_URL}/inspection/livertox/${encodeURIComponent(String(drugId))}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionLiverToxUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(`${API_BASE_URL}/inspection/livertox/update-config`, {
    method: "GET",
  });
}

export async function startInspectionLiverToxUpdateJob(
  payload: InspectionLiverToxOverrideRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/livertox/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionLiverToxUpdateJobStatus(
  jobId: string,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/livertox/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
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
  return requestJson<InspectionUpdateConfigResponse>(`${API_BASE_URL}/inspection/rag/update-config`, {
    method: "GET",
  });
}

export async function fetchInspectionRagDocuments(): Promise<InspectionRagDocumentsResponse> {
  return requestJson<InspectionRagDocumentsResponse>(`${API_BASE_URL}/inspection/rag/documents`, {
    method: "GET",
  });
}

export async function fetchInspectionRagVectorStore(): Promise<InspectionRagVectorStoreSummary> {
  return requestJson<InspectionRagVectorStoreSummary>(`${API_BASE_URL}/inspection/rag/vector-store`, {
    method: "GET",
  });
}

export async function startInspectionRagUpdateJob(
  payload: InspectionRagOverrideRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/rag/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionRagUpdateJobStatus(
  jobId: string,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/rag/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
  );
}

export async function cancelInspectionRagUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/rag/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: "POST" },
  );
}

export async function fetchModelConfigState(): Promise<ModelConfigStateResponse> {
  return requestJson<ModelConfigStateResponse>(`${API_BASE_URL}/model-config`, {
    method: "GET",
  });
}

export async function updateModelConfigState(
  payload: ModelConfigUpdateRequest,
): Promise<ModelConfigStateResponse> {
  return requestJson<ModelConfigStateResponse>(`${API_BASE_URL}/model-config`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function startModelPullJob(
  modelName: string,
): Promise<JobStartResponse> {
  const candidate = modelName.trim();
  if (!candidate) {
    throw new Error("[ERROR] Enter a model name to pull from Ollama.");
  }
  const queryString = buildQueryString({ name: candidate, stream: "true" });
  return requestJson<JobStartResponse>(
    `${API_BASE_URL}/models/pull/jobs${queryString}`,
    { method: "POST" },
  );
}

export async function fetchModelPullJobStatus(
  jobId: string,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<JobStatusResponse<OllamaPullJobResult>> {
  return requestJson<JobStatusResponse<OllamaPullJobResult>>(
    `${API_BASE_URL}/models/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
  );
}

export async function pullModels(models: string[]): Promise<ApiResult> {
  const selected = Array.from(new Set(models.filter((item) => !!item)));
  if (!selected.length) {
    return { message: "[ERROR] No models selected to pull.", json: null };
  }

  try {
    for (const model of selected) {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/models/pull?name=${encodeURIComponent(
          model,
        )}&stream=false`,
        { method: "GET" },
      );

      if (!response.ok) {
        return await parseApiResponse(
          response,
        );
      }
    }
  } catch (error) {
    return {
      message: normalizeThrownError(error, GENERIC_REQUEST_ERROR),
      json: null,
    };
  }

  return {
    message: `[INFO] Models available locally: ${selected.join(", ")}.`,
    json: null,
  };
}

export async function fetchAccessKeys(
  provider: AccessKeyProvider,
): Promise<AccessKeyRecord[]> {
  const encodedProvider = encodeURIComponent(provider);
  return requestJson<AccessKeyRecord[]>(
    `${API_BASE_URL}/access-keys?provider=${encodedProvider}`,
    {
      method: "GET",
    },
  );
}

export async function createAccessKey(
  provider: AccessKeyProvider,
  accessKey: string,
): Promise<AccessKeyRecord> {
  return requestJson<AccessKeyRecord>(`${API_BASE_URL}/access-keys`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ provider, access_key: accessKey }),
  });
}

export async function activateAccessKey(
  id: number,
  provider?: AccessKeyProvider,
): Promise<AccessKeyRecord> {
  const providerQuery = provider
    ? `?provider=${encodeURIComponent(provider)}`
    : "";
  return requestJson<AccessKeyRecord>(
    `${API_BASE_URL}/access-keys/${encodeURIComponent(String(id))}/activate${providerQuery}`,
    {
      method: "PUT",
    },
  );
}

export async function deleteAccessKey(
  id: number,
  provider?: AccessKeyProvider,
): Promise<void> {
  const providerQuery = provider
    ? `?provider=${encodeURIComponent(provider)}`
    : "";
  await requestJson<{ status: string; deleted: boolean }>(
    `${API_BASE_URL}/access-keys/${encodeURIComponent(String(id))}${providerQuery}`,
    {
      method: "DELETE",
    },
  );
}
